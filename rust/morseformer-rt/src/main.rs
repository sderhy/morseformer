use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ndarray::{s, Array, ArrayD, Axis, IxDyn};
use ndarray_npy::read_npy;
use ort::session::Session;
use ort::{inputs, value::TensorRef};
use serde::Deserialize;

#[derive(Parser)]
#[command(version, about = "Minimal morseformer ONNX runtime prototype")]
struct Cli {
    #[arg(long, default_value = "../../build/onnx/rnnt_phase11b")]
    onnx_dir: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Load the manifest and ONNX sessions, then print graph metadata.
    Info,
    /// Run encoder, predictor_step, and joint once with zero tensors.
    Smoke,
    /// Run a minimal greedy RNN-T decode on zero feature frames.
    GreedyZero {
        #[arg(long)]
        frames: Option<usize>,
        #[arg(long, default_value_t = 5)]
        max_emit_per_frame: usize,
    },
    /// Decode precomputed features from a .npy file.
    DecodeFeatures {
        features: PathBuf,
        #[arg(long, default_value_t = 5)]
        max_emit_per_frame: usize,
    },
}

#[derive(Debug, Deserialize)]
struct Manifest {
    format: String,
    opset: u32,
    graphs: Graphs,
    model: ModelManifest,
    runtime: RuntimeManifest,
}

#[derive(Debug, Deserialize)]
struct Graphs {
    encoder: String,
    predictor_step: String,
    joint: String,
}

#[derive(Debug, Deserialize)]
struct ModelManifest {
    input_dim: usize,
    d_model: usize,
    d_pred: usize,
    pred_lstm_layers: usize,
    d_joint: usize,
    vocab_size: usize,
    blank_index: usize,
}

#[derive(Debug, Deserialize)]
struct RuntimeManifest {
    sample_frames: usize,
    subsample: usize,
}

struct Runtime {
    manifest: Manifest,
    encoder: Session,
    predictor_step: Session,
    joint: Session,
}

impl Runtime {
    fn load(onnx_dir: PathBuf) -> Result<Self> {
        let manifest_path = onnx_dir.join("manifest.json");
        let manifest_text = fs::read_to_string(&manifest_path)
            .with_context(|| format!("reading {}", manifest_path.display()))?;
        let manifest: Manifest = serde_json::from_str(&manifest_text)
            .with_context(|| format!("parsing {}", manifest_path.display()))?;

        let encoder = load_session(onnx_dir.join(&manifest.graphs.encoder))?;
        let predictor_step = load_session(onnx_dir.join(&manifest.graphs.predictor_step))?;
        let joint = load_session(onnx_dir.join(&manifest.graphs.joint))?;

        Ok(Self {
            manifest,
            encoder,
            predictor_step,
            joint,
        })
    }

    fn print_info(&self) {
        println!("format: {}", self.manifest.format);
        println!("opset: {}", self.manifest.opset);
        println!(
            "model: input_dim={} d_model={} d_pred={} d_joint={} pred_layers={} vocab={} blank={}",
            self.manifest.model.input_dim,
            self.manifest.model.d_model,
            self.manifest.model.d_pred,
            self.manifest.model.d_joint,
            self.manifest.model.pred_lstm_layers,
            self.manifest.model.vocab_size,
            self.manifest.model.blank_index,
        );
        println!(
            "runtime: sample_frames={} subsample={}",
            self.manifest.runtime.sample_frames, self.manifest.runtime.subsample
        );
        print_session("encoder", &self.encoder);
        print_session("predictor_step", &self.predictor_step);
        print_session("joint", &self.joint);
    }

    fn smoke(mut self) -> Result<()> {
        let m = &self.manifest.model;
        let frames = self.manifest.runtime.sample_frames;

        let features = Array::<f32, _>::zeros(IxDyn(&[1, frames, m.input_dim]));
        let lengths = Array::<i64, _>::from_vec(vec![frames as i64]);
        let encoder_outputs = self.encoder.run(inputs![
            "features" => TensorRef::from_array_view(&features)?,
            "lengths" => TensorRef::from_array_view(&lengths)?,
        ])?;
        let enc_out = encoder_outputs["enc_out"].try_extract_array::<f32>()?;
        let enc_lengths = encoder_outputs["enc_lengths"].try_extract_array::<i64>()?;
        let ctc_log_probs = encoder_outputs["ctc_log_probs"].try_extract_array::<f32>()?;
        println!("encoder.enc_out shape: {:?}", enc_out.shape());
        println!("encoder.enc_lengths: {:?}", enc_lengths);
        println!("encoder.ctc_log_probs shape: {:?}", ctc_log_probs.shape());

        let token = Array::<i64, _>::from_shape_vec(IxDyn(&[1, 1]), vec![m.blank_index as i64])?;
        let h_in = Array::<f32, _>::zeros(IxDyn(&[m.pred_lstm_layers, 1, m.d_pred]));
        let c_in = Array::<f32, _>::zeros(IxDyn(&[m.pred_lstm_layers, 1, m.d_pred]));
        let predictor_outputs = self.predictor_step.run(inputs![
            "token" => TensorRef::from_array_view(&token)?,
            "h_in" => TensorRef::from_array_view(&h_in)?,
            "c_in" => TensorRef::from_array_view(&c_in)?,
        ])?;
        let pred_out = predictor_outputs["pred_out"].try_extract_array::<f32>()?;
        let h_out = predictor_outputs["h_out"].try_extract_array::<f32>()?;
        let c_out = predictor_outputs["c_out"].try_extract_array::<f32>()?;
        println!("predictor.pred_out shape: {:?}", pred_out.shape());
        println!("predictor.h_out shape: {:?}", h_out.shape());
        println!("predictor.c_out shape: {:?}", c_out.shape());

        let enc_frame = enc_out.slice(ndarray::s![.., 0..1, ..]).to_owned();
        let pred_frame = pred_out.to_owned();
        let joint_outputs = self.joint.run(inputs![
            "enc_frame" => TensorRef::from_array_view(&enc_frame)?,
            "pred_out" => TensorRef::from_array_view(&pred_frame)?,
        ])?;
        let logits = joint_outputs["logits"].try_extract_array::<f32>()?;
        println!("joint.logits shape: {:?}", logits.shape());
        println!("smoke passed");
        Ok(())
    }

    fn greedy_zero(self, frames: usize, max_emit_per_frame: usize) -> Result<()> {
        let m = &self.manifest.model;
        let features = Array::<f32, _>::zeros(IxDyn(&[1, frames, m.input_dim]));
        self.greedy_decode(features, max_emit_per_frame)
    }

    fn decode_features(self, path: PathBuf, max_emit_per_frame: usize) -> Result<()> {
        let features: ArrayD<f32> =
            read_npy(&path).with_context(|| format!("reading {}", path.display()))?;
        let features = match features.ndim() {
            2 => features.insert_axis(Axis(0)),
            3 => features,
            ndim => {
                anyhow::bail!(
                    "features must have shape [frames,input_dim] or [1,frames,input_dim], got {ndim}D"
                )
            }
        };
        self.greedy_decode(features, max_emit_per_frame)
    }

    fn greedy_decode(mut self, features: ArrayD<f32>, max_emit_per_frame: usize) -> Result<()> {
        let m = &self.manifest.model;
        let input_dim = m.input_dim;
        let pred_lstm_layers = m.pred_lstm_layers;
        let d_pred = m.d_pred;
        let blank_index = m.blank_index;
        let shape = features.shape();
        if shape.len() != 3 || shape[0] != 1 || shape[2] != input_dim {
            anyhow::bail!(
                "features must have shape [1,frames,{input_dim}], got {:?}",
                shape
            );
        }
        let frames = shape[1];
        let lengths = Array::<i64, _>::from_vec(vec![frames as i64]);

        let (enc_out, enc_lengths) = {
            let outputs = self.encoder.run(inputs![
                "features" => TensorRef::from_array_view(&features)?,
                "lengths" => TensorRef::from_array_view(&lengths)?,
            ])?;
            (
                outputs["enc_out"].try_extract_array::<f32>()?.to_owned(),
                outputs["enc_lengths"]
                    .try_extract_array::<i64>()?
                    .to_owned(),
            )
        };
        let enc_len = enc_lengths
            .as_slice()
            .and_then(|v| v.first().copied())
            .context("encoder did not return enc_lengths[0]")? as usize;

        let mut token = Array::<i64, _>::from_shape_vec(IxDyn(&[1, 1]), vec![blank_index as i64])?;
        let mut h = Array::<f32, _>::zeros(IxDyn(&[pred_lstm_layers, 1, d_pred]));
        let mut c = Array::<f32, _>::zeros(IxDyn(&[pred_lstm_layers, 1, d_pred]));
        let (mut pred_out, h_next, c_next) = self.predictor_step(&token, &h, &c)?;
        h = h_next;
        c = c_next;

        let mut tokens = Vec::new();
        for frame_idx in 0..enc_len {
            let enc_frame = enc_out
                .slice(s![.., frame_idx..frame_idx + 1, ..])
                .to_owned()
                .into_dyn();
            let mut emitted = 0;
            while emitted < max_emit_per_frame {
                let logits = self.joint_logits(&enc_frame, &pred_out)?;
                let tok = argmax(
                    logits
                        .as_slice()
                        .context("joint logits are not contiguous")?,
                );
                if tok == blank_index {
                    break;
                }
                tokens.push(tok);
                token = Array::<i64, _>::from_shape_vec(IxDyn(&[1, 1]), vec![tok as i64])?;
                let (next_pred, next_h, next_c) = self.predictor_step(&token, &h, &c)?;
                pred_out = next_pred;
                h = next_h;
                c = next_c;
                emitted += 1;
            }
        }

        println!("frames: {frames}");
        println!("encoder frames: {enc_len}");
        println!("tokens: {:?}", tokens);
        println!("text: {}", decode_tokens(&tokens));
        Ok(())
    }

    fn predictor_step(
        &mut self,
        token: &ArrayD<i64>,
        h: &ArrayD<f32>,
        c: &ArrayD<f32>,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>, ArrayD<f32>)> {
        let outputs = self.predictor_step.run(inputs![
            "token" => TensorRef::from_array_view(token)?,
            "h_in" => TensorRef::from_array_view(h)?,
            "c_in" => TensorRef::from_array_view(c)?,
        ])?;
        Ok((
            outputs["pred_out"].try_extract_array::<f32>()?.to_owned(),
            outputs["h_out"].try_extract_array::<f32>()?.to_owned(),
            outputs["c_out"].try_extract_array::<f32>()?.to_owned(),
        ))
    }

    fn joint_logits(
        &mut self,
        enc_frame: &ArrayD<f32>,
        pred_out: &ArrayD<f32>,
    ) -> Result<ArrayD<f32>> {
        let outputs = self.joint.run(inputs![
            "enc_frame" => TensorRef::from_array_view(enc_frame)?,
            "pred_out" => TensorRef::from_array_view(pred_out)?,
        ])?;
        Ok(outputs["logits"].try_extract_array::<f32>()?.to_owned())
    }
}

fn load_session(path: PathBuf) -> Result<Session> {
    Session::builder()
        .context("creating ONNX Runtime session builder")?
        .commit_from_file(&path)
        .with_context(|| format!("loading ONNX graph {}", path.display()))
}

fn print_session(name: &str, session: &Session) {
    println!("{name}:");
    println!("  inputs:");
    for input in session.inputs() {
        println!("    {}", input.name());
    }
    println!("  outputs:");
    for output in session.outputs() {
        println!("    {}", output.name());
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_value = f32::NEG_INFINITY;
    for (idx, value) in values.iter().copied().enumerate() {
        if value > best_value {
            best_idx = idx;
            best_value = value;
        }
    }
    best_idx
}

fn decode_tokens(tokens: &[usize]) -> String {
    const VOCAB: [&str; 49] = [
        "<blank>", " ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
        "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6",
        "7", "8", "9", ".", ",", "?", "!", "/", "=", "+", "-", "É", "À", "'",
    ];
    let mut out = String::new();
    for token in tokens {
        if *token == 0 {
            continue;
        }
        if let Some(ch) = VOCAB.get(*token) {
            out.push_str(ch);
        }
    }
    out.trim().to_string()
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let runtime = Runtime::load(cli.onnx_dir)?;

    match cli.command {
        Command::Info => runtime.print_info(),
        Command::Smoke => runtime.smoke()?,
        Command::GreedyZero {
            frames,
            max_emit_per_frame,
        } => {
            let frames = frames.unwrap_or(runtime.manifest.runtime.sample_frames);
            runtime.greedy_zero(frames, max_emit_per_frame)?;
        }
        Command::DecodeFeatures {
            features,
            max_emit_per_frame,
        } => runtime.decode_features(features, max_emit_per_frame)?,
    }
    Ok(())
}
