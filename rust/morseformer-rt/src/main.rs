use std::{f32::consts::PI, fs, path::PathBuf};

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
    /// Decode a WAV file directly with the Rust frontend.
    DecodeWav {
        wav: PathBuf,
        #[arg(long, default_value_t = 600.0)]
        freq: f32,
        #[arg(long, default_value_t = 100.0)]
        bandwidth: f32,
        #[arg(long, default_value_t = 8000)]
        target_sample_rate: usize,
        #[arg(long, default_value_t = 500)]
        frame_rate: usize,
        #[arg(long, default_value_t = 6.0)]
        window_seconds: f32,
        #[arg(long, default_value_t = 2.0)]
        hop_seconds: f32,
        #[arg(long, default_value_t = false)]
        no_windowing: bool,
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

struct DecodeResult {
    frames: usize,
    enc_frames: usize,
    aligned: Vec<(usize, usize)>,
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

    fn decode_wav(
        self,
        path: PathBuf,
        freq: f32,
        bandwidth: f32,
        target_sample_rate: usize,
        frame_rate: usize,
        window_seconds: f32,
        hop_seconds: f32,
        no_windowing: bool,
        max_emit_per_frame: usize,
    ) -> Result<()> {
        let (audio, sample_rate) = load_wav_mono(&path)?;
        let audio = if sample_rate == target_sample_rate {
            audio
        } else {
            resample_linear(&audio, sample_rate, target_sample_rate)
        };
        let features =
            extract_cw_features(&audio, target_sample_rate, freq, bandwidth, frame_rate)?;
        if no_windowing || window_seconds <= 0.0 {
            self.greedy_decode(features.insert_axis(Axis(0)), max_emit_per_frame)
        } else {
            self.decode_feature_windows(
                features,
                frame_rate,
                window_seconds,
                hop_seconds,
                max_emit_per_frame,
            )
        }
    }

    fn greedy_decode(mut self, features: ArrayD<f32>, max_emit_per_frame: usize) -> Result<()> {
        let result = self.greedy_decode_result(features, max_emit_per_frame)?;
        print_decode_result(&result);
        Ok(())
    }

    fn decode_feature_windows(
        mut self,
        features: ArrayD<f32>,
        frame_rate: usize,
        window_seconds: f32,
        hop_seconds: f32,
        max_emit_per_frame: usize,
    ) -> Result<()> {
        if features.ndim() != 2 {
            anyhow::bail!("windowed decode expects [frames,input_dim] features");
        }
        if window_seconds <= 0.0 || hop_seconds <= 0.0 || hop_seconds > window_seconds {
            anyhow::bail!(
                "invalid windowing: window_seconds={window_seconds} hop_seconds={hop_seconds}"
            );
        }
        let n_frames = features.shape()[0];
        let input_dim = self.manifest.model.input_dim;
        let window_frames = (window_seconds * frame_rate as f32).round() as usize;
        let hop_frames = (hop_seconds * frame_rate as f32).round() as usize;
        if window_frames == 0 || hop_frames == 0 {
            anyhow::bail!("window/hop produced zero frames");
        }

        let mut window_start = 0usize;
        let mut committed_until = 0usize;
        let mut tokens = Vec::new();
        let mut chunks = 0usize;

        loop {
            if window_start >= n_frames {
                break;
            }
            let window_end = (window_start + window_frames).min(n_frames);
            let is_first = window_start == 0;
            let is_final = window_end == n_frames;
            let window = features
                .slice(s![window_start..window_end, ..])
                .to_owned()
                .into_dyn()
                .insert_axis(Axis(0));
            let result = self.greedy_decode_result(window, max_emit_per_frame)?;
            let (commit_lo, commit_hi) = commit_zone_frames(
                window_start,
                window_end - window_start,
                window_frames,
                hop_frames,
                committed_until,
                is_first,
                is_final,
            );
            for (tok, enc_frame_idx) in result.aligned {
                let abs_feature_frame =
                    window_start + enc_frame_idx * self.manifest.runtime.subsample;
                if abs_feature_frame >= commit_lo && abs_feature_frame < commit_hi {
                    tokens.push(tok);
                }
            }
            committed_until = commit_hi;
            chunks += 1;
            if is_final {
                break;
            }
            window_start += hop_frames;
        }

        println!("frames: {n_frames}");
        println!("chunks: {chunks}");
        println!("tokens: {:?}", tokens);
        println!("text: {}", decode_tokens(&tokens));
        println!("input_dim: {input_dim}");
        Ok(())
    }

    fn greedy_decode_result(
        &mut self,
        features: ArrayD<f32>,
        max_emit_per_frame: usize,
    ) -> Result<DecodeResult> {
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

        let mut aligned = Vec::new();
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
                aligned.push((tok, frame_idx));
                token = Array::<i64, _>::from_shape_vec(IxDyn(&[1, 1]), vec![tok as i64])?;
                let (next_pred, next_h, next_c) = self.predictor_step(&token, &h, &c)?;
                pred_out = next_pred;
                h = next_h;
                c = next_c;
                emitted += 1;
            }
        }

        Ok(DecodeResult {
            frames,
            enc_frames: enc_len,
            aligned,
        })
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

fn print_decode_result(result: &DecodeResult) {
    let tokens: Vec<usize> = result.aligned.iter().map(|(tok, _frame)| *tok).collect();
    println!("frames: {}", result.frames);
    println!("encoder frames: {}", result.enc_frames);
    println!("tokens: {:?}", tokens);
    println!("text: {}", decode_tokens(&tokens));
}

fn commit_zone_frames(
    window_start: usize,
    window_len: usize,
    window_frames: usize,
    hop_frames: usize,
    committed_until: usize,
    is_first: bool,
    is_final: bool,
) -> (usize, usize) {
    let centre = window_start + window_frames / 2;
    let mut lo = centre.saturating_sub(hop_frames / 2);
    let mut hi = centre + (hop_frames - hop_frames / 2);

    if is_first {
        lo = 0;
    }
    if is_final {
        hi = window_start + window_len;
        if !is_first {
            lo = committed_until;
        }
    }
    if lo < committed_until {
        lo = committed_until;
    }
    if hi < lo {
        hi = lo;
    }
    (lo, hi)
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
        Command::DecodeWav {
            wav,
            freq,
            bandwidth,
            target_sample_rate,
            frame_rate,
            window_seconds,
            hop_seconds,
            no_windowing,
            max_emit_per_frame,
        } => runtime.decode_wav(
            wav,
            freq,
            bandwidth,
            target_sample_rate,
            frame_rate,
            window_seconds,
            hop_seconds,
            no_windowing,
            max_emit_per_frame,
        )?,
    }
    Ok(())
}

fn load_wav_mono(path: &PathBuf) -> Result<(Vec<f32>, usize)> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        anyhow::bail!("{} has zero channels", path.display());
    }

    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .with_context(|| format!("reading float samples from {}", path.display()))?,
        hound::SampleFormat::Int => {
            let scale = if spec.bits_per_sample == 0 {
                anyhow::bail!("{} reports bits_per_sample=0", path.display());
            } else {
                (1_i64 << (spec.bits_per_sample - 1)) as f32
            };
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<Vec<_>, _>>()
                .with_context(|| format!("reading int samples from {}", path.display()))?
        }
    };

    let mono = if channels == 1 {
        samples
    } else {
        samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    Ok((mono, spec.sample_rate as usize))
}

fn extract_cw_features(
    audio: &[f32],
    sample_rate: usize,
    tone_freq: f32,
    bandwidth: f32,
    frame_rate: usize,
) -> Result<ArrayD<f32>> {
    if frame_rate == 0 || sample_rate % frame_rate != 0 {
        anyhow::bail!("sample_rate={sample_rate} must be a multiple of frame_rate={frame_rate}");
    }
    let hop = sample_rate / frame_rate;
    if audio.len() < hop * 2 {
        return Ok(Array::<f32, _>::zeros(IxDyn(&[0, 1])));
    }

    let n_frames = audio.len() / hop;
    let nyquist = sample_rate as f32 / 2.0;
    if !tone_freq.is_finite() || tone_freq <= 0.0 || tone_freq >= nyquist {
        anyhow::bail!("tone_freq={tone_freq} must be in (0, {nyquist}) Hz");
    }
    if !bandwidth.is_finite() || bandwidth <= 0.0 {
        anyhow::bail!("bandwidth={bandwidth} must be positive");
    }

    let half_bandwidth = (bandwidth / 2.0).max(1.0);
    let cutoff = half_bandwidth.min(nyquist - 1.0);
    let rc = 1.0 / (2.0 * PI * cutoff);
    let dt = 1.0 / sample_rate as f32;
    let alpha = dt / (rc + dt);

    let mut i_state = 0.0_f32;
    let mut q_state = 0.0_f32;
    let mut envelope = Vec::with_capacity(audio.len());
    for (idx, sample) in audio.iter().copied().enumerate() {
        let t = idx as f32 / sample_rate as f32;
        let phase = 2.0 * PI * tone_freq * t;
        let i_raw = 2.0 * sample * phase.cos();
        let q_raw = -2.0 * sample * phase.sin();
        i_state += alpha * (i_raw - i_state);
        q_state += alpha * (q_raw - q_state);
        envelope.push((i_state * i_state + q_state * q_state).sqrt());
    }

    let mut features = Vec::with_capacity(n_frames);
    for frame_idx in 0..n_frames {
        let lo = frame_idx * hop;
        let hi = lo + hop;
        let mean_envelope = envelope[lo..hi].iter().copied().sum::<f32>() / hop as f32;
        features.push((mean_envelope + 1e-6).ln());
    }
    normalise(&mut features);
    Array::from_shape_vec(IxDyn(&[n_frames, 1]), features).context("building feature array")
}

fn resample_linear(audio: &[f32], src_rate: usize, dst_rate: usize) -> Vec<f32> {
    if audio.is_empty() || src_rate == dst_rate {
        return audio.to_vec();
    }
    let out_len = ((audio.len() as u128 * dst_rate as u128) / src_rate as u128) as usize;
    let mut out = Vec::with_capacity(out_len);
    let ratio = src_rate as f64 / dst_rate as f64;
    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let lo = src_pos.floor() as usize;
        let frac = (src_pos - lo as f64) as f32;
        let a = audio.get(lo).copied().unwrap_or(0.0);
        let b = audio.get(lo + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }
    out
}

fn normalise(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let var = values
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>()
        / values.len() as f32;
    let std = var.sqrt();
    if std < 1e-8 {
        for value in values {
            *value -= mean;
        }
    } else {
        for value in values {
            *value = (*value - mean) / std;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_tokens_skips_blank_and_maps_vocab() {
        assert_eq!(decode_tokens(&[4, 18, 1, 21, 6, 20, 21]), "CQ TEST");
        assert_eq!(decode_tokens(&[0, 2, 0, 3]), "AB");
    }

    #[test]
    fn commit_zone_tiles_first_middle_and_final_windows() {
        assert_eq!(
            commit_zone_frames(0, 3000, 3000, 1000, 0, true, false),
            (0, 2000)
        );
        assert_eq!(
            commit_zone_frames(1000, 3000, 3000, 1000, 2000, false, false),
            (2000, 3000),
        );
        assert_eq!(
            commit_zone_frames(8000, 1662, 3000, 1000, 9000, false, true),
            (9000, 9662),
        );
    }

    #[test]
    fn linear_resample_preserves_empty_and_identity() {
        let empty: Vec<f32> = vec![];
        assert!(resample_linear(&empty, 48000, 8000).is_empty());
        let audio = vec![0.0, 1.0, 0.0, -1.0];
        assert_eq!(resample_linear(&audio, 8000, 8000), audio);
    }

    #[test]
    fn linear_resample_changes_length_by_rate_ratio() {
        let audio = vec![0.0; 48_000];
        let out = resample_linear(&audio, 48_000, 8_000);
        assert_eq!(out.len(), 8_000);
    }
}
