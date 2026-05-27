use std::{
    collections::VecDeque,
    f32::consts::PI,
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{mpsc, Arc, Mutex},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use serde::Serialize;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AudioDevice {
    id: String,
    name: String,
    default: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DecodeResult {
    text: String,
    frames: usize,
    chunks: usize,
    raw_output: String,
}

#[derive(Default)]
struct AudioStats {
    sum_squares: f64,
    peak: f32,
    samples: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AudioLevel {
    rms: f32,
    peak: f32,
    db: f32,
    samples: usize,
    sample_rate: u32,
    channels: u16,
    signal: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SpectrumSlice {
    min_hz: f32,
    max_hz: f32,
    step_hz: f32,
    sample_rate: u32,
    bins: Vec<f32>,
    peak_hz: f32,
    peak: f32,
}

#[derive(Default)]
struct LiveCaptureState {
    capture: Mutex<Option<LiveCapture>>,
}

struct LiveCapture {
    stop_tx: mpsc::Sender<()>,
    shared: Arc<Mutex<LiveShared>>,
    sample_rate: u32,
    channels: u16,
}

struct LiveShared {
    samples: VecDeque<f32>,
    max_samples: usize,
}

#[tauri::command]
fn list_input_devices() -> Result<Vec<AudioDevice>, String> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|device| device.name().ok());
    let devices = host
        .input_devices()
        .map_err(|err| format!("failed to list input devices: {err}"))?;

    let mut out = Vec::new();
    for (idx, device) in devices.enumerate() {
        let name = device
            .name()
            .unwrap_or_else(|_| format!("Input device {}", idx + 1));
        let is_default = default_name.as_deref() == Some(name.as_str());
        out.push(AudioDevice {
            id: idx.to_string(),
            name,
            default: is_default,
        });
    }
    Ok(out)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct LiveCaptureInfo {
    sample_rate: u32,
    channels: u16,
}

#[tauri::command]
fn start_live_capture(
    state: tauri::State<'_, LiveCaptureState>,
    device_id: String,
) -> Result<LiveCaptureInfo, String> {
    let shared = Arc::new(Mutex::new(LiveShared {
        samples: VecDeque::new(),
        max_samples: 1,
    }));
    let (ready_tx, ready_rx) = mpsc::channel();
    let (stop_tx, stop_rx) = mpsc::channel();
    let thread_shared = Arc::clone(&shared);
    thread::spawn(move || {
        run_live_audio_thread(device_id, thread_shared, stop_rx, ready_tx);
    });

    let info = ready_rx
        .recv_timeout(Duration::from_secs(4))
        .map_err(|_| "live input did not start in time".to_string())??;

    let mut capture = state
        .capture
        .lock()
        .map_err(|_| "live capture lock poisoned".to_string())?;
    *capture = Some(LiveCapture {
        stop_tx,
        shared,
        sample_rate: info.sample_rate,
        channels: info.channels,
    });

    Ok(info)
}

#[tauri::command]
fn stop_live_capture(state: tauri::State<'_, LiveCaptureState>) -> Result<(), String> {
    let mut capture = state
        .capture
        .lock()
        .map_err(|_| "live capture lock poisoned".to_string())?;
    if let Some(capture) = capture.as_ref() {
        let _ = capture.stop_tx.send(());
    }
    *capture = None;
    Ok(())
}

fn run_live_audio_thread(
    device_id: String,
    shared: Arc<Mutex<LiveShared>>,
    stop_rx: mpsc::Receiver<()>,
    ready_tx: mpsc::Sender<Result<LiveCaptureInfo, String>>,
) {
    if let Err(err) = run_live_audio_thread_inner(device_id, shared, stop_rx, ready_tx.clone()) {
        let _ = ready_tx.send(Err(err));
    }
}

fn run_live_audio_thread_inner(
    device_id: String,
    shared: Arc<Mutex<LiveShared>>,
    stop_rx: mpsc::Receiver<()>,
    ready_tx: mpsc::Sender<Result<LiveCaptureInfo, String>>,
) -> Result<(), String> {
    use cpal::traits::{DeviceTrait, StreamTrait};

    let host = cpal::default_host();
    let device = input_device_by_id(&host, &device_id)?;
    let supported_config = device
        .default_input_config()
        .map_err(|err| format!("failed to read input config: {err}"))?;
    let sample_format = supported_config.sample_format();
    let config: cpal::StreamConfig = supported_config.into();
    let sample_rate = config.sample_rate.0;
    let channels = config.channels;
    {
        let mut shared = shared
            .lock()
            .map_err(|_| "live audio buffer lock poisoned".to_string())?;
        shared.max_samples = (sample_rate as usize * channels as usize * 14).max(1);
        shared.samples = VecDeque::with_capacity(shared.max_samples);
    }
    let stream_shared = Arc::clone(&shared);
    let err_fn = |err| eprintln!("live input stream error: {err}");

    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _| push_live_samples(data.iter().copied(), &stream_shared),
            err_fn,
            None,
        ),
        cpal::SampleFormat::F64 => device.build_input_stream(
            &config,
            move |data: &[f64], _| {
                push_live_samples(data.iter().map(|v| *v as f32), &stream_shared)
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I8 => device.build_input_stream(
            &config,
            move |data: &[i8], _| {
                push_live_samples(
                    data.iter().map(|v| *v as f32 / i8::MAX as f32),
                    &stream_shared,
                )
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _| {
                push_live_samples(
                    data.iter().map(|v| *v as f32 / i16::MAX as f32),
                    &stream_shared,
                )
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I32 => device.build_input_stream(
            &config,
            move |data: &[i32], _| {
                push_live_samples(
                    data.iter().map(|v| *v as f32 / i32::MAX as f32),
                    &stream_shared,
                )
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I64 => device.build_input_stream(
            &config,
            move |data: &[i64], _| {
                push_live_samples(
                    data.iter().map(|v| *v as f32 / i64::MAX as f32),
                    &stream_shared,
                )
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U8 => device.build_input_stream(
            &config,
            move |data: &[u8], _| {
                push_live_samples(
                    data.iter().map(|v| (*v as f32 - 128.0) / 128.0),
                    &stream_shared,
                )
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U16 => device.build_input_stream(
            &config,
            move |data: &[u16], _| {
                push_live_samples(
                    data.iter().map(|v| (*v as f32 - 32768.0) / 32768.0),
                    &stream_shared,
                )
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U32 => device.build_input_stream(
            &config,
            move |data: &[u32], _| {
                push_live_samples(
                    data.iter()
                        .map(|v| (*v as f32 - 2_147_483_648.0) / 2_147_483_648.0),
                    &stream_shared,
                )
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U64 => device.build_input_stream(
            &config,
            move |data: &[u64], _| {
                push_live_samples(
                    data.iter().map(|v| {
                        (*v as f64 - 9_223_372_036_854_775_808.0) as f32
                            / 9_223_372_036_854_775_808.0_f32
                    }),
                    &stream_shared,
                )
            },
            err_fn,
            None,
        ),
        _ => {
            return Err(format!(
                "unsupported input sample format: {sample_format:?}"
            ))
        }
    }
    .map_err(|err| format!("failed to open live input stream: {err}"))?;

    stream
        .play()
        .map_err(|err| format!("failed to start live input stream: {err}"))?;

    let _ = ready_tx.send(Ok(LiveCaptureInfo {
        sample_rate,
        channels,
    }));

    loop {
        match stop_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
    }
    drop(stream);

    Ok(())
}

#[tauri::command]
fn live_input_status(state: tauri::State<'_, LiveCaptureState>) -> Result<AudioLevel, String> {
    let capture = state
        .capture
        .lock()
        .map_err(|_| "live capture lock poisoned".to_string())?;
    let capture = capture
        .as_ref()
        .ok_or_else(|| "live capture is not running".to_string())?;
    let samples = recent_live_samples(capture, 0.5)?;
    Ok(audio_level_from_samples(
        &samples,
        capture.sample_rate,
        capture.channels,
    ))
}

#[tauri::command]
fn live_spectrum(state: tauri::State<'_, LiveCaptureState>) -> Result<SpectrumSlice, String> {
    let capture = state
        .capture
        .lock()
        .map_err(|_| "live capture lock poisoned".to_string())?;
    let capture = capture
        .as_ref()
        .ok_or_else(|| "live capture is not running".to_string())?;
    let samples = recent_live_samples(capture, 0.24)?;
    let mono = mono_samples(&samples, capture.channels);
    Ok(narrowband_spectrum(
        &mono,
        capture.sample_rate,
        400.0,
        1000.0,
        10.0,
    ))
}

#[tauri::command]
fn decode_live_window(
    state: tauri::State<'_, LiveCaptureState>,
    onnx_dir: String,
    freq: f32,
    target_sample_rate: usize,
    window_seconds: f32,
) -> Result<DecodeResult, String> {
    let repo_root = repo_root();
    let onnx_dir = resolve_path(&repo_root, &onnx_dir);
    let (samples, sample_rate, channels) = {
        let capture = state
            .capture
            .lock()
            .map_err(|_| "live capture lock poisoned".to_string())?;
        let capture = capture
            .as_ref()
            .ok_or_else(|| "live capture is not running".to_string())?;
        let window_seconds = window_seconds.clamp(1.0, 12.0);
        let samples = recent_live_samples(capture, window_seconds)?;
        let wanted = (capture.sample_rate as f32 * window_seconds).round() as usize
            * capture.channels as usize;
        if samples.len() < wanted / 2 {
            return Err(format!(
                "not enough live audio buffered yet: {:.1}s needed",
                window_seconds
            ));
        }
        (samples, capture.sample_rate, capture.channels)
    };

    let wav_path = write_temp_wav(&samples, sample_rate, channels)?;
    let result = run_runtime_decode_wav(
        &repo_root,
        wav_path.clone(),
        onnx_dir,
        freq,
        target_sample_rate,
        window_seconds,
        window_seconds,
        true,
    );
    let _ = fs::remove_file(wav_path);
    result
}

#[tauri::command]
fn capture_input_level(device_id: String, duration_ms: Option<u64>) -> Result<AudioLevel, String> {
    use cpal::traits::{DeviceTrait, StreamTrait};

    let host = cpal::default_host();
    let device = input_device_by_id(&host, &device_id)?;
    let supported_config = device
        .default_input_config()
        .map_err(|err| format!("failed to read input config: {err}"))?;
    let sample_format = supported_config.sample_format();
    let config: cpal::StreamConfig = supported_config.into();
    let sample_rate = config.sample_rate.0;
    let channels = config.channels;
    let stats = Arc::new(Mutex::new(AudioStats::default()));
    let stream_stats = Arc::clone(&stats);
    let err_fn = |err| eprintln!("input stream error: {err}");

    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _| update_stats(data.iter().copied(), &stream_stats),
            err_fn,
            None,
        ),
        cpal::SampleFormat::F64 => device.build_input_stream(
            &config,
            move |data: &[f64], _| update_stats(data.iter().map(|v| *v as f32), &stream_stats),
            err_fn,
            None,
        ),
        cpal::SampleFormat::I8 => device.build_input_stream(
            &config,
            move |data: &[i8], _| {
                update_stats(
                    data.iter().map(|v| *v as f32 / i8::MAX as f32),
                    &stream_stats,
                )
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _| {
                update_stats(
                    data.iter().map(|v| *v as f32 / i16::MAX as f32),
                    &stream_stats,
                )
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I32 => device.build_input_stream(
            &config,
            move |data: &[i32], _| {
                update_stats(
                    data.iter().map(|v| *v as f32 / i32::MAX as f32),
                    &stream_stats,
                )
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I64 => device.build_input_stream(
            &config,
            move |data: &[i64], _| {
                update_stats(
                    data.iter().map(|v| *v as f32 / i64::MAX as f32),
                    &stream_stats,
                )
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U8 => device.build_input_stream(
            &config,
            move |data: &[u8], _| {
                update_stats(
                    data.iter().map(|v| (*v as f32 - 128.0) / 128.0),
                    &stream_stats,
                )
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U16 => device.build_input_stream(
            &config,
            move |data: &[u16], _| {
                update_stats(
                    data.iter().map(|v| (*v as f32 - 32768.0) / 32768.0),
                    &stream_stats,
                )
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U32 => device.build_input_stream(
            &config,
            move |data: &[u32], _| {
                update_stats(
                    data.iter()
                        .map(|v| (*v as f32 - 2_147_483_648.0) / 2_147_483_648.0),
                    &stream_stats,
                )
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U64 => device.build_input_stream(
            &config,
            move |data: &[u64], _| {
                update_stats(
                    data.iter().map(|v| {
                        (*v as f64 - 9_223_372_036_854_775_808.0) as f32
                            / 9_223_372_036_854_775_808.0_f32
                    }),
                    &stream_stats,
                )
            },
            err_fn,
            None,
        ),
        _ => {
            return Err(format!(
                "unsupported input sample format: {sample_format:?}"
            ))
        }
    }
    .map_err(|err| format!("failed to open input stream: {err}"))?;

    stream
        .play()
        .map_err(|err| format!("failed to start input stream: {err}"))?;
    thread::sleep(Duration::from_millis(
        duration_ms.unwrap_or(250).clamp(100, 1000),
    ));
    drop(stream);

    let stats = stats
        .lock()
        .map_err(|_| "audio stats lock poisoned".to_string())?;
    let rms = if stats.samples > 0 {
        (stats.sum_squares / stats.samples as f64).sqrt() as f32
    } else {
        0.0
    };
    let peak = stats.peak;
    let db = if rms > 0.0 {
        20.0 * rms.log10()
    } else {
        -120.0
    };

    Ok(AudioLevel {
        rms,
        peak,
        db,
        samples: stats.samples,
        sample_rate,
        channels,
        signal: rms > 0.002 || peak > 0.01,
    })
}

#[tauri::command]
fn decode_wav(
    wav_path: String,
    onnx_dir: String,
    freq: f32,
    target_sample_rate: usize,
    window_seconds: f32,
    hop_seconds: f32,
    no_windowing: bool,
) -> Result<DecodeResult, String> {
    if wav_path.trim().is_empty() {
        return Err("WAV path is required".into());
    }
    let repo_root = repo_root();
    let onnx_dir = resolve_path(&repo_root, &onnx_dir);
    let wav_path = resolve_path(&repo_root, &wav_path);
    run_runtime_decode_wav(
        &repo_root,
        wav_path,
        onnx_dir,
        freq,
        target_sample_rate,
        window_seconds,
        hop_seconds,
        no_windowing,
    )
}

fn run_runtime_decode_wav(
    repo_root: &Path,
    wav_path: PathBuf,
    onnx_dir: PathBuf,
    freq: f32,
    target_sample_rate: usize,
    window_seconds: f32,
    hop_seconds: f32,
    no_windowing: bool,
) -> Result<DecodeResult, String> {
    let runtime_dir = repo_root.join("rust").join("morseformer-rt");
    let runtime_bin = runtime_dir
        .join("target")
        .join("debug")
        .join(executable_name("morseformer-rt"));
    if !runtime_bin.exists() {
        return Err(format!(
            "runtime binary not found at {}. Build it with: cd rust/morseformer-rt && cargo build",
            runtime_bin.display()
        ));
    }

    let mut args = vec![
        "--onnx-dir".to_string(),
        onnx_dir.display().to_string(),
        "decode-wav".to_string(),
        wav_path.display().to_string(),
        "--freq".to_string(),
        freq.to_string(),
        "--target-sample-rate".to_string(),
        target_sample_rate.to_string(),
        "--window-seconds".to_string(),
        window_seconds.to_string(),
        "--hop-seconds".to_string(),
        hop_seconds.to_string(),
    ];
    if no_windowing {
        args.push("--no-windowing".to_string());
    }

    let output = Command::new(&runtime_bin)
        .args(args)
        .current_dir(&runtime_dir)
        .output()
        .map_err(|err| format!("failed to run {}: {err}", runtime_bin.display()))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() {
        return Err(format!("{stdout}\n{stderr}"));
    }

    Ok(DecodeResult {
        text: parse_field(&stdout, "text:").unwrap_or_default(),
        frames: parse_field(&stdout, "frames:")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        chunks: parse_field(&stdout, "chunks:")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1),
        raw_output: stdout,
    })
}

fn push_live_samples(samples: impl Iterator<Item = f32>, shared: &Arc<Mutex<LiveShared>>) {
    let mut shared = match shared.lock() {
        Ok(shared) => shared,
        Err(_) => return,
    };
    for sample in samples {
        shared.samples.push_back(sample.clamp(-1.0, 1.0));
        while shared.samples.len() > shared.max_samples {
            shared.samples.pop_front();
        }
    }
}

fn recent_live_samples(capture: &LiveCapture, seconds: f32) -> Result<Vec<f32>, String> {
    let shared = capture
        .shared
        .lock()
        .map_err(|_| "live audio buffer lock poisoned".to_string())?;
    let wanted =
        (capture.sample_rate as f32 * seconds).round() as usize * capture.channels as usize;
    let start = shared.samples.len().saturating_sub(wanted);
    Ok(shared.samples.iter().skip(start).copied().collect())
}

fn mono_samples(samples: &[f32], channels: u16) -> Vec<f32> {
    let channels = usize::from(channels).max(1);
    if channels == 1 {
        return samples.to_vec();
    }
    samples
        .chunks(channels)
        .map(|frame| frame.iter().copied().sum::<f32>() / frame.len() as f32)
        .collect()
}

fn narrowband_spectrum(
    samples: &[f32],
    sample_rate: u32,
    min_hz: f32,
    max_hz: f32,
    step_hz: f32,
) -> SpectrumSlice {
    if samples.is_empty() || sample_rate == 0 {
        return SpectrumSlice {
            min_hz,
            max_hz,
            step_hz,
            sample_rate,
            bins: Vec::new(),
            peak_hz: min_hz,
            peak: 0.0,
        };
    }

    let n = samples.len();
    let mut bins = Vec::new();
    let mut peak = 0.0_f32;
    let mut peak_hz = min_hz;
    let mut hz = min_hz;
    while hz <= max_hz + 0.1 {
        let mut re = 0.0_f32;
        let mut im = 0.0_f32;
        for (idx, sample) in samples.iter().enumerate() {
            let window =
                0.5 - 0.5 * (2.0 * PI * idx as f32 / (n.saturating_sub(1).max(1)) as f32).cos();
            let angle = 2.0 * PI * hz * idx as f32 / sample_rate as f32;
            let value = sample * window;
            re += value * angle.cos();
            im -= value * angle.sin();
        }
        let magnitude = ((re * re + im * im).sqrt() / n as f32).max(0.0);
        if magnitude > peak {
            peak = magnitude;
            peak_hz = hz;
        }
        bins.push(magnitude);
        hz += step_hz;
    }
    let normaliser = peak.max(1e-6);
    for bin in &mut bins {
        *bin = (*bin / normaliser).sqrt().clamp(0.0, 1.0);
    }

    SpectrumSlice {
        min_hz,
        max_hz,
        step_hz,
        sample_rate,
        bins,
        peak_hz,
        peak,
    }
}

fn audio_level_from_samples(samples: &[f32], sample_rate: u32, channels: u16) -> AudioLevel {
    let mut sum_squares = 0.0_f64;
    let mut peak = 0.0_f32;
    for sample in samples {
        let sample = sample.clamp(-1.0, 1.0);
        peak = peak.max(sample.abs());
        sum_squares += f64::from(sample * sample);
    }
    let rms = if samples.is_empty() {
        0.0
    } else {
        (sum_squares / samples.len() as f64).sqrt() as f32
    };
    let db = if rms > 0.0 {
        20.0 * rms.log10()
    } else {
        -120.0
    };
    AudioLevel {
        rms,
        peak,
        db,
        samples: samples.len(),
        sample_rate,
        channels,
        signal: rms > 0.002 || peak > 0.01,
    }
}

fn write_temp_wav(samples: &[f32], sample_rate: u32, channels: u16) -> Result<PathBuf, String> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| format!("system time error: {err}"))?
        .as_millis();
    let path = std::env::temp_dir().join(format!("morseformer-live-{stamp}.wav"));
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer =
        hound::WavWriter::create(&path, spec).map_err(|err| format!("creating temp WAV: {err}"))?;
    for sample in samples {
        writer
            .write_sample(sample.clamp(-1.0, 1.0))
            .map_err(|err| format!("writing temp WAV: {err}"))?;
    }
    writer
        .finalize()
        .map_err(|err| format!("finalizing temp WAV: {err}"))?;
    Ok(path)
}

fn input_device_by_id(host: &cpal::Host, device_id: &str) -> Result<cpal::Device, String> {
    use cpal::traits::HostTrait;

    let mut devices = host
        .input_devices()
        .map_err(|err| format!("failed to list input devices: {err}"))?;
    if device_id.trim().is_empty() {
        return host
            .default_input_device()
            .ok_or_else(|| "no default input device found".to_string());
    }
    let target_idx = device_id
        .parse::<usize>()
        .map_err(|_| format!("invalid input device id: {device_id}"))?;
    devices
        .nth(target_idx)
        .ok_or_else(|| format!("input device {device_id} was not found"))
}

fn update_stats(samples: impl Iterator<Item = f32>, stats: &Arc<Mutex<AudioStats>>) {
    let mut guard = match stats.lock() {
        Ok(guard) => guard,
        Err(_) => return,
    };
    for sample in samples {
        let sample = sample.clamp(-1.0, 1.0);
        let abs = sample.abs();
        guard.peak = guard.peak.max(abs);
        guard.sum_squares += f64::from(sample * sample);
        guard.samples += 1;
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .expect("src-tauri should live under rust/morseformer-ui")
        .to_path_buf()
}

fn resolve_path(repo_root: &Path, path: &str) -> PathBuf {
    let path = strip_file_url(path.trim());
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        repo_root.join(path)
    }
}

fn strip_file_url(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("file://") {
        percent_decode(rest)
    } else {
        path.to_string()
    }
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex_value(bytes[i + 1]), hex_value(bytes[i + 2])) {
                out.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn parse_field(output: &str, prefix: &str) -> Option<String> {
    output.lines().find_map(|line| {
        line.strip_prefix(prefix)
            .map(|value| value.trim().to_string())
    })
}

fn executable_name(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

fn main() {
    tauri::Builder::default()
        .manage(LiveCaptureState::default())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            capture_input_level,
            decode_live_window,
            decode_wav,
            list_input_devices,
            live_input_status,
            live_spectrum,
            start_live_capture,
            stop_live_capture
        ])
        .run(tauri::generate_context!())
        .expect("error while running morseformer UI");
}
