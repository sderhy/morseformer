use std::{
    path::{Path, PathBuf},
    process::Command,
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

    let onnx_dir = resolve_path(&repo_root, &onnx_dir);
    let wav_path = resolve_path(&repo_root, &wav_path);
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
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![decode_wav, list_input_devices])
        .run(tauri::generate_context!())
        .expect("error while running morseformer UI");
}
