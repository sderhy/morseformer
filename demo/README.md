# morseformer — Gradio demo

A small Gradio app exposing the public morseformer decoder over a web
UI: upload a `.wav` or record from your browser mic, pick a preset, get
the transcript back. Designed for the Hugging Face Spaces deployment
slot but runs locally too.

## Run locally

```bash
pip install "morseformer[demo]"
python demo/app.py
# → http://127.0.0.1:7860
```

## Deploy to a Hugging Face Space

1. Create a new Space (SDK: **Gradio**).
2. Add a `requirements.txt` with at minimum:
   ```
   --extra-index-url https://download.pytorch.org/whl/cpu
   torch
   torchaudio
   morseformer[demo]
   ```
3. Push this `demo/app.py` as the Space's `app.py` (Gradio's expected
   entry point). The Space's environment downloads the model from the
   public `sderhy/morseformer` repo on first request.

## Notes

- The browser microphone takes the OS-default input device. The Gradio
  audio component does **not** expose USB-specific device selection — if
  you need to pick the USB capture explicitly, use the native PySide6 GUI
  instead (`morseformer gui`).
- Set `PORT` to bind a different port. `--server-name 0.0.0.0` makes the
  app reachable from the LAN.
