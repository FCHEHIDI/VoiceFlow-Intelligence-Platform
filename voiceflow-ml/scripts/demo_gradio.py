"""Minimal Gradio UI for diarization against the Rust inference engine.

Prerequisites
-------------
1. Start the Rust service (from ``voiceflow-inference/``) with ``JWT_SECRET_KEY``
   set and a valid ONNX under ``MODELS_DIR`` (stub or real weights).
2. Install demo deps: ``pip install gradio`` (or ``pip install -r requirements.txt``).

Run::

    cd voiceflow-ml
    set VOICEFLOW_INFER_URL=http://127.0.0.1:3000/infer
    python scripts/demo_gradio.py

Open-source audio you can download for testing (multi-speaker / meetings)
---------------------------------------------------------------------------
* **AMI meeting corpus** — benchmark for diarization; WAV + RTTM ground truth.
  https://groups.inf.ed.ac.uk/ami/download/
* **VoxConverse** (Oxford VGG) — conversational TV, many speakers.
  https://www.robots.ox.ac.uk/~vgg/data/voxconverse/
* **AISHELL-4** — Mandarin meetings; good for long recordings.
  https://www.openslr.org/111/
* **Fisher English** (LDC, license) — telephone speech; classic 2-speaker chunks.

Quick try without a big download: use any **LibriSpeech** utterance (single
speaker) to verify the pipeline; for *multiple* speakers you need meeting/call
data like AMI or VoxConverse.

This UI resamples to **16 kHz mono** client-side (``librosa``) before JSON
POST to ``/infer``, same contract as ``scripts/smoke_infer.py``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import httpx
import librosa
import numpy as np

DEFAULT_INFER = os.environ.get("VOICEFLOW_INFER_URL", "http://127.0.0.1:3000/infer")
SAMPLE_RATE = 16_000

SOURCES_BLURB = """
### Fichiers audio open source (idées)

| Source | Lien | Notes |
|--------|------|--------|
| **AMI** | [ami corpus](https://groups.inf.ed.ac.uk/ami/download/) | Réunions, RTTM fourni — référence diarization |
| **VoxConverse** | [voxconverse](https://www.robots.ox.ac.uk/~vgg/data/voxconverse/) | TV multi-locuteurs |
| **AISHELL-4** | [openslr 111](https://www.openslr.org/111/) | Réunions mandarin |
| **LibriSpeech** | [openslr 12](https://www.openslr.org/12/) | Mono locuteur — smoke test rapide |

Télécharge un extrait, puis **Upload** ici. Fenêtre max réglable pour éviter les timeouts sur de longs fichiers.
"""


def _load_audio_to_f32(path: str, max_seconds: float | None) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
    if max_seconds and max_seconds > 0:
        n = int(SAMPLE_RATE * max_seconds)
        y = y[:n]
    return y


def run_inference(
    audio_path: str | None,
    infer_url: str,
    window_secs: float,
    hop_secs: float,
    max_seconds: float,
) -> tuple[str, str]:
    if not audio_path or not Path(audio_path).is_file():
        return "Choisis un fichier audio.", ""

    try:
        trim = max_seconds if max_seconds and max_seconds > 0 else None
        y = _load_audio_to_f32(audio_path, trim)
    except Exception as exc:  # pragma: no cover
        return f"Erreur lecture audio: {exc}", ""

    if y.size == 0:
        return "Audio vide après découpe.", ""

    dur = y.size / SAMPLE_RATE
    payload = {
        "audio": y.astype(float).tolist(),
        "sample_rate": SAMPLE_RATE,
        "window_secs": float(window_secs),
        "hop_secs": float(hop_secs),
    }

    try:
        with httpx.Client(timeout=300.0) as client:
            r = client.post(infer_url.strip(), json=payload)
    except httpx.RequestError as exc:
        return (
            f"**Connexion impossible** vers `{infer_url}`.\n\n"
            f"Détail: `{exc}`\n\n"
            "Vérifie que `voiceflow_inference` tourne et que `JWT_SECRET_KEY` est défini.",
            "",
        )

    if r.status_code != 200:
        return f"HTTP **{r.status_code}**\n\n```\n{r.text[:2000]}\n```", ""

    data = r.json()
    segs = data.get("segments") or []
    lines = "\n".join(
        f"- **{s['start']:.2f}s → {s['end']:.2f}s** — `{s['speaker_id']}` "
        f"(conf. {float(s.get('confidence', 0)):.2f})"
        for s in segs
    )
    summary = (
        f"**Durée envoyée:** {dur:.1f} s @ {SAMPLE_RATE} Hz  \n"
        f"**Latence:** {data.get('latency_ms')} ms  \n"
        f"**Locuteurs (clusters):** {data.get('total_speakers')}  \n"
        f"**Modèle:** `{data.get('model_version')}`  \n\n"
        f"### Segments\n{lines or '_(aucun segment)_'}"
    )
    return summary, json.dumps(data, indent=2, ensure_ascii=False)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="VoiceFlow — démo diarisation") as demo:
        gr.Markdown("# VoiceFlow — démo diarisation")
        gr.Markdown(
            "Envoie l’audio au service Rust **`POST /infer`** (fenêtre glissante + clustering). "
            "Le modèle **stub** ne reflète pas la vraie qualité ; pour une vraie démo, remplace l’ONNX."
        )
        gr.Markdown(SOURCES_BLURB)

        with gr.Row():
            audio_in = gr.Audio(
                label="Fichier audio (WAV, MP3, …)",
                type="filepath",
            )
        with gr.Row():
            infer_url = gr.Textbox(
                label="URL d’inférence",
                value=DEFAULT_INFER,
                placeholder="http://127.0.0.1:3000/infer",
            )
        with gr.Row():
            window_secs = gr.Slider(0.5, 10.0, value=3.0, step=0.5, label="Fenêtre (s)")
            hop_secs = gr.Slider(0.25, 5.0, value=1.0, step=0.25, label="Pas (s)")
            max_seconds = gr.Slider(
                0, 600, value=120, step=5,
                label="Durée max envoyée (s) — 0 = fichier entier (attention aux gros fichiers)",
            )

        btn = gr.Button("Lancer la diarisation", variant="primary")
        out_md = gr.Markdown(label="Résultat")
        out_json = gr.Code(label="JSON brut", language="json")

        btn.click(
            fn=run_inference,
            inputs=[audio_in, infer_url, window_secs, hop_secs, max_seconds],
            outputs=[out_md, out_json],
        )

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False,
    )


if __name__ == "__main__":
    main()
