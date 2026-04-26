# Échantillons audio pour tests / démo

## Déjà dans le dépôt (petits fichiers, licence BSD côté SciPy)

| Fichier | Source | Usage |
|---------|--------|--------|
| `opensource_scipy_44100hz_pcm16.wav` | [scipy/io/tests/data](https://github.com/scipy/scipy/tree/main/scipy/io/tests/data) (`test-44100Hz-le-1ch-4bytes.wav`) | Smoke test format WAV ; resampler en 16 kHz avant `/infer` (Gradio le fait). |
| `opensource_scipy_44100hz_stereo_float32.wav` | idem (`test-44100Hz-2ch-32bit-float-le.wav`) | Test stéréo → mono dans Gradio / librosa. |
| `sample_16k_mono.wav` | Généré dans le repo | Déjà **16 kHz mono** — prêt pour le moteur Rust sans conversion. |

## Télécharger de la parole « réelle » (à lancer chez toi)

Le corpus **CMU Arctic** (anglais, US, licence libre pour recherche) : ~74–78 Mo par voix, voix nettes.

```powershell
cd voiceflow-ml\tests\fixtures\audio
curl.exe -L -O "http://festvox.org/cmu_arctic/packed/cmu_us_awb_arctic.tar.bz2"
tar -xjf cmu_us_awb_arctic.tar.bz2 cmu_us_awb_arctic/wav/arctic_a0001.wav
move cmu_us_awb_arctic\wav\arctic_a0001.wav cmu_us_awb_arctic_a0001.wav
```

Pour **deux voix différentes** (meilleur test diarisation), répète avec `cmu_us_bdl_arctic.tar.bz2` et concatène les WAV (ou enchaîne dans Audacity) puis exporte en **16 kHz mono**.

## Réunions multi-locuteurs (benchmarks)

- **AMI** : [https://groups.inf.ed.ac.uk/ami/download/](https://groups.inf.ed.ac.uk/ami/download/) — WAV + RTTM.
- **VoxConverse** : les fichiers ne sont plus hébergés sur le site Oxford ; utiliser un miroir académique ou Hugging Face si disponible.

## Commandes rapides

```powershell
# Convertir un fichier téléchargé en 16 kHz mono (pour smoke_infer sans Gradio)
ffmpeg -y -i entree.wav -ac 1 -ar 16000 -sample_fmt s16 sortie_16k.wav

python scripts/smoke_infer.py --audio voiceflow-ml/tests/fixtures/audio/sortie_16k.wav --max-seconds 60
```
