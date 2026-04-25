# AGENT 3 — ML Model Validation & DER Pipeline
**Role** : Senior ML Engineer — Speaker Diarization
**Duree estimee** : 4-6h
**Prerequis** : Agent 1 termine

---

## Contexte exact du modele (LIRE AVANT DE COMMENCER)

### Etat actuel — PROBLEME FONDAMENTAL

```
voiceflow-ml/models/diarization/   # Architecture Fast CNN (2.3M params)
voiceflow-ml/models/fast_cnn_diarization_optimized.onnx  # Exporte
```

Le modele actuel fait de la **classification binaire** sur pseudo-labels.
Il n'est **PAS** entraine pour la diarization.
Accuracy 61.5% = quasi-aleatoire sur labels synthetiques.

### Ce que doit faire le modele en production

Produire des **embeddings 512-dim** par segment audio.
Deux segments du meme locuteur -> distance cosine < 0.4.
Deux segments de locuteurs differents -> distance cosine > 0.7.

---

## DER — Metrique principale de diarization

```
DER = (False Alarm + Missed Speech + Speaker Confusion) / Total Speech Duration
```

| Seuil | Interpretation |
|-------|---------------|
| DER < 10% | Bon — production-ready |
| DER < 15% | Acceptable — production minimale |
| DER < 25% | Mauvais — ameliorations requises |
| DER > 25% | Inacceptable |

Format RTTM standard :
```
SPEAKER <file_id> 1 <onset_sec> <duration_sec> <NA> <NA> <speaker_id> <NA> <NA>
```

---

## Tache 1 — `voiceflow-ml/evaluation/diarization_evaluator.py` (CREER)

```python
class DiarizationEvaluator:
    """
    Evalue un systeme de diarization avec le DER standard.
    Utilise pyannote.metrics pour conformite avec le benchmark NIST.

    Args:
        collar: Tolerance en secondes aux frontieres (standard: 0.25s)
        skip_overlap: Exclure les regions de parole chevauchante
    """

    def __init__(self, collar: float = 0.25, skip_overlap: bool = False) -> None: ...

    def compute_der(
        self,
        reference_rttm: str,
        hypothesis_rttm: str,
    ) -> dict[str, float]:
        """
        Calcule le DER et ses composantes.

        Args:
            reference_rttm: Chemin vers le fichier RTTM de reference
            hypothesis_rttm: Chemin vers le fichier RTTM du systeme

        Returns:
            {
                'der': float,
                'false_alarm': float,
                'missed_speech': float,
                'speaker_confusion': float,
                'total_speech_duration': float
            }
        """
        ...

    def evaluate_batch(
        self, rttm_pairs: list[tuple[str, str]]
    ) -> dict[str, float]:
        """Evalue plusieurs paires et retourne les metriques aggregees."""
        ...
```

Ajouter dans `requirements.txt` : `pyannote.metrics>=3.2.0`

---

## Tache 2 — `voiceflow-ml/evaluation/embedding_validator.py` (CREER)

```python
class EmbeddingValidator:
    """
    Valide la qualite des embeddings produits par le modele ONNX.

    Criteres de qualite :
    - Intra-speaker distance < 0.4 (meme locuteur, segments differents)
    - Inter-speaker distance > 0.7 (locuteurs differents)
    - Norme L2 = 1.0 (normalisation verifiee)
    - Dimension = 512
    """

    def validate_onnx_model(
        self,
        onnx_path: str,
        test_audio_dir: str
    ) -> dict[str, float]:
        """
        Charge le modele ONNX et valide les embeddings produits.

        Returns:
            {
                'avg_intra_speaker_distance': float,  # cible < 0.4
                'avg_inter_speaker_distance': float,  # cible > 0.7
                'embedding_dim': int,                 # attendu = 512
                'is_normalized': bool,                # attendu True
                'threshold_accuracy': float           # cible > 0.85
                'onnx_output_node_name': str          # pour handoff Agent 4
            }
        """
        ...
```

---

## Tache 3 — Verifier/corriger `models/diarization/train.py`

Le fichier existe. Verifier qu'il utilise une loss de type embedding :

```python
# Si le modele a encore une couche softmax finale, la supprimer :
# SUPPRIMER la couche classification finale
# GARDER : backbone CNN + couche projection vers 512-dim
# AJOUTER : normalisation L2 en sortie (F.normalize(x, dim=1))

# Loss cible : TripletMarginLoss
criterion = torch.nn.TripletMarginLoss(margin=0.3, p=2)

# Format batch : (anchor, positive, negative)
# anchor et positive = meme locuteur, segments differents
# negative = locuteur different
```

---

## Tache 4 — `voiceflow-ml/scripts/validate_onnx.py` (CREER)

Script CLI standalone pour valider le modele ONNX exporte :

```bash
python scripts/validate_onnx.py   --model models/fast_cnn_diarization_optimized.onnx   --test-data tests/fixtures/audio/   --output validation_report.json
```

Output JSON :
```json
{
    "onnx_output_node_name": "embedding",
    "embedding_dim": 512,
    "is_normalized": true,
    "avg_intra_distance": 0.23,
    "avg_inter_distance": 0.78,
    "threshold_accuracy": 0.89,
    "validation_passed": true
}
```

---

## Tache 5 — `voiceflow-ml/api/routes/models.py` : endpoint training

Ajouter :
```python
POST /api/models/train
{
    "dataset": "librispeech_clean",
    "loss_type": "triplet",
    "epochs": 20
}
# -> 202 + { "job_id": "...", "status": "pending" }
```

---

## Contraintes (NE PAS toucher)

- `voiceflow-inference/` — hors scope
- Format export ONNX : conserver le nom de noeud output coherent avec handoff

---

## Verification finale

```bash
# Validation modele ONNX
python scripts/validate_onnx.py --model models/fast_cnn_diarization_optimized.onnx
# -> validation_passed: true, embedding_dim: 512

# DER sur donnees test
python -c "
from evaluation.diarization_evaluator import DiarizationEvaluator
e = DiarizationEvaluator()
r = e.compute_der('tests/fixtures/ref.rttm', 'tests/fixtures/hyp.rttm')
print(f'DER: {r["der"]:.1%}')
"
```

---

## Handoff pour Agent 4 (Rust)

Creer `cursor-agents/handoff-agent3.md` :
- [x] EmbeddingValidator execute sur fast_cnn_diarization_optimized.onnx
- [x] onnx_output_node_name : noter le nom exact (ex: "embedding")
- [x] embedding_dim : 512 (confirme)
- [x] is_normalized : True (confirme)
- [x] DER evaluateur cree (evaluation/diarization_evaluator.py)
