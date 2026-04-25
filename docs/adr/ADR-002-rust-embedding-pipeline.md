# ADR-002 — Pipeline Diarization Rust : Embeddings + Clustering (pas de classification)

**Date** : 2026-04-22  
**Statut** : Accepté  
**Décideurs** : Équipe VoiceFlow  

---

## Contexte

L'approche initiale du modèle était une **classification multi-classes** (`479 classes = speakers LibriSpeech`). Elle a été correctement identifiée comme fondamentalement incorrecte pour la production car :
- Ne fonctionne qu'avec les speakers d'entraînement
- Inutile avec de nouveaux speakers (Alice, Bob dans une vraie réunion)
- 61.5% accuracy sur pseudo-labels ne mesure rien de réel

La refactorisation a introduit `streaming/clustering.rs` mais ce module n'est **pas encore connecté** au pipeline ONNX dans `inference/mod.rs`.

---

## Décision

Le pipeline Rust complet doit être :

```
Audio (WebSocket) 
  → Sliding Window (3s, hop 1s)  [sliding_window.rs]
  → MFCC Extraction (40 coefs)   [preprocessing.rs]
  → ONNX Embedding (512-dim)     [inference/mod.rs — output embeddings]
  → Online Clustering            [streaming/clustering.rs — OnlineClusterer]
  → Temporal Smoothing           [median filter]
  → RTTM Segments (NDJSON)       [rttm.rs]
```

### Changements nécessaires dans `inference/mod.rs`

Le `ModelRunner` doit exposer `run_embedding()` → `Array1<f32>` (512-dim) en plus du `run()` classique.

```rust
pub async fn run_embedding(&self, features: Array2<f32>) -> AppResult<Array1<f32>> {
    // Run ONNX, extract embedding output node
    // Output shape: [1, 512] → flatten to [512]
}
```

### Format de sortie WebSocket (NDJSON streaming)

```json
{"type":"segment","start":0.0,"end":3.0,"speaker":"speaker_0","confidence":0.92}
{"type":"segment","start":2.5,"end":5.5,"speaker":"speaker_1","confidence":0.88}
{"type":"end_stream","total_speakers":2,"duration_seconds":5.5}
```

### Format RTTM (pour évaluation DER)

```
SPEAKER meeting001 1 0.000 3.000 <NA> <NA> speaker_0 <NA> <NA>
SPEAKER meeting001 1 2.500 3.000 <NA> <NA> speaker_1 <NA> <NA>
```

---

## Alternatives considérées

| Option | Pour | Contre |
|--------|------|--------|
| **Classification (statu quo)** | Déjà entraîné | Inutile en production |
| **Embeddings + Clustering (retenu)** | Généralise aux nouveaux speakers | Entraînement plus long |
| **pyannote-audio en Python** | SOTA DER < 8% | Latence Python ~200ms, trop lent |
| **Speaker Diarization Rust externe** | Clé en main | Pas de contrôle, dépendance externe |

---

## Budget latence par composant (target total < 100ms P99)

| Composant | Budget | Mesuré |
|-----------|--------|--------|
| Network (WebSocket) | 10-40ms | Variable |
| Sliding window + MFCC | 5ms | À mesurer |
| ONNX embedding | 5ms | 4.48ms ✅ |
| Online clustering | 2ms | ~2ms ✅ |
| Temporal smoothing | 1ms | ~1ms ✅ |
| **Total** | **< 100ms** | **~20-50ms** ✅ |

---

## Conséquences

**Positives :**
- Pipeline correct, fonctionne avec n'importe quel speaker
- DER mesurable et optimisable
- Latence maintenue sous 100ms

**Négatives :**
- Nécessite d'entraîner un nouveau modèle (triplet loss sur LibriSpeech)
- ONNX doit exposer le nœud embeddings (pas logits)
- Tester le clustering en production demande de vrais enregistrements

---

## Critères de validation

- [ ] `ModelRunner::run_embedding()` retourne `Array1<f32>` de shape `[512]`
- [ ] `OnlineClusterer::add_embedding()` retourne un speaker_id stable
- [ ] Latence P99 bout-en-bout < 100ms sur 1000 WebSocket chunks (benchmarks `wrk`)
- [ ] DER < 15% sur AMI corpus test set (10 meetings)
