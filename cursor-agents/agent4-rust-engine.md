# AGENT 4 — Rust Engine — Pipeline Diarization Complet
**Role** : Senior Rust Engineer
**Duree estimee** : 6-8h
**Prerequis** : Agent 1 termine + handoff-agent3.md (nom noeud ONNX connu)

---

## Contexte exact du projet

Service `voiceflow-inference/` — Axum 0.7 + Tokio + ORT 2.0.0-rc.10.

### Etat actuel (lire les fichiers avant de modifier)

```
voiceflow-inference/src/
├── main.rs              # Axum router + AppState + graceful shutdown
├── config.rs            # Config env vars
├── error.rs             # AppError types
├── api/handlers.rs      # POST /infer, WebSocket /ws/stream
├── inference/mod.rs     # ModelRunner ONNX — retourne LOGITS (a changer)
├── metrics/mod.rs       # Prometheus — OK, NE PAS MODIFIER
└── streaming/
    ├── mod.rs           # WebSocket handler — connecter au pipeline
    └── clustering.rs    # OnlineClusterer — EXISTE, NON CONNECTE au pipeline
```

### Probleme fondamental

`inference/mod.rs::ModelRunner::run()` retourne des logits de classification.
`streaming/clustering.rs::OnlineClusterer` existe mais n'est jamais appele.

### Objectif

Connecter le pipeline complet :
```
Audio -> SlidingWindow -> MFCC -> ONNX embedding -> OnlineClusterer -> NDJSON
```

---

## Architecture cible

```
WebSocket audio chunks (JSON)
  -> audio_buffer accumulation
  -> SlidingWindow::new(window=3.0s, hop=1.0s, sample_rate=16000)
  -> Pour chaque fenetre :
     a. MFCC 40 coefs (preprocessing existant ou a creer)
     b. ModelRunner::run_embedding(mfcc) -> Array1<f32> [512]
     c. OnlineClusterer::add_embedding(emb, timestamp) -> speaker_id: usize
  -> smooth_labels()
  -> Serialiser vers NDJSON
  -> Envoyer via socket
```

---

## Budget latence (contrainte dure)

| Composant | Budget P99 |
|-----------|-----------|
| MFCC extraction | < 5ms |
| ONNX embedding inference | < 5ms |
| Clustering assignment | < 2ms |
| Temporal smoothing | < 1ms |
| Total par fenetre | **< 15ms** |

Ajouter des mesures : `INFERENCE_LATENCY.observe(elapsed.as_secs_f64())`

---

## Tache 1 — `inference/mod.rs` : ajouter `run_embedding()`

Le `ModelRunner` garde la methode `run()` existante (ne pas supprimer).
Ajouter une nouvelle methode :

```rust
/// Extrait un embedding 512-dim a partir de features MFCC.
///
/// # Arguments
/// * `features` - Features MFCC, shape [1, 40, time_frames]
///
/// # Returns  
/// * `Array1<f32>` de shape [512], normalise L2
///
/// # Note
/// Le nom du noeud output ONNX est dans handoff-agent3.md
pub async fn run_embedding(
    &self,
    features: Array2<f32>,
) -> AppResult<Array1<f32>> {
    // 1. Creer un ORT tensor depuis features
    // 2. Executer la session ONNX
    // 3. Extraire le noeud "embedding" (PAS "logits")
    // 4. Normaliser L2 le vecteur
    // 5. Retourner Array1<f32> shape [512]
}
```

---

## Tache 2 — `streaming/sliding_window.rs` (CREER)

```rust
/// Segmente un buffer audio PCM en fenetres chevauchantes.
///
/// # Arguments
/// * `audio` - Samples PCM f32 a sample_rate Hz
/// * `window_secs` - Duree de chaque fenetre (defaut: 3.0s)
/// * `hop_secs` - Pas entre le debut de 2 fenetres (defaut: 1.0s)
/// * `sample_rate` - Frequence d'echantillonnage (16000 pour ce projet)
///
/// # Returns
/// * Vec de (start_seconds, end_seconds, window_samples)
pub fn sliding_window(
    audio: &[f32],
    window_secs: f64,
    hop_secs: f64,
    sample_rate: u32,
) -> Vec<(f64, f64, Vec<f32>)> { ... }
```

---

## Tache 3 — `streaming/clustering.rs` : completer et connecter

Le module existe. Verifier qu'il expose ces methodes et les completer si besoin :

```rust
impl OnlineClusterer {
    pub fn new() -> Self { ... }

    /// Ajoute un embedding et retourne le speaker_id assigne.
    /// Utilise la distance cosine avec le threshold par defaut = 0.5
    pub fn add_embedding(
        &mut self,
        embedding: Array1<f32>,
        timestamp_seconds: f64
    ) -> usize { ... }

    /// Applique un median filter (fenetre=5 frames) pour stabiliser.
    pub fn smooth_labels(&self) -> Vec<(f64, f64, usize)> { ... }

    /// Retourne les segments (start, end, speaker_id).
    pub fn get_segments(&self) -> Vec<(f64, f64, usize)> { ... }
}
```

---

## Tache 4 — `streaming/mod.rs` : pipeline WebSocket complet

Format de reception (JSON par message) :
```json
{ "type": "audio_chunk", "data": [0.1, -0.2, ...], "sequence": 0 }
{ "type": "end_stream" }
```

Format de reponse (NDJSON, un objet par ligne) :
```json
{"type":"segment","start":0.0,"end":3.0,"speaker":"speaker_0","confidence":0.92}
{"type":"segment","start":2.5,"end":5.5,"speaker":"speaker_1","confidence":0.88}
{"type":"end_stream","total_speakers":2,"duration_seconds":5.5}
```

---

## Tache 5 — `api/handlers.rs` : POST /infer

Modifier pour retourner des segments (pas des logits) :

```
POST /infer
Body: { "audio": [f32, ...], "sample_rate": 16000 }

Response 200:
{
  "segments": [
    { "start": 0.0, "end": 3.0, "speaker": "speaker_0" },
    { "start": 2.5, "end": 5.5, "speaker": "speaker_1" }
  ],
  "latency_ms": 25
}
```

---

## Contraintes (NE PAS toucher)

- `metrics/mod.rs` — ne pas supprimer les metriques Prometheus existantes
- `Cargo.toml` — ne pas ajouter de crates non justifiees
- `error.rs` — utiliser les AppError existants

---

## Verification finale

```bash
# Build release (zero warning)
cargo build --release --manifest-path voiceflow-inference/Cargo.toml 2>&1 | grep "^error"

# Tests
cargo test --manifest-path voiceflow-inference/Cargo.toml 2>&1 | tail -5

# Clippy
cargo clippy --manifest-path voiceflow-inference/Cargo.toml -- -D warnings

# Test POST /infer manuel
curl -X POST http://localhost:3000/infer   -H "Content-Type: application/json"   -d '{"audio": [0.0, 0.1, -0.1], "sample_rate": 16000}'
# Doit retourner des segments
```

---

## Handoff pour Agents 7 et 8

Creer `cursor-agents/handoff-agent4.md` :
- [x] run_embedding() implemente, output shape [512], normalise L2
- [x] sliding_window() implemente : fenetres 3s, hop 1s
- [x] OnlineClusterer connecte au pipeline WebSocket
- [x] WebSocket envoie segments NDJSON
- [x] POST /infer retourne segments JSON
- [x] Latence P99 mesure : noter valeur mesuree (cible < 15ms/fenetre)
