# AGENT 1 — Security Foundation
**Rôle** : Senior Security Engineer  
**Durée estimée** : 2-3h  
**Prérequis** : Aucun — commencer ici

---

## Contexte exact du projet

**VoiceFlow Intelligence Platform** — système hybride Python/Rust de speaker diarization temps réel.

```
VoiceFlow-Intelligence-Platform/
├── voiceflow-ml/
│   ├── api/main.py               # FastAPI app — OK structurellement
│   ├── core/config.py            # ⚠️ jwt_secret_key hardcodé ligne 34
│   ├── core/database.py
│   ├── core/redis_client.py
│   └── requirements.txt
├── voiceflow-inference/
│   └── src/config.rs             # ⚠️ pas de validation si JWT_SECRET_KEY absent
├── docker-compose.yml            # ⚠️ POSTGRES_PASSWORD=voiceflow_password en clair
└── (pas de .env.example)
```

### Problèmes de sécurité identifiés (OWASP Top 10)

| # | Fichier | Problème | OWASP |
|---|---------|---------|-------|
| 1 | `core/config.py:34` | `jwt_secret_key = "your-secret-key-change-in-production"` | A02 |
| 2 | `docker-compose.yml:9` | `POSTGRES_PASSWORD: voiceflow_password` en clair | A02 |
| 3 | `docker-compose.yml` | `GF_SECURITY_ADMIN_PASSWORD=admin` | A02 |
| 4 | Aucun fichier | Pas de validation taille/type fichier audio | A03 |
| 5 | Aucun fichier | Pas de SecretsLoader AWS Secrets Manager | A02 |
| 6 | `api/main.py` | CORS `allow_origins` trop permissif en prod | A05 |

---

## Tâches

### Tâche 1 — `voiceflow-ml/core/secrets_manager.py` (CRÉER)

Classe `SecretsLoader` :
- En `ENV=production` : charge depuis AWS Secrets Manager via boto3
- En développement : fallback sur `os.environ`
- Cache TTL = 300s (évite appels répétés)
- Ne logue JAMAIS les valeurs — uniquement les noms des secrets
- Raise `SecretsLoadError` si un secret requis est absent

```python
@lru_cache(maxsize=1)
def get_secrets_loader() -> SecretsLoader:
    """Singleton compatible FastAPI Depends."""
    return SecretsLoader()
```

Ajouter `boto3>=1.34.0` et `botocore>=1.34.0` dans `requirements.txt`.

### Tâche 2 — `voiceflow-ml/core/config.py` (MODIFIER)

```python
# SUPPRIMER cette ligne (défaut dangereux)
jwt_secret_key: str = "your-secret-key-change-in-production"

# REMPLACER PAR (pas de défaut)
jwt_secret_key: str

@field_validator('jwt_secret_key')
@classmethod
def validate_jwt_secret_length(cls, v: str) -> str:
    if len(v) < 32:
        raise ValueError(
            "JWT_SECRET_KEY must be >= 32 chars. "
            "Generate: openssl rand -hex 32"
        )
    return v
```

Supprimer aussi le défaut `postgres_password: str = "voiceflow_password"`.

### Tâche 3 — `docker-compose.yml` (MODIFIER)

```yaml
# REMPLACER toutes les valeurs hardcodées
environment:
  POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}
  
# Et pour Grafana :
  GF_SECURITY_ADMIN_PASSWORD: ${GF_SECURITY_ADMIN_PASSWORD:?Required}
```

### Tâche 4 — `.env.example` (CRÉER à la racine)

```bash
# VoiceFlow Intelligence Platform — Environment Variables
# Copier : cp .env.example .env  |  Ne jamais committer .env

ENV=development
JWT_SECRET_KEY=<openssl rand -hex 32>
POSTGRES_USER=voiceflow
POSTGRES_PASSWORD=<strong-password>
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=voiceflow
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=<strong-password>

# AWS (production)
AWS_REGION=eu-west-1
# DB_SECRET_ARN=arn:aws:secretsmanager:...
# JWT_SECRET_ARN=arn:aws:secretsmanager:...
```

### Tâche 5 — `voiceflow-ml/api/middleware/input_validation.py` (CRÉER)

Middleware qui valide les fichiers audio uploadés :
- Taille max : `settings.audio_max_size_mb` (100 MB)
- Type : vérification des magic bytes (RIFF=wav, fLaC=flac, OggS=ogg, \xff\xfb=mp3)
- Rejette les fichiers dont l'extension ne correspond pas aux magic bytes
- Raise `HTTPException(400)` avec message clair

### Tâche 6 — `voiceflow-inference/src/config.rs` (MODIFIER)

```rust
pub fn validate_required_secrets() {
    for key in &["JWT_SECRET_KEY"] {
        if std::env::var(key).is_err() {
            panic!(
                "Required secret '{}' not set. Generate: openssl rand -hex 32",
                key
            );
        }
    }
}
```

Appeler `validate_required_secrets()` au début de `main()`.

### Tâche 7 — Vérifier `.gitignore`

```gitignore
# Secrets — must be in .gitignore
.env
*.env
.env.*
!.env.example
```

---

## Contraintes (NE PAS toucher)

- `voiceflow-ml/models/` — hors scope
- `voiceflow-inference/src/inference/` — hors scope
- Structure des routes API existantes
- `prometheus.yml`

---

## Vérification finale

```bash
# Zéro secret hardcodé
grep -rn "your-secret\|voiceflow_password" \
  voiceflow-ml/core/ voiceflow-inference/src/ docker-compose.yml

# Service refuse de démarrer sans JWT_SECRET_KEY
unset JWT_SECRET_KEY && docker-compose up ml-service 2>&1 | grep -i "error\|required"
```

---

## Handoff pour Agents 2, 3, 4

Écrire `cursor-agents/handoff-agent1.md` :
```markdown
# Agent 1 Handoff
- [x] Secrets hardcodés supprimés de config.py, docker-compose.yml
- [x] SecretsLoader AWS créé dans core/secrets_manager.py
- [x] .env.example créé à la racine
- [x] Input validation middleware créé
- [x] Rust config valide JWT_SECRET_KEY au démarrage
```
