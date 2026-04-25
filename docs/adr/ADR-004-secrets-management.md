# ADR-004 — Gestion des Secrets : AWS Secrets Manager + .env local

**Date** : 2026-04-22  
**Statut** : Accepté  
**Décideurs** : Équipe VoiceFlow  

---

## Contexte

Problèmes de sécurité actuels (critiques) :

1. `voiceflow-ml/core/config.py` ligne 34 :
   ```python
   jwt_secret_key: str = "your-secret-key-change-in-production"
   ```
   Valeur par défaut hardcodée → en production, si `JWT_SECRET_KEY` absent de l'env, le service démarre avec une clé prévisible.

2. `docker-compose.yml` lignes 8-10 :
   ```yaml
   POSTGRES_PASSWORD: voiceflow_password
   ```
   Credentials en clair dans le VCS.

3. `voiceflow-inference/src/config.rs` : pas de validation → service démarre sans JWT secret valide.

4. Aucune rotation de secrets.

---

## Décision

### Stratégie multi-environnement

| Environnement | Méthode | Outil |
|--------------|---------|-------|
| Local dev | `.env` (gitignored) | pydantic-settings |
| CI/CD | GitHub Secrets | Variables d'env injectées |
| Staging/Prod AWS | AWS Secrets Manager | boto3 + Fargate env injection |

### Python : `core/secrets_manager.py`

```python
class SecretsLoader:
    """
    Charge les secrets selon l'environnement.
    
    En production (ENV=production) : AWS Secrets Manager via boto3.
    En développement : variables d'environnement locales.
    Cache TTL = 5 minutes (évite appels répétés à l'API AWS).
    Ne logue JAMAIS les valeurs — seulement les noms.
    """
    
    def get_secret(self, secret_name: str) -> str: ...
    def get_secret_json(self, secret_arn: str) -> dict: ...
```

### Python : `core/config.py` — suppressions des défauts dangereux

```python
# AVANT (dangereux)
jwt_secret_key: str = "your-secret-key-change-in-production"

# APRÈS (sécurisé)
jwt_secret_key: str  # Pas de défaut → ValueError si absent

@validator('jwt_secret_key')
def validate_jwt_secret(cls, v):
    if len(v) < 32:
        raise ValueError("JWT_SECRET_KEY must be at least 32 characters")
    return v
```

### Rust : `config.rs` — panique explicite si secret absent

```rust
pub fn jwt_secret() -> String {
    env::var("JWT_SECRET_KEY")
        .expect("JWT_SECRET_KEY must be set — generate with: openssl rand -hex 32")
}
```

### AWS Fargate : injection via ECS Task Definition

```hcl
secrets = [
  { name = "JWT_SECRET_KEY",    valueFrom = "${aws_secretsmanager_secret.jwt.arn}" },
  { name = "POSTGRES_PASSWORD", valueFrom = "${aws_secretsmanager_secret.db.arn}:password::" },
  { name = "REDIS_PASSWORD",    valueFrom = "${aws_secretsmanager_secret.redis.arn}:auth_token::" }
]
```

ECS récupère les valeurs au démarrage du task — jamais stockées dans les logs ou le code.

### `.env.example` (à la racine du repo)

```bash
# Application
ENV=development
JWT_SECRET_KEY=<generate: openssl rand -hex 32>

# Database  
POSTGRES_USER=voiceflow
POSTGRES_PASSWORD=<strong-random-password>
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=voiceflow

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=<optional-redis-auth>

# AWS (production uniquement)
AWS_REGION=eu-west-1
DB_SECRET_ARN=<arn:aws:secretsmanager:...>
JWT_SECRET_ARN=<arn:aws:secretsmanager:...>

# Services
RUST_SERVICE_URL=http://inference-engine:3000
GRAFANA_ADMIN_PASSWORD=<strong-password>
```

---

## Alternatives considérées

| Option | Pour | Contre |
|--------|------|--------|
| **AWS Secrets Manager (retenu)** | Rotation auto, audit trail, natif ECS | Coût ($0.40/secret/mois) |
| **HashiCorp Vault** | Très puissant, multi-cloud | Infrastructure supplémentaire |
| **Parameter Store SSM** | Gratuit pour Standard | Pas de rotation auto, moins puissant |
| **Kubernetes Secrets** | Natif K8s | Pas K8s (ADR-003) |

---

## Conséquences

**Positives :**
- Aucune valeur sensible dans le code ou les logs
- Rotation de secrets possible sans redéploiement
- Audit trail complet (qui a accédé à quel secret, quand)
- Conformité OWASP A02 (Cryptographic Failures)

**Négatives :**
- Coût marginal AWS Secrets Manager
- Module `SecretsLoader` à maintenir

---

## Critères de validation

- [ ] `grep -r "password\|secret\|key" --include="*.py" voiceflow-ml/` → zéro valeur hardcodée
- [ ] `grep -r "password\|secret\|key" --include="*.rs" voiceflow-inference/` → zéro valeur hardcodée
- [ ] `grep "POSTGRES_PASSWORD" docker-compose.yml` → toujours `${POSTGRES_PASSWORD}` (variable env)
- [ ] Service refuse de démarrer si `JWT_SECRET_KEY` absent
- [ ] `.env.example` committé, `.env` dans `.gitignore`
- [ ] `git log --all -- "**/.env"` → aucun commit avec fichier `.env` réel
