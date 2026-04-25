# ADR-005 — CI/CD : GitHub Actions + ECR + ECS Blue/Green

**Date** : 2026-04-22  
**Statut** : Accepté  
**Décideurs** : Équipe VoiceFlow  

---

## Contexte

Actuellement, il n'existe aucun pipeline CI/CD. Le déploiement est entièrement manuel :
- Pas de tests automatisés en CI
- Pas de build Docker automatique
- Pas de push ECR
- Pas de stratégie de rollback

---

## Décision

**GitHub Actions** pour le CI, **AWS CodeDeploy** pour le déploiement blue/green ECS.

### Pipelines

#### 1. `ci.yml` — Déclenché sur chaque PR + push main

```
Trigger: pull_request, push to main

Jobs (parallèles) :
  test-python:
    - checkout
    - setup Python 3.11
    - pip install -r requirements.txt
    - pytest tests/ --cov --cov-fail-under=70
    - mypy voiceflow-ml/ --strict
    - flake8 voiceflow-ml/
    - bandit -r voiceflow-ml/  # Security scan

  test-rust:
    - checkout
    - setup Rust 1.75 (toolchain stable)
    - cargo test --verbose
    - cargo clippy -- -D warnings
    - cargo fmt --check
    - cargo audit  # Security scan deps

  security-scan:
    - trivy image (Docker images)
    - semgrep (code patterns)
    - gitleaks detect (secrets scan)
```

#### 2. `deploy-staging.yml` — Déclenché sur push main après CI vert

```
Trigger: workflow_run (ci.yml completed, conclusion=success)

Jobs séquentiels :
  build-push-ml:
    - docker buildx build voiceflow-ml/
    - docker tag → ECR ml-service:sha-${{ github.sha }}
    - docker push ECR
    - aws ecs register-task-definition (nouvelle version)

  build-push-inference:
    - docker buildx build voiceflow-inference/ (multi-stage Rust)
    - docker tag → ECR inference-engine:sha-${{ github.sha }}
    - docker push ECR
    - aws ecs register-task-definition

  deploy-ecs-staging:
    - aws ecs update-service --deployment-controller=CODE_DEPLOY
    - CodeDeploy blue/green deployment
    - Health check timeout: 5 minutes
    - Auto-rollback si health check fail

  integration-tests-staging:
    - pytest tests/integration/ --env staging
    - load test (wrk, 30s)
    - smoke test WebSocket streaming
```

#### 3. `deploy-prod.yml` — Déclenché manuellement (GitHub environment protection)

```
Trigger: workflow_dispatch (manuel)
Environment: production (require approval)

Jobs :
  promote-staging-to-prod:
    - Réutilise l'image ECR déjà buildée (même SHA)
    - aws ecs update-service --cluster voiceflow-prod
    - CodeDeploy blue/green (traffic shift 10% → 50% → 100% sur 10 min)
    - CloudWatch alarms → rollback auto si error_rate > 5%
```

### Gestion des secrets CI/CD

```yaml
# GitHub Secrets (repository settings)
AWS_ACCESS_KEY_ID           # IAM role CI (droits ECR push + ECS deploy seulement)
AWS_SECRET_ACCESS_KEY
AWS_REGION
ECR_ML_REGISTRY_URL
ECR_INFERENCE_REGISTRY_URL
```

IAM policy CI (least privilege) :
- `ecr:GetAuthorizationToken`
- `ecr:BatchCheckLayerAvailability`
- `ecr:PutImage`
- `ecs:RegisterTaskDefinition`
- `ecs:UpdateService`
- `codedeploy:CreateDeployment`

---

## Stratégie de tags Docker

| Tag | Usage | Exemple |
|-----|-------|---------|
| `sha-<commit>` | Immutable, référence exacte | `sha-a3f2b1c` |
| `staging-latest` | Dernier déployé sur staging | `staging-latest` |
| `prod-<version>` | Release production | `prod-1.2.0` |
| `latest` | **Interdit en production** | — |

---

## Alternatives considérées

| Option | Pour | Contre |
|--------|------|--------|
| **GitHub Actions + CodeDeploy (retenu)** | Natif GitHub, zero infra CI | Coût CodeDeploy marginal |
| **GitLab CI** | Powerful, self-hosted possible | Migration coût si déjà sur GitHub |
| **AWS CodePipeline** | Full AWS native | Interface complexe, plus cher |
| **CircleCI / Buildkite** | Rapides, DX top | Coût additionnel |

---

## Conséquences

**Positives :**
- Zéro déploiement manuel sur staging/prod
- Rollback automatique si health checks fail
- Tests obligatoires avant tout merge (branch protection)
- Images Docker immutables (par SHA)

**Négatives :**
- Build Rust long (~5-10 min) → cache Cargo layers dans GitHub Actions
- `cargo audit` peut bloquer sur des CVE dans des deps transitives

---

## Critères de validation

- [ ] PR bloquée si `cargo test` ou `pytest` fail
- [ ] Image ECR poussée avec tag `sha-${GITHUB_SHA}` à chaque merge main
- [ ] ECS staging auto-updaté dans les 15 minutes après merge main
- [ ] Déploiement prod nécessite approbation manuelle (GitHub environment protection)
- [ ] `gitleaks detect` ne trouve aucun secret dans le code
- [ ] Rollback automatique déclenché si error_rate > 5% pendant 5 minutes
