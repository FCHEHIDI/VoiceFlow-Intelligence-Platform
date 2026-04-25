# AGENT 6 — CI/CD Pipeline GitHub Actions
**Role** : Senior DevOps / Platform Engineer
**Duree estimee** : 4-5h
**Prerequis** : Agent 5 termine (ECR + ECS + handoff-agent5.md)

---

## Contexte

Actuellement : deploiement 100% manuel, aucun pipeline CI.

### 3 workflows a creer

```
.github/
└── workflows/
    ├── ci.yml              # Tests sur chaque PR
    ├── deploy-staging.yml  # Auto-deploy sur merge main
    └── deploy-prod.yml     # Deploy prod (approbation manuelle)
```

---

## Workflow 1 — `.github/workflows/ci.yml`

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

jobs:
  test-python:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: voiceflow-ml

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov=. --cov-fail-under=70 -q
      - run: mypy . --ignore-missing-imports
      - run: flake8 . --max-line-length=100 --exclude=.venv,migrations
      - name: Security scan (bandit)
        run: pip install bandit && bandit -r . -ll -x tests/

  test-rust:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: voiceflow-inference

    steps:
      - uses: actions/checkout@v4

      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy

      - uses: Swatinem/rust-cache@v2

      - run: cargo test --verbose
      - run: cargo clippy -- -D warnings
      - run: cargo fmt --check
      - name: Security audit
        run: cargo install cargo-audit && cargo audit

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Requis pour gitleaks

      - name: Scan secrets (gitleaks)
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: SAST scan (semgrep)
        uses: semgrep/semgrep-action@v1
        with:
          config: p/owasp-top-ten
```

---

## Workflow 2 — `.github/workflows/deploy-staging.yml`

```yaml
name: Deploy Staging

on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]
    branches: [main]

env:
  AWS_REGION: eu-west-1
  IMAGE_TAG: sha-${{ github.sha }}

jobs:
  build-push:
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest
    permissions:
      id-token: write   # Requis pour OIDC
      contents: read

    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS via OIDC (pas de static keys)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_CI_ROLE_ARN }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Login ECR
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build & push ml-service
        uses: docker/build-push-action@v5
        with:
          context: ./voiceflow-ml
          push: true
          tags: |
            ${{ secrets.ECR_ML_URL }}:${{ env.IMAGE_TAG }}
            ${{ secrets.ECR_ML_URL }}:staging-latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Build & push inference-engine
        uses: docker/build-push-action@v5
        with:
          context: ./voiceflow-inference
          push: true
          tags: |
            ${{ secrets.ECR_INFERENCE_URL }}:${{ env.IMAGE_TAG }}
            ${{ secrets.ECR_INFERENCE_URL }}:staging-latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-ecs:
    needs: build-push
    runs-on: ubuntu-latest
    steps:
      - name: Configure AWS
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_CI_ROLE_ARN }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Update ECS services
        run: |
          for service in ml-service inference-engine; do
            aws ecs update-service               --cluster voiceflow-staging               --service $service               --force-new-deployment
          done

      - name: Wait for stability (max 10 min)
        run: |
          aws ecs wait services-stable             --cluster voiceflow-staging             --services ml-service inference-engine

  smoke-tests:
    needs: deploy-ecs
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: |
          curl -f ${{ secrets.STAGING_ALB_URL }}/health
          curl -f ${{ secrets.STAGING_ALB_URL }}:3000/health
```

---

## Workflow 3 — `.github/workflows/deploy-prod.yml`

```yaml
name: Deploy Production

on:
  workflow_dispatch:
    inputs:
      image_tag:
        description: 'Tag a deployer (default: staging-latest)'
        required: false
        default: 'staging-latest'
      confirm:
        description: 'Taper DEPLOY pour confirmer'
        required: true

environment: production  # Requiert approbation dans GitHub Settings

jobs:
  gate:
    if: ${{ github.event.inputs.confirm == 'DEPLOY' }}
    runs-on: ubuntu-latest
    steps:
      - run: echo "Deploying ${{ github.event.inputs.image_tag }} to production"

  deploy:
    needs: gate
    runs-on: ubuntu-latest
    steps:
      - name: Configure AWS
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_PROD_ROLE_ARN }}
          aws-region: eu-west-1

      - name: Update ECS prod
        run: |
          for service in ml-service inference-engine; do
            aws ecs update-service               --cluster voiceflow-prod               --service $service               --force-new-deployment
          done
          aws ecs wait services-stable             --cluster voiceflow-prod             --services ml-service inference-engine
```

---

## GitHub Secrets requis

```
AWS_CI_ROLE_ARN          # IAM Role OIDC pour staging (ECR push + ECS update)
AWS_PROD_ROLE_ARN        # IAM Role OIDC pour prod (ECS update uniquement)
ECR_ML_URL               # URL repo ECR ml-service
ECR_INFERENCE_URL        # URL repo ECR inference-engine
STAGING_ALB_URL          # https://voiceflow-staging-alb.xxx.amazonaws.com
```

---

## IAM Role CI — Politique least privilege

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": ["ecs:UpdateService", "ecs:DescribeServices"],
      "Resource": "arn:aws:ecs:*:*:service/voiceflow-staging/*"
    }
  ]
}
```

---

## Branch Protection Rules (a configurer dans GitHub Settings)

```
Branch: main
- Require status checks: test-python, test-rust, security-scan
- Require branches up to date before merging
- Require 1 approval
- Dismiss stale reviews
- Prevent force pushes
```

---

## Verification finale

```bash
# Simuler PR
git checkout -b test/ci-check
echo "test" > /tmp/test.txt && git add . && git commit -m "test: CI check"
git push origin test/ci-check
# -> CI doit passer tous les checks

# Verifier image ECR apres merge main
aws ecr describe-images --repository-name ml-service   --query 'imageDetails[].imageTags'
# -> Doit contenir sha-<commit> et staging-latest
```
