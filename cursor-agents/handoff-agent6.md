# Agent 6 Handoff — CI/CD Pipeline

## CI/CD

### Workflow files created

| File | Trigger | Purpose |
| ---- | ------- | ------- |
| `.github/workflows/ci.yml` | `pull_request`, `push` to `main` | Lint, type-check, test, security scan for Python (`voiceflow-ml`) and Rust (`voiceflow-inference`); gitleaks + semgrep on the whole repo. Uses a `concurrency` group keyed on `workflow + ref` so superseded runs are cancelled. |
| `.github/workflows/deploy-staging.yml` | `workflow_run` of `CI` on `main` with `conclusion == 'success'`, plus manual `workflow_dispatch` | Build + push both Docker images to ECR (tags `sha-${{ github.sha }}` and `staging-latest`), force a new ECS deployment in `voiceflow-staging`, wait for stability, run `/health` smoke tests against the staging ALB. AWS auth via GitHub OIDC (no static keys). |
| `.github/workflows/deploy-prod.yml` | `workflow_dispatch` only (inputs: `image_tag`, `confirm`) | `gate` job validates `confirm == 'DEPLOY'`; `deploy` job runs in the `production` GitHub Environment (manual approver gate) and calls `aws ecs update-service` / `aws ecs wait services-stable` against cluster `voiceflow-prod`. |

### IAM artifacts

- `infra/iam/ci-policy.json` — least-privilege policy for the CI/staging role:
  - ECR push permissions (`GetAuthorizationToken`, `BatchCheckLayerAvailability`, `PutImage`, `InitiateLayerUpload`, `UploadLayerPart`, `CompleteLayerUpload`, plus `BatchGetImage` / `GetDownloadUrlForLayer` so buildx can read cache layers) on `Resource: "*"` (ECR auth token cannot be scoped).
  - `ecs:UpdateService` and `ecs:DescribeServices` scoped to `arn:aws:ecs:*:*:service/voiceflow-staging/*`.
- `infra/iam/ci-trust-policy.json` — OIDC trust policy template for the role:
  - Federated principal: `arn:aws:iam::ACCOUNT:oidc-provider/token.actions.githubusercontent.com`
  - `sts:AssumeRoleWithWebIdentity`
  - `aud == sts.amazonaws.com`
  - `sub` restricted (StringLike) to `repo:OWNER/REPO:ref:refs/heads/main` and `repo:OWNER/REPO:environment:production`.
  - **Replace `OWNER`, `REPO`, `ACCOUNT` placeholders before `aws iam create-role`.**
- The production role (`AWS_PROD_ROLE_ARN`) should reuse the same trust policy but with a policy that only allows `ecs:UpdateService` / `ecs:DescribeServices` on `arn:aws:ecs:*:*:service/voiceflow-prod/*` (no ECR push).

### GitHub Secrets required

Configure under **Settings → Secrets and variables → Actions → Repository secrets**:

| Secret | Used by | Description |
| ------ | ------- | ----------- |
| `AWS_CI_ROLE_ARN` | `deploy-staging.yml` | OIDC role assumed by CI for ECR push + ECS staging deploy. |
| `AWS_PROD_ROLE_ARN` | `deploy-prod.yml` | OIDC role assumed for ECS prod deploy (no ECR). |
| `ECR_ML_URL` | `deploy-staging.yml` | ECR repo URI for `ml-service` (e.g. `123456789012.dkr.ecr.eu-west-1.amazonaws.com/voiceflow/ml-service`). |
| `ECR_INFERENCE_URL` | `deploy-staging.yml` | ECR repo URI for `inference-engine`. |
| `STAGING_ALB_URL` | `deploy-staging.yml` | Base URL of the staging ALB (e.g. `https://voiceflow-staging-alb.eu-west-1.elb.amazonaws.com`). |

`GITHUB_TOKEN` is provided automatically by Actions and does not need to be configured.

### GitHub Environments to configure

- **`production`** (Settings → Environments → New environment):
  - Required reviewers: at least one approver from the SRE/Platform team.
  - Optional wait timer (e.g. 5 min) for cool-off.
  - Deployment branches: `main` only.
  - Environment secrets / variables can override `AWS_PROD_ROLE_ARN` if you prefer a per-environment scope.

- (Optional) **`staging`** environment for visibility / deployment history; not strictly required.

### Branch protection rules to enable on `main`

Configure under **Settings → Branches → Branch protection rules → `main`**:

- Require a pull request before merging.
  - Require **1 approving review**.
  - Dismiss stale pull request approvals when new commits are pushed.
- Require status checks to pass before merging:
  - `test-python`
  - `test-rust`
  - `security-scan`
  - Require branches to be up to date before merging.
- Require conversation resolution before merging.
- Restrict who can push to matching branches (no direct pushes).
- Disallow force pushes and branch deletion.
- (Recommended) Require signed commits.

### Operational notes & caveats

- The Python job exports `JWT_SECRET_KEY` (>= 32 chars) and `POSTGRES_PASSWORD` env vars before running pytest, satisfying Agent 1's `Settings` validation in `voiceflow-ml/core/config.py`.
- `pytest` runs with `-m "not hardware and not slow"` to skip hardware-dependent and long-running tests in CI; markers must remain registered in `voiceflow-ml/pytest.ini` / `pyproject.toml`.
- `cargo audit` is allowed to fail (`continue-on-error: true`) because RustSec advisories on transitive deps would otherwise block PRs; review its output regularly.
- `deploy-staging.yml` uses `workflow_run` so it only fires after a successful CI run on `main`. The `if: github.event.workflow_run.conclusion == 'success'` guard is enforced on the first job; `workflow_dispatch` bypasses it for emergency redeploys.
- Both deploy workflows authenticate via GitHub OIDC; no long-lived AWS keys are stored anywhere.
- The pre-existing `ml-pipeline.yml` and `inference-pipeline.yml` workflows can be removed once `ci.yml` is green on a few PRs; they are kept for now to avoid losing coverage during the transition.
- Smoke tests assume the inference engine is reachable on port 3000 of the same ALB hostname; adjust if Agent 5 uses a different ALB / DNS name.
- Replace `OWNER`, `REPO`, and `ACCOUNT` in `infra/iam/ci-trust-policy.json` before applying with `aws iam create-role --assume-role-policy-document file://infra/iam/ci-trust-policy.json`.
