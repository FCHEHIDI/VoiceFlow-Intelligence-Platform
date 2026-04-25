# ADR-003 — Infrastructure AWS : ECS Fargate vs Kubernetes vs EC2

**Date** : 2026-04-22  
**Statut** : Accepté  
**Décideurs** : Équipe VoiceFlow  

---

## Contexte

Le projet doit passer de Docker Compose local à une infrastructure production AWS. Trois options ont été évaluées pour orchestrer les containers `voiceflow-ml` (Python, port 8000) et `voiceflow-inference` (Rust, port 3000).

Contraintes :
- Équipe petite (< 5 dev)
- Budget optimisé (pas de trafic constant)
- WebSocket sur l'inference engine (sticky sessions obligatoires)
- ONNX model partagé via volume (S3 → montage EFS ou copie au démarrage)
- SLA : 99.9% uptime, P99 < 100ms

---

## Décision

**ECS Fargate** avec les composants AWS suivants :

```
Internet
  → Route53 (DNS)
  → ACM (TLS/SSL)
  → ALB (Application Load Balancer)
     ├── /api/* → Target Group: ml-service (port 8000)
     │              ECS Service: voiceflow-ml (Fargate)
     └── /ws/*  → Target Group: inference-engine (port 3000)
     │   /infer    Sticky Sessions: duration_cookie=24h
                   ECS Service: voiceflow-inference (Fargate)

  Privé :
  ├── Aurora PostgreSQL Serverless v2 (RDS)
  ├── ElastiCache Redis 7.x (cluster mode OFF — réplication simple)
  ├── S3 (models bucket, audio-uploads bucket)
  └── Secrets Manager (JWT secret, DB password, Redis auth)
```

### Dimensionnement ECS Tasks

| Service | CPU | Memory | Min tasks | Max tasks |
|---------|-----|--------|-----------|-----------|
| ml-service | 1024 (1 vCPU) | 2048 MB | 1 | 4 |
| inference-engine | 512 | 1024 MB | 2 | 10 |

Auto-scaling inference-engine sur `websocket_connections_active > 200`.

### ALB WebSocket

ALB supporte nativement les WebSockets. Sticky sessions via `AWSALB` cookie (durée 24h) pour garantir qu'un client reste sur le même container Fargate pendant une session de streaming.

---

## Alternatives considérées

| Option | Pour | Contre | Décision |
|--------|------|--------|----------|
| **ECS Fargate (retenu)** | Serverless containers, peu d'ops | Plus cher qu'EC2 au long terme | ✅ Retenu |
| **EKS (Kubernetes)** | Flexibilité max, SOTA ops | Overhead énorme pour petite équipe | ❌ Prématuré |
| **EC2 + Docker Compose** | Simple, cheap | Gestion AMI, patch OS, pas d'auto-scaling | ❌ Non prod-ready |
| **App Runner** | Zero config | Pas de WebSocket, pas de VPC privé | ❌ Incompatible |
| **Lambda** | Serverless | Cold starts, 15min max, WebSocket limité | ❌ Incompatible |

---

## Conséquences

**Positives :**
- Pas de gestion de serveurs (OS, patches)
- Auto-scaling natif via ECS Service Auto Scaling
- Intégration native avec ALB, Secrets Manager, CloudWatch, X-Ray
- Blue/Green deployments via AWS CodeDeploy
- Isolation réseau via VPC/Security Groups

**Négatives :**
- `voiceflow-inference` Rust compile en binaire statique → besoin `Dockerfile` multi-stage (déjà présent)
- ONNX model doit être accessible depuis Fargate → S3 + copie au démarrage du container (pas d'EFS pour simplifier)
- Coût estimé dev : ~$50/mois, prod : ~$200-400/mois (selon trafic)

---

## Architecture Terraform (modules)

```
infra/
├── modules/
│   ├── networking/     # VPC, subnets, SGs, NAT GW
│   ├── ecs/            # Fargate cluster, task defs, services, IAM
│   ├── rds/            # Aurora PostgreSQL Serverless v2
│   ├── elasticache/    # Redis
│   ├── s3/             # Models + audio buckets
│   ├── secrets/        # Secrets Manager
│   ├── ecr/            # ECR repositories (ml-service + inference)
│   └── monitoring/     # CloudWatch dashboards, alarms, X-Ray
└── environments/
    ├── dev/            # Single AZ, min capacity
    └── prod/           # Multi-AZ, auto-scaling
```

---

## Critères de validation

- [ ] `terraform plan` sans erreurs sur compte AWS de dev
- [ ] `terraform apply` déploie les 2 services en < 10 minutes
- [ ] ALB health checks verts sur `/health` pour les 2 services
- [ ] WebSocket sticky sessions fonctionnelles (test avec 100 connexions concurrentes)
- [ ] Secrets Manager injecte les variables d'environnement sans valeurs en clair
- [ ] `terraform destroy` nettoie toutes les ressources sans erreur
