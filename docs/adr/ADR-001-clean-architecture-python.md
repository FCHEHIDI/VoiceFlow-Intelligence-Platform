# ADR-001 — Clean Architecture pour le service Python ML

**Date** : 2026-04-22  
**Statut** : Accepté  
**Décideurs** : Équipe VoiceFlow  

---

## Contexte

Le service `voiceflow-ml/` est actuellement structuré en couches plates :  
- `api/routes/` contient de la logique métier directement dans les endpoints FastAPI  
- Pas de couche `domain/` (entités métier pures)  
- Pas de couche `services/` (orchestration de la logique)  
- Pas de couche `repositories/` (accès DB encapsulé)  
- Pas de workers Celery pour les tâches asynchrones batch

Conséquences : couplage fort, tests unitaires impossibles sans DB, code non testable.

---

## Décision

Adopter l'**architecture hexagonale (Clean Architecture)** pour `voiceflow-ml/` :

```
voiceflow-ml/
├── domain/               # Entités métier pures — zéro dépendance externe
│   ├── job.py            # Job, JobStatus (Enum)
│   ├── segment.py        # Segment, Speaker
│   └── model_version.py  # ModelVersion, ModelMetadata
├── repositories/         # Ports → implémentations SQLAlchemy
│   ├── base.py           # BaseRepository[T] générique
│   ├── job_repository.py
│   └── model_repository.py
├── services/             # Use cases — orchestration pure
│   ├── inference_service.py
│   ├── model_service.py
│   └── audio_service.py  # Validation + preprocessing audio
├── workers/              # Celery tasks (broker=Redis)
│   ├── celery_app.py
│   └── inference_tasks.py
└── api/routes/           # Controllers ultra-minces — appels services uniquement
```

### Règles d'architecture

1. **`domain/`** : aucune import de SQLAlchemy, Redis, FastAPI. Dataclasses ou Pydantic pur.
2. **`repositories/`** : implémentent des interfaces Python (ABC), prennent `AsyncSession` en injection.
3. **`services/`** : prennent les repositories en injection → testables avec des mocks.
4. **`api/routes/`** : ≤ 30 lignes par endpoint. Validation Pydantic + appel service + retour HTTP.
5. **`workers/`** : Celery tasks qui font `await service.method()` via `asyncio.run()` ou `loop.run_until_complete()`.

---

## Alternatives considérées

| Option | Pour | Contre |
|--------|------|--------|
| **Architecture plate (statu quo)** | Rien à changer | Non testable, couplage fort |
| **Clean Architecture (retenu)** | Testable, extensible, DI propre | Migration nécessaire |
| **DDD complet** | Très structuré | Over-engineering pour ce scope |

---

## Conséquences

**Positives :**
- Tests unitaires services sans base de données
- Injection de dépendances propre avec `fastapi.Depends`
- Possibilité de changer ORM sans toucher la logique métier
- Celery workers isolés, testables séparément

**Négatives :**
- Migration du code existant (Agent 2)
- Complexité accrue si l'équipe est petite

---

## Critères de validation

- [ ] `pytest tests/unit/` passe sans docker, sans DB, sans Redis
- [ ] Aucun import SQLAlchemy dans `domain/`
- [ ] Chaque endpoint controller ≤ 30 lignes
- [ ] Coverage services/ ≥ 80%
