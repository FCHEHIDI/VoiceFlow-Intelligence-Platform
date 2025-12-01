# Project Cleanup & Refinement Plan

## 🔍 Audit Results

### Redundant Export Scripts (TO REMOVE)
The following scripts are **deprecated** and replaced by the unified `models/diarization/export_onnx.py`:

1. ❌ `export_onnx_direct.py` - Basic ONNX export (superseded)
2. ❌ `export_onnx_force_legacy.py` - Legacy fallback (superseded)
3. ❌ `export_onnx_legacy.py` - Old implementation (superseded)
4. ❌ `export_onnx_patched.py` - Temporary workaround (superseded)
5. ❌ `export_transformer_onnx.py` - Old transformer export (superseded)
6. ❌ `export_transformer_torchscript.py` - TorchScript export (not used)
7. ❌ `export_wsl.py` - WSL-specific export (not needed)
8. ❌ `export_and_optimize_separate.py` - Split workflow (superseded)

**Action**: Archive to `archive/legacy_exports/` or delete

### Redundant Utility Scripts (TO CONSOLIDATE)
1. ❌ `benchmark_onnx.py` - **Replaced** by `models/diarization/benchmark.py` (more comprehensive)
2. ⚠️ `convert_to_fp16.py` - Functionality in `export_onnx.py` (quantization flag)
3. ⚠️ `quantize_to_int8.py` - Functionality in `export_onnx.py` (quantization flag)

**Action**: Remove or consolidate into `tools/` directory if unique functionality exists

### Documentation Redundancy (TO CONSOLIDATE)
1. ⚠️ `ONNX_PERFORMANCE_SUMMARY.md` - Outdated performance claims (1428ms P99)
2. ⚠️ `STATUS.md` - Temporary summary document
3. ⚠️ `REFACTORING_SUMMARY.md` - Interim technical document
4. ✅ `REALITY_CHECK.md` - Keep (production assessment)
5. ✅ `models/diarization/README.md` - Keep (API documentation)

**Action**: Consolidate into comprehensive `docs/PERFORMANCE_ANALYSIS.md`

### Test Scripts (TO ORGANIZE)
1. ✅ `test_refactoring.py` - Move to `tests/integration/test_model_refactoring.py`
2. ✅ `test_gpu_inference.py` - Move to `tests/hardware/test_gpu_inference.py`
3. ⚠️ `deploy_and_test_gcp.py` - Move to `scripts/gcp/deploy_and_test.py`
4. ⚠️ `prepare_gcp_deployment.py` - Move to `scripts/gcp/prepare_deployment.py`

### Training Scripts (KEEP & DOCUMENT)
1. ✅ `train_transformer.py` - Sophisticated model training
2. ✅ `train_fast_cnn.py` - Fast CNN model training

**Action**: Add comprehensive inline documentation

### Generated Files (TO IGNORE)
1. ⚠️ `test_audio.wav` - Test artifact (add to .gitignore)
2. ⚠️ `test_audio_f32.bin` - Test artifact (add to .gitignore)

---

## 🎯 Proposed Final Structure

```
voiceflow-ml/
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
├── pytest.ini
│
├── models/
│   ├── __init__.py
│   ├── diarization/
│   │   ├── __init__.py
│   │   ├── model.py                 # ✅ Core model architectures (DOCUMENTED)
│   │   ├── export_onnx.py           # ✅ Unified ONNX export pipeline (DOCUMENTED)
│   │   ├── benchmark.py             # ✅ Benchmarking utility (DOCUMENTED)
│   │   └── README.md                # ✅ API documentation
│   │
│   └── checkpoints/                 # Model checkpoints (gitignored)
│       └── fast_cnn_diarization_best.pth
│
├── scripts/
│   ├── training/
│   │   ├── train_sophisticated.py   # Renamed from train_transformer.py (DOCUMENTED)
│   │   └── train_fast_cnn.py        # ✅ Fast CNN training (DOCUMENTED)
│   │
│   ├── gcp/
│   │   ├── deploy_and_test.py       # GCP deployment script (DOCUMENTED)
│   │   └── prepare_deployment.py    # GCP preparation script (DOCUMENTED)
│   │
│   └── utils/
│       └── generate_test_audio.py   # Test audio generation
│
├── tests/
│   ├── unit/
│   │   ├── test_model.py            # Unit tests for model.py
│   │   └── test_export.py           # Unit tests for export_onnx.py
│   │
│   ├── integration/
│   │   └── test_model_refactoring.py  # Integration tests
│   │
│   └── hardware/
│       ├── test_cpu_inference.py    # CPU performance tests
│       └── test_gpu_inference.py    # GPU performance tests
│
├── api/                             # FastAPI service
│   ├── __init__.py
│   ├── main.py
│   └── routes/
│
├── core/                            # Core utilities
│   ├── __init__.py
│   └── config.py
│
├── docs/                            # Consolidated documentation
│   ├── PERFORMANCE_ANALYSIS.md      # Consolidated performance docs
│   └── API_REFERENCE.md             # API documentation
│
└── archive/                         # Deprecated files (not in git)
    └── legacy_exports/
```

---

## 🔧 Implementation Steps

### Phase 1: Cleanup (30 min)
1. Create `archive/legacy_exports/` directory
2. Move all redundant export scripts to archive
3. Remove redundant benchmark/quantization scripts
4. Update .gitignore for test artifacts

### Phase 2: Reorganization (45 min)
1. Create `scripts/` subdirectories
2. Move and rename training scripts
3. Move GCP deployment scripts
4. Move test scripts to appropriate test/ subdirectories

### Phase 3: Documentation (90 min)
1. Add comprehensive inline docs to `model.py`
2. Add comprehensive inline docs to `export_onnx.py`
3. Add comprehensive inline docs to `benchmark.py`
4. Add comprehensive inline docs to training scripts
5. Consolidate performance docs into `docs/PERFORMANCE_ANALYSIS.md`
6. Update all README files with new paths

### Phase 4: SLA Validation (30 min)
1. Audit all latency claims across documentation
2. Create consistent terminology:
   - **Model inference**: 4.48ms P99 (ONNX on CPU)
   - **Rust processing**: ~5-10ms overhead
   - **Network latency**: 10-40ms typical
   - **End-to-end P99**: <100ms target (40-80ms typical)
3. Update README.md performance table
4. Update GCP_DEPLOYMENT.md expectations

### Phase 5: GCP T4 Testing (60 min)
1. Review GCP_DEPLOYMENT.md for accuracy
2. Create docker-compose.gpu.yml
3. Add CUDA provider configuration scripts
4. Create automated GPU benchmark script
5. Document expected T4 performance (10-20ms P99 model inference)

### Phase 6: Containerization Plan (45 min)
1. Design separate model training project structure
2. Define model versioning strategy (semantic versioning)
3. Create integration contract (ONNX file format, input/output specs)
4. Document CI/CD pipeline for model updates
5. Create hot-reload mechanism for Rust server

---

## 📋 Checklist

### Cleanup
- [ ] Archive 8 redundant export scripts
- [ ] Remove redundant benchmark/quantization scripts
- [ ] Add test artifacts to .gitignore
- [ ] Remove obsolete documentation files

### Reorganization
- [ ] Create scripts/ structure (training/, gcp/, utils/)
- [ ] Move training scripts and rename for clarity
- [ ] Move GCP deployment scripts
- [ ] Reorganize tests/ into unit/integration/hardware
- [ ] Update all import statements

### Documentation
- [ ] Document model.py (architecture, parameters, usage)
- [ ] Document export_onnx.py (pipeline, optimization, quantization)
- [ ] Document benchmark.py (metrics, providers, output)
- [ ] Document train_fast_cnn.py (hyperparameters, datasets, checkpoints)
- [ ] Document train_sophisticated.py
- [ ] Consolidate performance documentation
- [ ] Update main README with new structure

### SLA Validation
- [ ] Audit latency claims in README.md
- [ ] Audit latency claims in GCP_DEPLOYMENT.md
- [ ] Audit latency claims in models/diarization/README.md
- [ ] Audit latency claims in REALITY_CHECK.md
- [ ] Create consistent terminology document
- [ ] Update performance benchmark table

### GCP T4 Testing
- [ ] Review GCP_DEPLOYMENT.md accuracy
- [ ] Create docker-compose.gpu.yml
- [ ] Add CUDA provider scripts
- [ ] Create GPU benchmark automation
- [ ] Document expected T4 performance
- [ ] Create deployment verification tests

### Containerization
- [ ] Design model training project architecture
- [ ] Define model versioning strategy
- [ ] Document ONNX contract specifications
- [ ] Design CI/CD pipeline for models
- [ ] Design hot-reload mechanism
- [ ] Create integration documentation

---

## 🎯 Success Criteria

1. **Zero redundancy**: No duplicate functionality across scripts
2. **Clear structure**: Intuitive directory organization
3. **Comprehensive documentation**: Every critical file has inline comments explaining purpose, parameters, edge cases
4. **Consistent SLAs**: All documentation uses same latency terminology and targets
5. **Production-ready**: GCP T4 deployment tested and validated
6. **Professional quality**: No code smells, inconsistencies, or outdated references
7. **Maintainable**: New team members can understand structure in <30 minutes
