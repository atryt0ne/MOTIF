# MOTIF Stochastic Hypergraph Sampling - Project Plan

## Project Goal
Implement stochastic motif sampling for MOTIF's relation hypergraph construction to enable scaling to larger knowledge graphs. The key contribution is showing that sampling a subset of motifs preserves model accuracy while providing significant speedup.

---

## Repository Information

- **Upstream:** https://github.com/HxyScotthuang/MOTIF
- **Fork:** https://github.com/atryt0ne/MOTIF
- **Working Directory:** `D:\Coding_Workspace\GDL\MOTIF`

---

## Completed Work

### 1. Preprocessing Timing Analysis ✅
- Identified bottleneck: 3-path motif computation (99% of time/memory)
- WN18RR baseline: 50s preprocessing, 96M motif instances
- Created timing notebook: `motif_preprocessing_timing.ipynb`

### 2. Implemented Sampling Functions ✅
- Added `sample_sparse_coo()` in `motif/tasks.py` (lines 504-545)
- Added `build_relation_hypergraph_sampled()` in `motif/tasks.py` (lines 548-756)
- Sampling is proportional stratified (each motif type sampled independently)

### 3. Verified Sampling Correctness ✅
- Tested sample_prob=[1.0, 0.5, 0.25, 0.1]
- All ratios ~1.0 (proportional reduction verified)
- Preprocessing speedup achieved: 6x at p=0.25

### 4. Fixed Colab Compatibility Issues ✅
- Fixed `graph.device` AttributeError (PyG Data objects don't have `.device`)
- Disabled Triton kernels (`use_triton=False` in `motif/layers.py`)
- Changes pushed to fork

---

## Current Blocker: OOM Error During Training

### Error Location
```
/content/MOTIF/motif/layers.py in all_but_one_trick()
--> shifted_backward = torch.cat([...])
OutOfMemoryError: CUDA out of memory
```

### Root Cause
The HypergraphLayer's `all_but_one_trick` function computes cumulative products over all nodes in each hyperedge. With 18K hyperedges and batch processing, this creates large intermediate tensors.

### Memory Analysis
- GPU: 14.56 GiB T4
- PyTorch allocated: 13.96 GiB
- Free: 125 MiB
- Failing allocation: 438 MiB

---

## Solutions to Try (In Priority Order)

### Option A: Reduce Batch Size
Current batch_size=64. Try:
- [ ] batch_size=16
- [ ] batch_size=8
- [ ] batch_size=4

### Option B: Reduce Hidden Dimensions
Current hidden_dims=[64, 64, 64, 64, 64, 64]. Try:
- [ ] hidden_dims=[32, 32, 32, 32, 32, 32]
- [ ] hidden_dims=[64, 64, 64]

### Option C: Reduce Number of Negatives
Current num_negative=256. Try:
- [ ] num_negative=64
- [ ] num_negative=32

### Option D: Gradient Checkpointing
- [ ] Implement gradient checkpointing in HypergraphLayer
- [ ] Trade compute for memory

### Option E: Mixed Precision Training
- [ ] Use `torch.cuda.amp` for FP16 training
- [ ] Reduces memory by ~50%

### Option F: Sample Even More Aggressively
- [ ] Test with sample_prob=0.1 or 0.05
- [ ] Fewer hyperedges = less memory in message passing

### Option G: CPU Training (Slower but works)
- [ ] Set device='cpu'
- [ ] Will be slow but can validate the approach works

---

## Remaining Tasks

### Immediate (Unblock Training)
1. [ ] Modify training config in notebook to use smaller batch_size
2. [ ] Add `num_negative` parameter to training functions
3. [ ] Test with reduced hyperparameters until OOM is resolved
4. [ ] Once training works with p=1.0, test with p=0.5, p=0.25

### Training Experiments
1. [ ] Run training with sample_prob=1.0 (baseline)
2. [ ] Run training with sample_prob=0.5
3. [ ] Run training with sample_prob=0.25
4. [ ] Run training with sample_prob=0.1

### Evaluation
1. [ ] Compare MRR across different sample_prob values
2. [ ] Compare Hits@1, Hits@3, Hits@10
3. [ ] Compare training time per epoch
4. [ ] Create summary table and plots

### Documentation
1. [ ] Document final results in `subsampling_notes.txt`
2. [ ] Create "Pareto Frontier" plot (accuracy vs speed)
3. [ ] Write up contribution summary

---

## Key Files

| File | Purpose |
|------|---------|
| `motif/tasks.py` | Contains sampling functions |
| `motif/layers.py` | HypergraphLayer (OOM location) |
| `motif/models.py` | MOTIF model definition |
| `motif_preprocessing_timing.ipynb` | Main experiment notebook |
| `subsampling_notes.txt` | Technical notes and results |

---

## How to Continue

1. Read this PLAN.md file
2. Open `motif_preprocessing_timing.ipynb` in Colab
3. Try solutions from "Solutions to Try" section above
4. Start with reducing batch_size in the training config
5. Once training works, run experiments and compare MRR

---

## Expected Outcome

If sampling works, we expect:
- sample_prob=1.0: Baseline MRR (e.g., 0.45 on WN18RR)
- sample_prob=0.5: ~Same MRR, 2x faster
- sample_prob=0.25: ~Same MRR, 4x faster (ideal outcome)
- sample_prob=0.1: Slight MRR drop, 10x faster

If we can show "90% accuracy with 75% fewer motifs", that's a valuable contribution for scaling MOTIF to larger graphs.
