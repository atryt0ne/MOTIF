# Stochastic Hypergraph Sampling for MOTIF - Final Report

## Objective
Test whether stochastic motif sampling can speed up MOTIF training while preserving accuracy.

---

## What We Did

1. **Implemented proportional stratified sampling** - Sample each motif type independently with probability `p`
2. **Added `sample_sparse_coo()` and `build_relation_hypergraph_sampled()`** in `motif/tasks.py`
3. **Fixed Colab compatibility** - Device detection, disabled Triton kernels
4. **Ran training experiments** on WN18RR with p=[0.25, 0.5, 1.0]

---

## Results

| sample_prob | Edges | Pre(s) | Train(s) | MRR | Rel.MRR |
|-------------|-------|--------|----------|-----|---------|
| 0.25 | 4,712 | 0.21 | 2546 | 0.308 | 77.3% |
| 0.50 | 9,334 | 0.02 | 2558 | 0.363 | 90.9% |
| 1.00 | 18,652 | 0.02 | 2581 | 0.399 | 100% |

---

## Key Findings

**❌ Sampling hurts accuracy**: 23% MRR drop at p=0.25, 9% drop at p=0.5

**❌ No training speedup**: Training time ~2550s regardless of sample_prob

**✓ Preprocessing is fast**: ~0.02s for hypergraph construction (negligible)

---

## Conclusion

**Stochastic hypergraph sampling does not work for MOTIF.**

The preprocessing was never the bottleneck - training message passing is. Sampling motifs reduces hypergraph size but the model still processes all entities and relations during training. The accuracy drop indicates MOTIF needs dense motif information to learn effectively.

---

## Repository

Fork: https://github.com/atryt0ne/MOTIF

---

## Takeaway

This negative result is still valuable: it shows MOTIF's expressiveness depends on complete motif coverage, and future scaling efforts should focus on the entity encoder rather than the hypergraph preprocessor.
