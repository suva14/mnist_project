# Hyperparameter Experiments – MNIST (TinyGrad)

This document logs the hyperparameter exploration for both the **MLP** and **CNN** models trained on the MNIST dataset using **TinyGrad**.  
Each experiment records the main training parameters, data augmentation, and resulting test accuracy.

---

## MLP Experiments

| # | LR | Batch Size | Steps | Optimizer | Activation | Angle/Scale/Shift | Accuracy (%) | Comment |
|:-:|----|-------------|--------|------------|-------------|---------------|---------------|----------|
| 1 | 0.02 | 512 | 100 | Muon | SiLU | 15/0.1/0.1 | 94.17 | Instructor baseline |
| 2 | 0.02 | 512 | 200 | Muon | SiLU | 0/0/0 | 86.72 | No augmentation → severe underfitting |
| 3 | 0.01 | 256 | 500 | Muon | SiLU | 10/0.05/0.05 | 97.75 | Lower LR + more steps → better convergence |
| 4 | 0.005 | 128 | 1000 | Muon | SiLU | 15/0.1/0.1 | 97.71 | LR too low slows convergence |
| 5 | 0.01 | 128 | 1000 | Adam | ReLU | 15/0.1/0.1 | 96.12 | Adam less effective than Muon for MLP |
| 6 | 0.001 | 256 | 500 | Adam | SiLU | 10/0.05/0.05 | 96.46 | Very small LR requires more steps |
| 7 | 0.02 | 512 | 500 | Muon | ReLU | 10/0.1/0.1 | 98.04 | ReLU converges faster than SiLU |
| 8 | 0.01 | 512 | 1500 | Adam | ReLU | 10/0.1/0.1 | **98.20** | **Best: balance between steps/LR/augmentation** |

**Best MLP Accuracy:** 98.20%

---

## CNN Experiments

| # | LR | Batch Size | Steps | Optimizer | Activation | Angle/Scale/Shift | Accuracy (%) | Comment |
|:-:|----|-------------|--------|------------|--------------|---------------|---------------|----------|
| 1 | 0.01 | 512 | 200 | Muon | ReLU | 15/0.1/0.1 | 98.81 | Good starting point, CNN very effective |
| 2 | 0.02 | 256 | 500 | Muon | ReLU | 0/0/0 | 89.20 | No augmentation → overfitting despite CNN |
| 3 | 0.01 | 256 | 500 | Muon | ReLU | 10/0.05/0.05 | 99.28 | Moderate augmentation is optimal |
| 4 | 0.005 | 128 | 1000 | Adam | ReLU | 15/0.1/0.1 | **99.45** | **Best: low LR + longer training** |
| 5 | 0.005 | 64 | 1500 | Adam | ReLU | 0/0.1/0.1 | 99.45 | Equivalent to #4 but 2× slower (batch=64) |
| 6 | 0.02 | 256 | 1000 | Adam | SiLU | 10/0.1/0.1 | 99.21 | SiLU slower to converge than ReLU |
| 7 | 0.01 | 512 | 500 | Muon | SiLU | 10/0.05/0.05 | 99.41 | Muon competitive for mid-range training |
| 8 | 0.005 | 128 | 2000 | Adam | ReLU | 10/0.1/0.1 | 99.31 | Too long, no significant gain vs #4 |

**Best CNN Accuracy:** 99.45%

---

## MLP vs CNN Comparison

| Metric | MLP Best | CNN Best | Winner |
|----------|----------|----------|---------|
| Accuracy | 98.20% | 99.45% | **CNN (+1.25%)** |
| Required Steps | 1500 | 1000 | **CNN (33% faster)** |
| Parameters | ~670K | ~1.2M | MLP (lighter) |
| Robustness | Moderate | Excellent | **CNN** |

**Conclusion:**  
The CNN outperforms the MLP in both precision and learning efficiency, at the cost of a heavier model.  
For MNIST, the CNN is clearly superior.

---

## Final Hyperparameters

**For MLP:**
- Optimal configuration: LR=0.01, BS=512, Steps=1500, Adam, ReLU  
- Moderate augmentation: 10°/0.1/0.1

**For CNN:**
- Optimal configuration: LR=0.005, BS=128, Steps=1000, Adam, ReLU  
- Strong augmentation: 15°/0.1/0.1

**Note:** Geometric augmentation appears to be the main factor influencing performance for both architectures.
