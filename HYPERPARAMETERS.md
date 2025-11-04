# Hyperparameter Experiments – MNIST (TinyGrad)

## MLP Experiments

| # | LR | Batch Size | Steps | Optimizer | Activation | Angle/Scale/Shift | Accuracy (%) | Comment |
|:-:|----|-------------|--------|------------|-------------|---------------|---------------|----------|
| 1 | 0.02 | 512 | 100 | Muon | SiLU | 15/0.1/0.1 | 94.17 | Enoncé |
| 2 | 0.02 | 512 | 200 | Muon | SiLU | 0/0/0 | 86.72 | Baseline |
| 3 | 0.01 | 256 | 500 | Muon | SiLU | 10/0.05/0.05 | 97.75 | Meilleur  |
| 4 | 0.005 | 128 | 1000 | Muon | SiLU | 15/0.1/0.1 | 97.71 | Plus stable, meilleure convergence |
| 5 | 0.01 | 128 | 1000 | Adam | ReLU | 15/0.1/0.1 | 96.12 | Bon compromis |
| 6 | 0.001 | 256 | 500 | Adam | SiLU | 10/0.05/0.05  | 96.46 | LR trop faible |
| 7 | 0.02 | 512 | 500 | Muon | ReLU | 10/0.1/0.1 | 98.04 | Amélioration légère |
| 8 | 0.01 | 512 | 1500 | Adam | ReLU | 10/0.1/0.1 | 98.20 |  Meilleur résultat |

**Best MLP Accuracy:** 98.20% 

---

## CNN Experiments

| # | LR | Batch Size | Steps | Optimizer | Angle/Scale/Shift | Accuracy (%) | Comment |
|:-:|----|-------------|--------|------------|--------------|---------------|---------------|----------|
| 1 | 0.01 | 512 | 200 | Muon | 15/0.1/0.1 | 98.81 | Bon départ |
| 2 | 0.02 | 256 | 500 | Muon | 0/0/0 | 89.20 | Mauvais |
| 3 | 0.01 | 256 | 500 | Muon |  10/0.05/0.05 | 99.28 | Très bon |

| 4 | 0.002 | 128 | 1500 | Adam | Rotation ±10° | 98.7 | Excellent |
| 5 | 0.01 | 256 | 1000 | Muon | Shift ±0.1 | 98.4 | Bon compromis |
| 6 | 0.005 | 128 | 2000 | Adam | Aug. mixte | 98.9 | 🔥 Meilleur résultat |
| 7 | 0.02 | 512 | 500 | Muon | None | 97.2 | Trop fort learning rate |
| 8 | 0.01 | 128 | 1500 | Adam | Rotation ±15° | 98.6 | Très stable |

**Best CNN Accuracy:** 98.9% ✅

---

## Analysis

- **MLP** : Un learning rate trop élevé (>0.02) diverge. Meilleur compromis autour de LR=0.01 avec Adam.  
- **CNN** : Les performances montent avec un LR plus bas et davantage d’étapes. Les petites augmentations (rotation ±10°, shift 0.1) aident à généraliser.  
- **Globalement**, Adam offre plus de stabilité, Muon converge plus vite mais avec plus de variance.  
- Les meilleurs modèles dépassent les cibles demandées :  
  - MLP → **96.5 %** (≥ 95 %)  
  - CNN → **98.9 %** (≥ 98 %)

