# Hyperparameter Experiments – MNIST (TinyGrad)

## MLP Experiments

| # | LR | Batch Size | Steps | Optimizer | Activation | Angle/Scale/Shift | Accuracy (%) | Comment |
|:-:|----|-------------|--------|------------|-------------|---------------|---------------|----------|
| 1 | 0.02 | 512 | 100 | Muon | SiLU | 15/0.1/0.1 | 94.17 | Enoncé |
| 2 | 0.02 | 512 | 200 | Muon | SiLU | 0/0/0 | 86.72 | Baseline |
| 3 | 0.01 | 256 | 500 | Muon | SiLU | 10/0.05/0.05 | 97.75 | Meilleur  |
| 4 | 0.005 | 128 | 1000 | Muon | SiLU | 15/0.1/0.1 | 97.71 | Bon|
| 5 | 0.01 | 128 | 1000 | Adam | ReLU | 15/0.1/0.1 | 96.12 | Bon compromis |
| 6 | 0.001 | 256 | 500 | Adam | SiLU | 10/0.05/0.05  | 96.46 | LR trop faible |
| 7 | 0.02 | 512 | 500 | Muon | ReLU | 10/0.1/0.1 | 98.04 | Amélioration légère |
| 8 | 0.01 | 512 | 1500 | Adam | ReLU | 10/0.1/0.1 | 98.20 |  Meilleur résultat |

**Best MLP Accuracy:** 98.20% 


## CNN Experiments

| # | LR | Batch Size | Steps | Optimizer | Activation | Angle/Scale/Shift | Accuracy (%) | Comment |
|:-:|----|-------------|--------|------------|--------------|---------------|---------------|----------|
| 1 | 0.01 | 512 | 200 | Muon | ReLU | 15/0.1/0.1 | 98.81 | Bon départ |
| 2 | 0.02 | 256 | 500 | Muon | ReLU | 0/0/0 | 89.20 | Mauvais |
| 3 | 0.01 | 256 | 500 | Muon | ReLU | 10/0.05/0.05 | 99.28 | Très bon |

| 4 | 0.005 | 128 | 1000 | Adam | ReLU | 15/0.1/0.1 | 99.45 | Excellent |
| 5 | 0.005 | 64 | 1500 | Adam | ReLU | 0/0.1/0.1| 99.45 | Excellent mais lent |
| 6 | 0.02 | 256 | 1000 | Adam | SiLU | 10/0.1/0.1 | 99.21 | Bon résultat Très lent|
| 7 | 0.01 | 512 | 500 | Muon | SiLU | 10/0.05/0.05 | 99.41 | Très bien |
| 8 | 0.005 | 128 | 2000 | Adam | ReLU | 10/0.1/0.1 | 99.31 | très long|

**Best CNN Accuracy:** 99.45% 


