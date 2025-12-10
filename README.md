# MNIST Digit Classifier 

![Project Screenshot](mnist.gif)

*Interactive handwritten digit recognition powered by TinyGrad and WebGPU*

##  Live Demo

** [Try it live on GitHub Pages](https://suva14.github.io/mnist_project/)**

---

##  Overview

This project implements a complete machine learning pipeline from training to deployment:
- **Training**: Two neural networks (MLP and CNN) trained on MNIST using [TinyGrad](https://github.com/tinygrad/tinygrad)
- **Export**: Models compiled to WebGPU shaders for browser execution
- **Web App**: Interactive single-page application where users draw digits and get real-time predictions

The entire inference runs **client-side** using WebGPU, enabling fast predictions (~10-20ms) without server calls.

---

##  Features

-  **Interactive Drawing Canvas** with pen, eraser, and clear tools
-  **Two Model Architectures**: Switch between MLP and CNN
-  **Real-time Visualization**: Probability bar chart for all 10 digits (0-9)
-  **GPU-Accelerated Inference**: WebGPU for maximum performance
-  **Fully Responsive**: Works on desktop, tablet, and mobile
-  **High Accuracy**: 98.20% (MLP) and 99.45% (CNN) on test set

---

##  Model Summary

| Model | Test Accuracy | 
|-------|-------------|
| **MLP** | **98.20%** |
| **CNN** | **99.45%** |

---

## 🛠️ Technologies Used

- **Training Framework**: [TinyGrad](https://github.com/tinygrad/tinygrad) - Minimalist deep learning framework
- **Compute Backend**: [WebGPU](https://www.w3.org/TR/webgpu/) - Modern GPU API for the web
- **Frontend**: Vanilla JavaScript with [Tailwind CSS](https://tailwindcss.com/)
- **Visualization**: [Chart.js](https://www.chartjs.org/) for probability plots
- **Dataset**: [MNIST](http://yann.lecun.com/exdb/mnist/) 

---

##  Quick Start

### Prerequisites

- Python 3.11+
- `pip install tinygrad`
- WebGPU-compatible browser (Chrome/Edge 113+, Firefox Nightly, Safari with enabled WebGPU)

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/suva14/mnist_project.git
cd mnist_project
```

2. **Train models** (optional, pre-trained models included)
```bash
# Train MLP (98.20% accuracy)
STEPS=1500 JIT=1 python mnist_mlp.py

# Train CNN (99.45% accuracy)
STEPS=1000 JIT=1 python mnist_convnet.py
```

3. **Start local server**
```bash
python -m http.server 8000
```
OR use Live Server Extension on VSCode.
4. **Open in browser**
```
http://localhost:8000
```

---

##  Project Structure

```
mnist_project/
├── index.html              # Main web application
├── script.js               # Inference logic and UI
├── mnist_mlp.py            # MLP training script
├── mnist_convnet.py        # CNN training script
├── export_model.py         # WebGPU export utilities
├── mnist_mlp/              # Exported MLP model
│   ├── mnist_mlp.js
│   └── mnist_mlp.webgpu.safetensors
├── mnist_convnet/          # Exported CNN model
│   ├── mnist_convnet.js
│   └── mnist_convnet.webgpu.safetensors
├── HYPERPARAMETERS.md      # Experiment log and analysis
└── README.md
```

---

##  Training Details

### Data Augmentation
To improve generalization, training uses geometric transformations:
- **Rotation**: ±15°
- **Scale**: ±10%
- **Translation**: ±10%

### Best Hyperparameters

**MLP:**
- Learning Rate: 0.01
- Batch Size: 512
- Steps: 1500
- Optimizer: Adam
- Activation: ReLU

**CNN:**
- Learning Rate: 0.005
- Batch Size: 128
- Steps: 1000
- Optimizer: Adam
- Activation: ReLU

For full experimental results, see [HYPERPARAMETERS.md](HYPERPARAMETERS.md).

---

##  Usage Guide

1. **Select a model** from the dropdown (MLP or CNN)
2. **Draw a digit** (0-9) on the black canvas using your mouse or finger
3. **View prediction** - The model classifies your digit in real-time
4. **Check confidence** - Bar chart shows probability for each digit
5. **Clear canvas** to try another digit

**Tips for best results:**
- Draw in the center of the canvas
- Make digits large and clear
- Try both models to compare performance
- Make sure WebGPU is enabled on the browser

---

##  Performance Metrics

| Metric | MLP | CNN |
|--------|-----|-----|
| Training Time | ~5 min (1500 steps) | ~8 min (1000 steps) |
| Model Size | 2.55 MB | 3.42 MB |
| Inference Time | ~5ms | ~10ms |


##  Project Retrospective

### Technical Challenges

1. **Windows File Locking**: Initial PermissionError when saving checkpoints. Solved by using timestamped temporary files and retry logic.

2. **Color Inversion Bug**: Canvas used black background with white drawing, but initial normalization inverted colors. Fixed by removing the `(255 - v)` inversion step.

3. **WebGPU Compatibility**: Ensured shader-f16 feature is available and handled graceful fallback for unsupported browsers.




