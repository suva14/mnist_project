let net;
const models = ["mnist_mlp", "mnist_convnet"];
const modelSelect = document.getElementById("model");
const statusText = document.getElementById("status");
const timerText = document.getElementById("timer");
const predText = document.getElementById("prediction");
const timeText = document.getElementById("inference-time");
const canvas = document.getElementById("draw-canvas");
const ctx = canvas.getContext("2d");

// Init canvas
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Drawing state
let drawing = false;
let erasing = false;
let needsPredict = false;
let predicting = false;

// Mouse events
canvas.addEventListener("mousedown", () => (drawing = true));
canvas.addEventListener("mouseup", () => {
  drawing = false;
  needsPredict = true;
});
canvas.addEventListener("mousemove", (e) => draw(e));

// Touch events for mobile
canvas.addEventListener("touchstart", (e) => {
  e.preventDefault();
  drawing = true;
});
canvas.addEventListener("touchend", (e) => {
  e.preventDefault();
  drawing = false;
  needsPredict = true;
});
canvas.addEventListener("touchmove", (e) => {
  e.preventDefault();
  const touch = e.touches[0];
  const rect = canvas.getBoundingClientRect();
  const x = touch.clientX - rect.left;
  const y = touch.clientY - rect.top;
  
  if (drawing) {
    ctx.fillStyle = erasing ? "black" : "white";
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, Math.PI * 2);
    ctx.fill();
  }
});

function draw(e) {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  ctx.fillStyle = erasing ? "black" : "white";
  ctx.beginPath();
  ctx.arc(x, y, 10, 0, Math.PI * 2);
  ctx.fill();
}

document.getElementById("pen").onclick = () => (erasing = false);
document.getElementById("eraser").onclick = () => (erasing = true);
document.getElementById("clear").onclick = () => {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  predText.textContent = "-";
  timeText.textContent = "-";
  updateChart(Array(10).fill(0));
};

// GPU + model loader
const error = (err) => {
  statusText.textContent = `Error: ${err}`;
  console.error(err);
};

const timer = async (func, label = "") => {
  const start = performance.now();
  const out = await func();
  const delta = (performance.now() - start).toFixed(1);
  timerText.textContent = `${delta} ms ${label}`;
  return out;
};

const getDevice = async () => {
  if (!navigator.gpu) {
    error("WebGPU not supported");
    return null;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    error("No GPU adapter found");
    return null;
  }
  return await adapter.requestDevice({
    requiredFeatures: ["shader-f16"],
    powerPreference: "high-performance",
  });
};

const loadNet = async (modelName) => {
  const jsPath = `./${modelName}/${modelName}.js`;
  const netPath = `./${modelName}/${modelName}.webgpu.safetensors`;
  try {
    statusText.textContent = "Loading model...";
    const device = await getDevice();
    if (!device) return;
    
    const tinygrad = (await import(jsPath)).default;
    net = await timer(() => tinygrad.load(device, netPath), "(compilation)");
    statusText.textContent = "Ready!";
  } catch (e) {
    error(`Failed to load: ${e.message}`);
  }
};

// Softmax helper
function softmax(arr) {
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / sum);
}

// Prediction (optimized to avoid crashes)
async function predict() {
  if (!net || predicting) return;
  
  predicting = true;
  
  try {
    // Resize to 28x28
    const small = document.createElement("canvas");
    small.width = 28;
    small.height = 28;
    const sctx = small.getContext("2d");
    sctx.drawImage(canvas, 0, 0, 28, 28);
    
    const imgData = sctx.getImageData(0, 0, 28, 28);
    
    const input = Float32Array.from(
      imgData.data.filter((_, i) => i % 4 === 0),
      (v) => v / 127.5 - 1.0
    );

    const start = performance.now();
    const res = await net(input);
    const delta = (performance.now() - start).toFixed(1);
    
    const logits = Array.from(new Float32Array(res[0]));
    const probs = softmax(logits);
    const pred = probs.indexOf(Math.max(...probs));

    predText.textContent = pred;
    timeText.textContent = delta;
    updateChart(probs);
  } catch (e) {
    console.error("Prediction error:", e);
  } finally {
    predicting = false;
  }
}

// Prediction loop (debounced)
setInterval(() => {
  if (needsPredict && !predicting) {
    needsPredict = false;
    predict();
  }
}, 300); // Prédit max toutes les 300ms

// Chart.js setup (taille fixe)
const chartCtx = document.getElementById("prob-chart").getContext("2d");
const chart = new Chart(chartCtx, {
  type: "bar",
  data: {
    labels: [...Array(10).keys()],
    datasets: [
      { 
        label: "Probability", 
        data: Array(10).fill(0), 
        backgroundColor: "#3b82f6" 
      },
    ],
  },
  options: {
    scales: { 
      y: { 
        min: 0, 
        max: 1
      } 
    },
    animation: false,
  },
});

function updateChart(probs) {
  chart.data.datasets[0].data = probs;
  chart.update();
}

// Setup models
async function setup() {
  for (const model of models) {
    const opt = document.createElement("option");
    opt.value = model;
    opt.textContent = model;
    modelSelect.appendChild(opt);
  }
  
  modelSelect.addEventListener("change", (e) => loadNet(e.target.value));
  await loadNet(models[0]);
}

setup();