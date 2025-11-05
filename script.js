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
ctx.lineWidth = 2;
ctx.lineCap = "round";
ctx.lineJoin = "round";

// Drawing state
let drawing = false;
let erasing = false;
let needsPredict = false;
let predicting = false;
let lastX = 0;
let lastY = 0;

// Better pen size for recognition
const PEN_SIZE = 15; // Plus gros pour mieux ressembler à MNIST
const ERASER_SIZE = 20;

// Mouse events
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseleave", stopDrawing);

// Touch events for mobile
canvas.addEventListener("touchstart", (e) => {
  e.preventDefault();
  const touch = e.touches[0];
  const rect = canvas.getBoundingClientRect();
  lastX = touch.clientX - rect.left;
  lastY = touch.clientY - rect.top;
  drawing = true;
});

canvas.addEventListener("touchend", (e) => {
  e.preventDefault();
  stopDrawing();
});

canvas.addEventListener("touchmove", (e) => {
  e.preventDefault();
  if (!drawing) return;
  
  const touch = e.touches[0];
  const rect = canvas.getBoundingClientRect();
  const x = touch.clientX - rect.left;
  const y = touch.clientY - rect.top;
  
  drawLine(lastX, lastY, x, y);
  lastX = x;
  lastY = y;
});

function startDrawing(e) {
  drawing = true;
  const rect = canvas.getBoundingClientRect();
  lastX = e.clientX - rect.left;
  lastY = e.clientY - rect.top;
}

function stopDrawing() {
  if (drawing) {
    drawing = false;
    needsPredict = true;
  }
}

function draw(e) {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  
  drawLine(lastX, lastY, x, y);
  lastX = x;
  lastY = y;
}

function drawLine(x1, y1, x2, y2) {
  ctx.strokeStyle = erasing ? "black" : "white";
  ctx.lineWidth = erasing ? ERASER_SIZE : PEN_SIZE;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
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
    error(`Failed: ${e.message}`);
  }
};

// Softmax
function softmax(arr) {
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / sum);
}

// Better preprocessing (center and crop)
function preprocessCanvas() {
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imgData.data;
  
  // Find bounding box of drawing
  let minX = canvas.width, minY = canvas.height;
  let maxX = 0, maxY = 0;
  
  for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
      const i = (y * canvas.width + x) * 4;
      if (data[i] > 50) { // If pixel is bright
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }
  
  // Add padding (20% on each side)
  const pad = 20;
  minX = Math.max(0, minX - pad);
  minY = Math.max(0, minY - pad);
  maxX = Math.min(canvas.width, maxX + pad);
  maxY = Math.min(canvas.height, maxY + pad);
  
  const width = maxX - minX;
  const height = maxY - minY;
  
  // If canvas is empty
  if (width <= 0 || height <= 0) {
    return new Float32Array(784).fill(-1);
  }
  
  // Crop and center
  const temp = document.createElement("canvas");
  temp.width = temp.height = 28;
  const tctx = temp.getContext("2d");
  
  tctx.fillStyle = "black";
  tctx.fillRect(0, 0, 28, 28);
  
  // Scale to fit in 20x20 box (leave 4px border)
  const scale = Math.min(20 / width, 20 / height);
  const scaledW = width * scale;
  const scaledH = height * scale;
  const offsetX = (28 - scaledW) / 2;
  const offsetY = (28 - scaledH) / 2;
  
  tctx.drawImage(canvas, minX, minY, width, height, 
                 offsetX, offsetY, scaledW, scaledH);
  
  // Convert to array
  const resized = tctx.getImageData(0, 0, 28, 28);
  return Float32Array.from(
    resized.data.filter((_, i) => i % 4 === 0),
    (v) => v / 127.5 - 1.0
  );
}

// Prediction
async function predict() {
  if (!net || predicting) return;
  
  predicting = true;
  
  try {
    const input = preprocessCanvas();
    
    const start = performance.now();
    const res = await net(input);
    const delta = (performance.now() - start).toFixed(1);
    
    const logits = Array.from(new Float32Array(res[0]));
    const probs = softmax(logits);
    const pred = probs.indexOf(Math.max(...probs));
    const conf = (Math.max(...probs) * 100).toFixed(0);

    predText.textContent = `${pred} (${conf}%)`;
    timeText.textContent = delta;
    updateChart(probs);
  } catch (e) {
    console.error("Prediction error:", e);
  } finally {
    predicting = false;
  }
}

// Prediction loop
setInterval(() => {
  if (needsPredict && !predicting) {
    needsPredict = false;
    predict();
  }
}, 300);

// Chart
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

// Setup
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