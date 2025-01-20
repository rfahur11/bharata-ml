const modelPath = "model.onnx";

const aksaraLabels = [
  "Ha",
  "Na",
  "Ca",
  "Ra",
  "Ka",
  "Da",
  "Ta",
  "Sa",
  "Wa",
  "La",
  "Pa",
  "Dha",
  "Ja",
  "Ya",
  "Nya",
  "Ma",
  "Ga",
  "Ba",
  "Tha",
  "Nga",
];

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let isDrawing = false;

// Event untuk menggambar di canvas
canvas.addEventListener("mousedown", () => (isDrawing = true));
canvas.addEventListener("mouseup", () => (isDrawing = false));
canvas.addEventListener("mousemove", draw);

// Fungsi menggambar di canvas
function draw(event) {
  if (!isDrawing) return;
  ctx.lineWidth = 10;
  ctx.lineCap = "round";
  ctx.strokeStyle = "black";

  ctx.lineTo(event.offsetX, event.offsetY);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(event.offsetX, event.offsetY);
}

// Fungsi untuk preprocess input dari canvas ke bentuk tensor
function preprocessCanvas() {
  // Ambil data gambar dari canvas
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  // Resize canvas ke 128x128 (ukuran input model)
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = 128;
  tempCanvas.height = 128;
  const tempCtx = tempCanvas.getContext("2d");
  tempCtx.drawImage(canvas, 0, 0, 128, 128);

  // Ambil data gambar yang sudah di-resize
  const resizedImgData = tempCtx.getImageData(0, 0, 128, 128);

  // Convert ke RGB float32 (tensor shape: [1, 3, 128, 128])
  const floatData = new Float32Array(3 * 128 * 128);
  for (let i = 0; i < resizedImgData.data.length; i += 4) {
    const r = resizedImgData.data[i] / 255.0;
    const g = resizedImgData.data[i + 1] / 255.0;
    const b = resizedImgData.data[i + 2] / 255.0;

    const index = i / 4;
    floatData[index] = r; // Channel R
    floatData[index + 128 * 128] = g; // Channel G
    floatData[index + 2 * 128 * 128] = b; // Channel B
  }

  return new ort.Tensor("float32", floatData, [1, 3, 128, 128]);
}

// Fungsi untuk menjalankan prediksi dengan model ONNX
async function runModel() {
  const session = await ort.InferenceSession.create(modelPath);
  const inputTensor = preprocessCanvas();

  // Persiapkan input untuk model
  const feeds = { [session.inputNames[0]]: inputTensor };

  // Jalankan model
  const results = await session.run(feeds);
  const output = results[session.outputNames[0]].data;

  // Cari label dengan probabilitas tertinggi
  const maxIndex = output.indexOf(Math.max(...output));
  const predictedLabel = aksaraLabels[maxIndex];

  // Tampilkan hasil
  document.getElementById(
    "result"
  ).textContent = `Aksara Jawa: ${predictedLabel}`;
}

// Event listener untuk tombol prediksi
document.getElementById("predict-btn").addEventListener("click", runModel);
