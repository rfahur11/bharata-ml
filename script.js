// Wait for DOM to load
document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("file-input");
  const imageElement = document.getElementById("input-image");
  const predictButton = document.getElementById("predict-button");
  const outputElement = document.getElementById("output");

  let session;

  // Load ONNX model
  async function loadModel() {
    try {
      outputElement.innerText = "Loading model...";
      session = await ort.InferenceSession.create("./model.onnx"); // Update path to your model
      outputElement.innerText =
        "Model loaded successfully. Please upload an image.";
    } catch (error) {
      console.error("Error loading the model:", error);
      outputElement.innerText =
        "Failed to load the model. Check the console for errors.";
    }
  }

  // Initialize model loading
  loadModel();

  // Handle file input change
  fileInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        imageElement.src = e.target.result;
        imageElement.classList.remove("hidden");
        predictButton.disabled = false;
      };
      reader.readAsDataURL(file);
    }
  });

  // Handle prediction button click
  predictButton.addEventListener("click", async () => {
    if (!session) {
      outputElement.innerText = "Model not loaded yet.";
      return;
    }

    outputElement.innerText = "Processing image...";

    try {
      // Preprocess image
      const tensor = await preprocessImage(imageElement);

      // Create ONNX input
      const input = { input: tensor };

      // Run inference
      const output = await session.run(input);

      // Process prediction
      const predictions = output[Object.keys(output)[0]].data; // Extract output tensor
      const predictedIndex = predictions.indexOf(Math.max(...predictions));

      // Define labels (adjust based on your model)
      const labels = [
        "ba",
        "ca",
        "da",
        "dha",
        "ga",
        "ha",
        "ja",
        "ka",
        "la",
        "ma",
        "na",
        "nga",
        "nya",
        "pa",
        "ra",
        "sa",
        "ta",
        "tha",
        "wa",
        "ya",
      ]; // Replace with actual labels
      outputElement.innerText = `Prediction: ${labels[predictedIndex]}`;

      tensor.dispose(); // Free memory
    } catch (error) {
      console.error("Error during prediction:", error);
      outputElement.innerText =
        "Error during prediction. Check the console for details.";
    }
  });

  // Preprocess the input image
  async function preprocessImage(img) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    // Resize image to match model input size
    const width = 128; // Replace with your model input width
    const height = 128; // Replace with your model input height
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(img, 0, 0, width, height);

    // Get image data
    const imageData = ctx.getImageData(0, 0, width, height);
    const { data } = imageData;

    // Normalize image data to [0, 1]
    const normalizedData = new Float32Array(width * height * 3);
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i] / 255;
      const g = data[i + 1] / 255;
      const b = data[i + 2] / 255;

      const idx = (i / 4) * 3;
      normalizedData[idx] = r;
      normalizedData[idx + 1] = g;
      normalizedData[idx + 2] = b;
    }

    // Create a tensor
    return new ort.Tensor("float32", normalizedData, [1, 3, height, width]); // [batch, channels, height, width]
  }
});
