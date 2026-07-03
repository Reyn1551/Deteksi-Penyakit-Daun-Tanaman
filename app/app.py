"""
app/app.py — Web App Deteksi Penyakit Daun Tanaman
Deploy lokal dengan Flask, atau ke Hugging Face / Render

Jalankan:
    pip install flask tensorflow opencv-python pillow
    python app/app.py
"""

import os
import io
import json
import base64
import numpy as np
import cv2
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
from tensorflow import keras

# ─────────────────────────────────────────────
# Inisialisasi app
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB max upload

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.keras")
LABEL_PATH = os.getenv("LABEL_PATH", "models/label_map.json")
IMG_SIZE   = 224

print(f"Memuat model dari {MODEL_PATH}...")
model = keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH) as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Cari layer konvolusi terakhir untuk Grad-CAM
LAST_CONV_LAYER = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        LAST_CONV_LAYER = layer.name
        break
print(f"Model dimuat. Layer Grad-CAM: {LAST_CONV_LAYER}")

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Konversi bytes gambar ke array siap inferensi."""
    nparr    = np.frombuffer(file_bytes, np.uint8)
    img_bgr  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res  = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_input = img_res.astype("float32")
    return img_rgb, img_res, np.expand_dims(img_input, axis=0)


def get_gradcam_heatmap(img_array: np.ndarray, pred_index: int) -> np.ndarray:
    grad_model = keras.Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(LAST_CONV_LAYER).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_channel = predictions[:, pred_index]
    grads       = tape.gradient(class_channel, conv_output)
    pooled_grads= tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap     = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap     = tf.squeeze(heatmap)
    heatmap     = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_b64(img_rgb_resized: np.ndarray, heatmap: np.ndarray) -> str:
    """Buat overlay Grad-CAM dan kembalikan sebagai base64 PNG."""
    heatmap_resized = cv2.resize(heatmap, (img_rgb_resized.shape[1], img_rgb_resized.shape[0]))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    img_uint8       = np.uint8(img_rgb_resized)
    overlay         = cv2.addWeighted(img_uint8, 0.55, heatmap_rgb, 0.45, 0)

    _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")


def img_to_b64(img_rgb: np.ndarray) -> str:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buf  = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode("utf-8")


def format_class_name(raw: str) -> dict:
    """Pisahkan nama tanaman dan penyakit dari format PlantVillage."""
    parts   = raw.split("___")
    plant   = parts[0].replace("_", " ") if len(parts) > 0 else raw
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in disease.lower()
    return {"plant": plant, "disease": disease, "is_healthy": is_healthy}


# ─────────────────────────────────────────────
# HTML Template (single-file frontend)
# ─────────────────────────────────────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Deteksi Penyakit Daun Tanaman</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f0fdf4; color: #1a1a1a; }
  
  header { background: #15803d; color: white; padding: 1.25rem 2rem;
           display: flex; align-items: center; gap: 1rem; }
  header h1 { font-size: 1.4rem; font-weight: 600; }
  header span { font-size: 1.8rem; }
  
  .container { max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  
  .upload-area { border: 2px dashed #86efac; background: white; border-radius: 12px;
                 padding: 3rem 2rem; text-align: center; cursor: pointer;
                 transition: border-color 0.2s; }
  .upload-area:hover, .upload-area.dragover { border-color: #16a34a; background: #f0fdf4; }
  .upload-area input { display: none; }
  .upload-area .icon { font-size: 3rem; margin-bottom: 1rem; }
  .upload-area p { color: #6b7280; font-size: 0.95rem; margin-top: 0.5rem; }
  
  .btn { background: #16a34a; color: white; border: none; padding: 0.75rem 2rem;
         border-radius: 8px; font-size: 1rem; cursor: pointer; margin-top: 1rem;
         transition: background 0.2s; }
  .btn:hover { background: #15803d; }
  .btn:disabled { background: #9ca3af; cursor: not-allowed; }
  
  .preview img { max-width: 300px; border-radius: 8px; margin-top: 1rem; border: 1px solid #e5e7eb; }
  
  .result-card { background: white; border-radius: 12px; padding: 1.5rem;
                 margin-top: 2rem; border: 1px solid #e5e7eb; display: none; }
  .result-card.show { display: block; }
  
  .status-badge { display: inline-block; padding: 0.35rem 0.85rem; border-radius: 99px;
                  font-weight: 600; font-size: 0.85rem; margin-bottom: 1rem; }
  .healthy   { background: #dcfce7; color: #15803d; }
  .diseased  { background: #fee2e2; color: #dc2626; }
  
  h2.plant-name { font-size: 1.3rem; color: #15803d; }
  p.disease-name { font-size: 1.05rem; color: #374151; margin: 0.3rem 0 1rem; }
  
  .confidence-bar { background: #f3f4f6; border-radius: 99px; height: 10px; overflow: hidden; margin: 0.3rem 0 0.1rem; }
  .confidence-fill { height: 100%; border-radius: 99px; background: #16a34a; transition: width 0.6s; }
  
  .topk-list { list-style: none; padding: 0; }
  .topk-list li { display: flex; justify-content: space-between; padding: 0.4rem 0;
                  border-bottom: 1px solid #f3f4f6; font-size: 0.9rem; }
  .topk-list li:last-child { border-bottom: none; }
  .topk-score { font-weight: 600; color: #16a34a; }
  
  .gradcam-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
  .gradcam-grid img { width: 100%; border-radius: 8px; border: 1px solid #e5e7eb; }
  .gradcam-grid p { font-size: 0.8rem; color: #6b7280; text-align: center; margin-top: 0.3rem; }
  
  .loader { display: none; text-align: center; padding: 2rem; }
  .spinner { width: 40px; height: 40px; border: 4px solid #86efac;
             border-top-color: #16a34a; border-radius: 50%; animation: spin 0.8s linear infinite; margin: auto; }
  @keyframes spin { to { transform: rotate(360deg); } }
  
  .section-title { font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
                   color: #9ca3af; letter-spacing: 0.05em; margin: 1.25rem 0 0.5rem; }
  
  @media (max-width: 600px) { .gradcam-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <span>🌿</span>
  <div>
    <h1>Deteksi Penyakit Daun Tanaman</h1>
    <p style="font-size:0.8rem; opacity:0.85">EfficientNet-B0 + Grad-CAM · PlantVillage Dataset</p>
  </div>
</header>

<div class="container">
  <div class="upload-area" id="dropZone">
    <input type="file" id="fileInput" accept="image/*">
    <div class="icon">📷</div>
    <strong>Klik atau seret gambar daun di sini</strong>
    <p>Format: JPG, PNG, WEBP · Maks. 10MB</p>
    <div class="preview" id="preview"></div>
    <button class="btn" id="analyzeBtn" onclick="analyzeImage()" disabled>🔍 Analisis Penyakit</button>
  </div>
  
  <div class="loader" id="loader">
    <div class="spinner"></div>
    <p style="margin-top:1rem; color:#6b7280">Menganalisis gambar...</p>
  </div>
  
  <div class="result-card" id="resultCard">
    <div id="statusBadge" class="status-badge"></div>
    <h2 class="plant-name" id="plantName"></h2>
    <p class="disease-name" id="diseaseName"></p>
    
    <p class="section-title">Tingkat Keyakinan (Top Prediksi)</p>
    <div class="confidence-bar">
      <div class="confidence-fill" id="confBar" style="width:0%"></div>
    </div>
    <p id="confText" style="font-size:0.85rem; color:#6b7280; margin-bottom:0.5rem"></p>
    
    <p class="section-title">Top-5 Prediksi</p>
    <ul class="topk-list" id="topkList"></ul>
    
    <p class="section-title">Grad-CAM — Area Fokus Model</p>
    <div class="gradcam-grid">
      <div>
        <img id="origImg" src="" alt="Gambar asli">
        <p>Gambar asli</p>
      </div>
      <div>
        <img id="camImg" src="" alt="Grad-CAM overlay">
        <p>Grad-CAM overlay (merah = area paling berpengaruh)</p>
      </div>
    </div>
  </div>
</div>

<script>
let selectedFile = null;

document.getElementById("dropZone").addEventListener("click", e => {
  if (e.target.id !== "analyzeBtn") document.getElementById("fileInput").click();
});

document.getElementById("fileInput").addEventListener("change", e => {
  selectedFile = e.target.files[0];
  if (selectedFile) showPreview(selectedFile);
});

["dragenter","dragover"].forEach(evt => {
  document.getElementById("dropZone").addEventListener(evt, e => {
    e.preventDefault(); document.getElementById("dropZone").classList.add("dragover");
  });
});
["dragleave","drop"].forEach(evt => {
  document.getElementById("dropZone").addEventListener(evt, e => {
    e.preventDefault(); document.getElementById("dropZone").classList.remove("dragover");
    if (evt === "drop" && e.dataTransfer.files.length) {
      selectedFile = e.dataTransfer.files[0];
      showPreview(selectedFile);
    }
  });
});

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById("preview").innerHTML =
      `<img src="${e.target.result}" alt="Preview">`;
    document.getElementById("analyzeBtn").disabled = false;
    document.getElementById("resultCard").classList.remove("show");
  };
  reader.readAsDataURL(file);
}

async function analyzeImage() {
  if (!selectedFile) return;
  document.getElementById("loader").style.display = "block";
  document.getElementById("resultCard").classList.remove("show");
  document.getElementById("analyzeBtn").disabled = true;

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const res  = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    displayResult(data);
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    document.getElementById("loader").style.display = "none";
    document.getElementById("analyzeBtn").disabled = false;
  }
}

function displayResult(data) {
  const badge = document.getElementById("statusBadge");
  badge.textContent = data.is_healthy ? "✅ Sehat" : "⚠️ Terdeteksi Penyakit";
  badge.className   = "status-badge " + (data.is_healthy ? "healthy" : "diseased");

  document.getElementById("plantName").textContent   = "Tanaman: " + data.plant;
  document.getElementById("diseaseName").textContent = "Kondisi: " + data.disease;

  const pct = Math.round(data.confidence * 100);
  document.getElementById("confBar").style.width = pct + "%";
  document.getElementById("confText").textContent  = `Keyakinan model: ${pct}%`;

  const topkList = document.getElementById("topkList");
  topkList.innerHTML = data.top_k.map(([cls, conf]) => {
    const parts = cls.split("___");
    const label = (parts[0] + " — " + (parts[1] || "")).replace(/_/g, " ");
    return `<li><span>${label}</span><span class="topk-score">${(conf*100).toFixed(1)}%</span></li>`;
  }).join("");

  document.getElementById("origImg").src = "data:image/png;base64," + data.original_b64;
  document.getElementById("camImg").src  = "data:image/png;base64," + data.gradcam_b64;

  document.getElementById("resultCard").classList.add("show");
  document.getElementById("resultCard").scrollIntoView({ behavior: "smooth" });
}
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada file gambar yang dikirim."}), 400

    file       = request.files["image"]
    file_bytes = file.read()

    try:
        img_rgb_orig, img_rgb_res, img_array = preprocess_image(file_bytes)
    except Exception as e:
        return jsonify({"error": f"Gagal memproses gambar: {str(e)}"}), 400

    # Prediksi
    preds    = model.predict(img_array, verbose=0)[0]
    top_idx  = np.argsort(preds)[::-1][:5]
    top_preds= [(idx_to_class[i], float(preds[i])) for i in top_idx]

    pred_idx   = int(top_idx[0])
    pred_class = idx_to_class[pred_idx]
    confidence = float(preds[pred_idx])
    info       = format_class_name(pred_class)

    # Grad-CAM
    heatmap    = get_gradcam_heatmap(img_array, pred_idx)
    gradcam_b64= overlay_heatmap_b64(img_rgb_res, heatmap)
    original_b64= img_to_b64(img_rgb_res)

    return jsonify({
        "prediction"   : pred_class,
        "plant"        : info["plant"],
        "disease"      : info["disease"],
        "is_healthy"   : info["is_healthy"],
        "confidence"   : confidence,
        "top_k"        : top_preds,
        "gradcam_b64"  : gradcam_b64,
        "original_b64" : original_b64,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH, "classes": len(idx_to_class)})


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"\n🌿 Aplikasi berjalan di http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
