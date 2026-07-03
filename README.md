# Deteksi Penyakit Daun Tanaman
### Transfer Learning EfficientNet-B0 + Grad-CAM · PlantVillage Dataset
**Target Akurasi: ≥ 97%**

---

## Struktur Project

```
plant_disease_project/
├── notebooks/
│   ├── 01_train_efficientnet.py    ← Training lengkap (2-fase)
│   └── 02_gradcam.py               ← Visualisasi Grad-CAM
├── app/
│   └── app.py                      ← Web app Flask untuk demo
├── models/                         ← Model & label map (dibuat saat training)
│   ├── best_model.keras
│   └── label_map.json
├── outputs/                        ← Kurva training, confusion matrix
├── gradcam_results/                ← Hasil visualisasi Grad-CAM
├── data/                           ← Dataset (download dari Kaggle)
│   └── plantvillage dataset/
│       └── color/                  ← 38 folder kelas penyakit
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone / Setup environment

```bash
# Buat virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

**Via Kaggle CLI:**
```bash
# Pasang API key Kaggle:
# Letakkan kaggle.json di ~/.kaggle/kaggle.json
# chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d data/
```

**Manual:** Download dari https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset  
Letakkan di folder `data/plantvillage dataset/`

### 3. Training Model

```bash
python notebooks/01_train_efficientnet.py
```

Proses training terdiri dari **2 fase**:
- **Fase 1** (~15 epoch): Freeze base EfficientNet, train head classifier → akurasi awal ~90%
- **Fase 2** (~20 epoch): Unfreeze top layers, fine-tune end-to-end → akurasi ≥97%

Output yang dihasilkan:
- `models/best_model.keras` — model terbaik
- `models/label_map.json` — mapping kelas
- `outputs/training_curves.png` — kurva akurasi & loss
- `outputs/confusion_matrix.png` — confusion matrix
- `outputs/classification_report.txt` — precision/recall per kelas

### 4. Grad-CAM Visualization

```bash
# Edit bagian TEST_IMAGES di file, lalu:
python notebooks/02_gradcam.py
```

### 5. Jalankan Web App

```bash
python app/app.py
# Buka browser: http://localhost:5000
```

---

## Arsitektur Model

```
Input (224×224×3)
      │
      ▼
EfficientNetB0 (pretrained ImageNet)
  ├── 237 layer total
  └── Top layers di-unfreeze saat fine-tune
      │
      ▼
GlobalAveragePooling2D
      │
BatchNormalization
      │
Dropout(0.4)
      │
Dense(512, relu)
      │
BatchNormalization
      │
Dropout(0.3)
      │
Dense(38, softmax) ← 38 kelas penyakit
      │
      ▼
  Prediksi
```

**Mengapa EfficientNet-B0?**
- Sangat efisien: hanya 5.3M parameter, lebih ringan dari VGG/ResNet
- Compound scaling: lebar, kedalaman, dan resolusi dioptimalkan bersama
- Pretrained ImageNet memberikan fitur tekstur yang kuat untuk daun
- Terbukti mencapai 97–99% pada PlantVillage di berbagai paper

---

## Dataset PlantVillage

| Properti         | Detail                                    |
|------------------|-------------------------------------------|
| Total gambar     | ~54,309 gambar                            |
| Jumlah kelas     | 38 kelas (14 tanaman, 26 penyakit + sehat)|
| Format           | JPG, resolusi bervariasi                  |
| Split            | 80% train / 20% validasi                 |
| License          | Open Source (CC BY 4.0)                   |

**Contoh kelas:**
- `Tomato___Bacterial_spot`
- `Apple___Apple_scab`
- `Potato___Early_blight`
- `Corn___Northern_Leaf_Blight`
- `Tomato___healthy`

---

## Grad-CAM — Explainable AI

Grad-CAM (Gradient-weighted Class Activation Mapping) menunjukkan **area mana pada gambar daun yang paling berpengaruh** terhadap prediksi model.

**Cara kerja:**
1. Ambil gradien output kelas terhadap feature map layer konvolusi terakhir
2. Pool gradien secara global (average pooling)
3. Bobot feature map dengan gradien tersebut
4. Overlay sebagai heatmap berwarna di atas gambar asli

**Interpretasi warna:**
- 🔴 **Merah** = area paling penting untuk prediksi
- 🟡 **Kuning** = area cukup penting
- 🔵 **Biru** = area kurang relevan

---

## 📈 Target Performa

| Metrik            | Target    | Catatan                         |
|-------------------|-----------|---------------------------------|
| Val Accuracy      | ≥ 97%     | Dengan fine-tuning EfficientNet |
| Top-5 Accuracy    | ≥ 99.5%   | Hampir pasti benar di top-5     |
| Inference time    | < 100ms   | CPU biasa, per gambar           |
| Model size        | ~25 MB    | .keras format                   |

---

## Cara Deploy

### A. Lokal (Flask)
```bash
python app/app.py
# http://localhost:5000
```

### B. Hugging Face Spaces (Gratis, Public)
```bash
# 1. Buat akun di huggingface.co
# 2. Buat Space baru dengan SDK: Gradio atau Streamlit
# 3. Upload model dan kode

pip install gradio
# Buat app_gradio.py (lihat contoh di bawah)
```

**app_gradio.py (alternatif deploy HF):**
```python
import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json, cv2

model = keras.models.load_model("models/best_model.keras")
with open("models/label_map.json") as f:
    idx_to_class = {v: k for k, v in json.load(f).items()}

def predict(image):
    img = cv2.resize(image, (224, 224)) / 255.0
    preds = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    return {idx_to_class[i].replace("___", " — "): float(preds[i])
            for i in np.argsort(preds)[::-1][:5]}

gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=5),
    title="Plant Disease Detection",
    description="Upload gambar daun tanaman untuk deteksi penyakit",
).launch()
```

### C. Render / Railway (Cloud)
```bash
# Tambahkan Procfile:
echo "web: python app/app.py" > Procfile

# Set environment variable:
# MODEL_PATH=models/best_model.keras
# PORT=8080
```

---

## Tips Meningkatkan Akurasi

| Teknik                     | Dampak        | Cara                                              |
|----------------------------|---------------|---------------------------------------------------|
| Data augmentation kuat     | +1–2%         | Sudah diimplementasikan di script training        |
| Label smoothing            | +0.5%         | `CategoricalCrossentropy(label_smoothing=0.1)`    |
| Learning rate warmup       | +0.5–1%       | `WarmupCosineDecay` scheduler                     |
| Test Time Augmentation     | +0.5%         | Prediksi 5–10 augmentasi, rata-rata probabilitas  |
| Ensemble 2–3 model         | +1–2%         | EfficientNetB0 + MobileNetV2 + ResNet50           |
| Mixed Precision (FP16)     | 2× lebih cepat| `tf.keras.mixed_precision.set_global_policy('mixed_float16')` |

---

## Nilai Inovatif Project

1. **Grad-CAM Explainability** — Tidak hanya prediksi, tapi *menjelaskan mengapa* model memutuskan demikian, relevan untuk petani dan peneliti
2. **Two-Phase Transfer Learning** — Strategi freeze → unfreeze yang efisien dan mencegah catastrophic forgetting
3. **Web App Siap Deploy** — Langsung bisa digunakan di lapangan via browser
4. **38 Kelas Penyakit** — Cakupan luas: 14 tanaman berbeda, siap untuk aplikasi pertanian nyata

---

## Referensi

- Mohanty et al. (2016) — *Using Deep Learning for Image-Based Plant Disease Detection*, Frontiers in Plant Science
- Tan & Le (2019) — *EfficientNet: Rethinking Model Scaling for CNNs*, ICML
- Selvaraju et al. (2017) — *Grad-CAM: Visual Explanations from Deep Networks*, ICCV
- PlantVillage Dataset: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
# Deteksi-Penyakit-Daun-Tanaman
# Deteksi-Penyakit-Daun-Tanaman
