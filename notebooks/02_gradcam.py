# ============================================================
# GRAD-CAM VISUALIZATION
# Menampilkan area daun yang menjadi fokus model saat prediksi
# ============================================================

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# Konfigurasi
# ─────────────────────────────────────────────
MODEL_PATH  = "models/best_model.keras"
LABEL_PATH  = "models/label_map.json"
GRADCAM_DIR = "gradcam_results"
IMG_SIZE    = 224

os.makedirs(GRADCAM_DIR, exist_ok=True)

# Load model dan label map
print("Memuat model...")
model = keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH) as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Cari nama layer konvolusi terakhir EfficientNetB0
# Biasanya 'top_conv' atau 'block7a_project_conv'
LAST_CONV_LAYER = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        LAST_CONV_LAYER = layer.name
        break
print(f"Layer konvolusi terakhir: {LAST_CONV_LAYER}")

# ─────────────────────────────────────────────
# Fungsi Grad-CAM
# ─────────────────────────────────────────────
def get_gradcam_heatmap(
    model: keras.Model,
    img_array: np.ndarray,
    last_conv_layer_name: str,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """
    Menghasilkan Grad-CAM heatmap untuk gambar input.
    
    Args:
        model             : Model Keras yang sudah dilatih
        img_array         : Array gambar shape (1, H, W, 3), sudah dinormalisasi
        last_conv_layer_name: Nama layer konvolusi terakhir
        pred_index        : Indeks kelas target (None = kelas prediksi tertinggi)
    
    Returns:
        heatmap: np.ndarray shape (H, W) nilai antara 0-1
    """
    # Buat submodel: input → conv layer output + final prediction
    grad_model = keras.Model(
        inputs  = model.inputs,
        outputs = [
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ]
    )

    with tf.GradientTape() as tape:
        last_conv_output, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradien output kelas terhadap feature map konvolusi terakhir
    grads = tape.gradient(class_channel, last_conv_output)

    # Pooling gradien secara spasial (Global Average Pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Bobot feature map dengan gradien
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalisasi ke rentang [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay heatmap Grad-CAM di atas gambar asli.
    
    Args:
        original_img: Gambar BGR (OpenCV) atau RGB
        heatmap     : Heatmap array shape (H, W) nilai 0-1
        alpha       : Transparansi overlay (0=tidak terlihat, 1=solid)
        colormap    : OpenCV colormap (COLORMAP_JET / COLORMAP_HOT / COLORMAP_INFERNO)
    
    Returns:
        superimposed: Gambar RGB dengan heatmap overlay
    """
    # Resize heatmap ke ukuran gambar asli
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    # Konversi ke colormap
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Pastikan gambar asli dalam uint8
    if original_img.max() <= 1.0:
        img_uint8 = np.uint8(255 * original_img)
    else:
        img_uint8 = original_img.copy()

    superimposed = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_rgb, alpha, 0)
    return superimposed


def predict_and_gradcam(
    image_path: str,
    model: keras.Model,
    idx_to_class: dict,
    last_conv_layer: str,
    img_size: int = 224,
    top_k: int = 3,
    save_path: Optional[str] = None,
    show: bool = True,
) -> dict:
    """
    Prediksi satu gambar dan tampilkan Grad-CAM.
    
    Returns dict berisi prediksi top-k dan confidence score.
    """
    # ── Load & preprocess ──
    img_bgr  = cv2.imread(image_path)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res  = cv2.resize(img_rgb, (img_size, img_size))
    img_norm = img_res.astype("float32")
    img_arr  = np.expand_dims(img_norm, axis=0)

    # ── Prediksi ──
    preds     = model.predict(img_arr, verbose=0)[0]
    top_idx   = np.argsort(preds)[::-1][:top_k]
    top_preds = [(idx_to_class[i], float(preds[i])) for i in top_idx]

    pred_idx   = top_idx[0]
    pred_class = top_preds[0][0]
    confidence = top_preds[0][1]

    # ── Grad-CAM ──
    heatmap   = get_gradcam_heatmap(model, img_arr, last_conv_layer, pred_index=pred_idx)
    overlay   = overlay_gradcam(img_res, heatmap, alpha=0.45)

    # ── Visualisasi ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")

    axes[0].imshow(img_res)
    axes[0].set_title("Gambar Asli", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    pred_label = pred_class.replace("___", " — ").replace("_", " ")
    axes[2].set_title(f"Overlay: {pred_label}\nConf: {confidence:.2%}", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # Bar chart prediksi top-k di bawah
    fig2, ax = plt.subplots(figsize=(8, 3))
    names  = [c.replace("___", "\n").replace("_", " ")[:40] for c, _ in top_preds]
    scores = [s for _, s in top_preds]
    colors = ["#2563EB" if i == 0 else "#94A3B8" for i in range(len(scores))]
    bars   = ax.barh(names[::-1], scores[::-1], color=colors[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence Score")
    ax.set_title(f"Top-{top_k} Prediksi", fontweight="bold")
    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{score:.2%}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        fig2.savefig(save_path.replace(".png", "_topk.png"), dpi=150, bbox_inches="tight")
        print(f"Hasil disimpan ke: {save_path}")

    if show:
        plt.show()

    plt.close("all")

    return {
        "image_path"  : image_path,
        "prediction"  : pred_class,
        "confidence"  : confidence,
        "top_k"       : top_preds,
    }


# ─────────────────────────────────────────────
# Visualisasi batch — beberapa gambar sekaligus
# ─────────────────────────────────────────────
def batch_gradcam(
    image_paths: list,
    model: keras.Model,
    idx_to_class: dict,
    last_conv_layer: str,
    img_size: int = 224,
    cols: int = 3,
    save_path: str = "gradcam_results/batch_gradcam.png",
):
    """Membuat grid Grad-CAM untuk banyak gambar sekaligus."""
    n     = len(image_paths)
    rows  = n
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for i, img_path in enumerate(image_paths):
        img_bgr  = cv2.imread(img_path)
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res  = cv2.resize(img_rgb, (img_size, img_size))
        img_norm = img_res.astype("float32")
        img_arr  = np.expand_dims(img_norm, axis=0)

        preds    = model.predict(img_arr, verbose=0)[0]
        pred_idx = np.argmax(preds)
        pred_cls = idx_to_class[pred_idx].replace("___", " — ").replace("_", " ")
        conf     = preds[pred_idx]

        heatmap  = get_gradcam_heatmap(model, img_arr, last_conv_layer, pred_index=pred_idx)
        overlay  = overlay_gradcam(img_res, heatmap, alpha=0.45)

        axes[i, 0].imshow(img_res)
        axes[i, 0].set_title("Original", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[i, 1].set_title("Heatmap", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"{pred_cls}\n{conf:.2%}", fontsize=9)
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Batch Grad-CAM disimpan ke {save_path}")


# ─────────────────────────────────────────────
# CONTOH PENGGUNAAN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Ganti dengan path gambar daun yang ingin dianalisis
    TEST_IMAGES = [
        "data/plantvillage dataset/color/Tomato___Bacterial_spot/00416c7d-0b85-46a5-befd-b1adc7a48234___GCREC_Bact.Sp 3083.JPG",
        # Tambah path gambar lain di sini...
    ]

    for img_path in TEST_IMAGES:
        if os.path.exists(img_path):
            result = predict_and_gradcam(
                image_path      = img_path,
                model           = model,
                idx_to_class    = idx_to_class,
                last_conv_layer = LAST_CONV_LAYER,
                top_k           = 5,
                save_path       = f"{GRADCAM_DIR}/gradcam_{Path(img_path).stem[:30]}.png",
                show            = True,
            )
            print("\nHasil prediksi:")
            print(f"  Kelas    : {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Top-5    :")
            for cls, conf in result["top_k"]:
                print(f"    {cls:<50} {conf:.4f}")
        else:
            print(f"File tidak ditemukan: {img_path}")
