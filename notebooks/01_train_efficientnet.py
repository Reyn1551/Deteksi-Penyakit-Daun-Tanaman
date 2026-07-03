# ============================================================
# PROJECT: Deteksi Penyakit Daun Tanaman
# Model   : EfficientNet-B0 (Transfer Learning)
# Dataset : PlantVillage (via Kaggle)
# Target  : >= 97% Accuracy
# ============================================================

# ─────────────────────────────────────────────
# STEP 0 — Install dependencies
# ─────────────────────────────────────────────
# Jalankan di terminal / Kaggle/Colab cell:
# pip install tensorflow opencv-python matplotlib seaborn scikit-learn kaggle

# ─────────────────────────────────────────────
# STEP 1 — Download Dataset dari Kaggle
# ─────────────────────────────────────────────
# Option A: Kaggle CLI (pastikan kaggle.json sudah dikonfigurasi)
# kaggle datasets download -d abdallahalidev/plantvillage-dataset
# unzip plantvillage-dataset.zip -d data/

# Option B: Download manual dari
# https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
# Letakkan folder 'plantvillage dataset' di direktori kerja

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import cv2
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────
# STEP 2 — Konfigurasi (sesuaikan path-mu)
# ─────────────────────────────────────────────
CONFIG = {
    "DATA_DIR"      : "data/plantvillage dataset/color",  # Folder gambar berwarna
    "MODEL_SAVE"    : "models/best_model.keras",
    "LABEL_MAP"     : "models/label_map.json",
    "IMG_SIZE"      : 224,
    "BATCH_SIZE"    : 32,
    "EPOCHS"        : 30,
    "LEARNING_RATE" : 1e-3,
    "FINE_TUNE_LR"  : 1e-5,
    "FINE_TUNE_AT"  : 100,    # Layer ke-N ke atas yang di-unfreeze saat fine-tune
    "SEED"          : 42,
    "VAL_SPLIT"     : 0.2,
}

# Reproducibility
random.seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])
tf.random.set_seed(CONFIG["SEED"])

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print(f"TensorFlow version : {tf.__version__}")
print(f"GPU tersedia       : {tf.config.list_physical_devices('GPU')}")

# ─────────────────────────────────────────────
# STEP 3 — Data Loading & Augmentation
# ─────────────────────────────────────────────
IMG_SIZE    = CONFIG["IMG_SIZE"]
BATCH_SIZE  = CONFIG["BATCH_SIZE"]

# Augmentasi untuk training — mencegah overfitting
train_datagen = ImageDataGenerator(
    rescale            = 1.0 / 255,
    validation_split   = CONFIG["VAL_SPLIT"],
    rotation_range     = 40,
    width_shift_range  = 0.2,
    height_shift_range = 0.2,
    shear_range        = 0.2,
    zoom_range         = 0.2,
    horizontal_flip    = True,
    vertical_flip      = True,
    fill_mode          = "nearest",
)

# Validasi: hanya rescale, tanpa augmentasi
val_datagen = ImageDataGenerator(
    rescale          = 1.0 / 255,
    validation_split = CONFIG["VAL_SPLIT"],
)

train_gen = train_datagen.flow_from_directory(
    CONFIG["DATA_DIR"],
    target_size  = (IMG_SIZE, IMG_SIZE),
    batch_size   = BATCH_SIZE,
    class_mode   = "categorical",
    subset       = "training",
    seed         = CONFIG["SEED"],
    shuffle      = True,
)

val_gen = val_datagen.flow_from_directory(
    CONFIG["DATA_DIR"],
    target_size  = (IMG_SIZE, IMG_SIZE),
    batch_size   = BATCH_SIZE,
    class_mode   = "categorical",
    subset       = "validation",
    seed         = CONFIG["SEED"],
    shuffle      = False,
)

NUM_CLASSES = train_gen.num_classes
CLASS_NAMES = list(train_gen.class_indices.keys())
print(f"\nJumlah kelas      : {NUM_CLASSES}")
print(f"Total training    : {train_gen.samples}")
print(f"Total validasi    : {val_gen.samples}")

# Simpan label map
with open(CONFIG["LABEL_MAP"], "w") as f:
    json.dump(train_gen.class_indices, f, indent=2)
print(f"Label map disimpan ke {CONFIG['LABEL_MAP']}")

# ─────────────────────────────────────────────
# STEP 4 — Visualisasi Sample Data
# ─────────────────────────────────────────────
def visualize_samples(generator, class_names, n=9):
    images, labels = next(generator)
    idx_to_class = {v: k for k, v in generator.class_indices.items()}
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            label_idx = np.argmax(labels[i])
            # Format nama kelas agar lebih mudah dibaca
            class_name = idx_to_class[label_idx].replace("___", "\n").replace("_", " ")
            ax.set_title(class_name, fontsize=9, pad=4)
        ax.axis("off")
    plt.suptitle("Sample Gambar Training", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("outputs/sample_images.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Sampel gambar disimpan ke outputs/sample_images.png")

visualize_samples(train_gen, CLASS_NAMES)

# ─────────────────────────────────────────────
# STEP 5 — Membangun Model EfficientNet-B0
# ─────────────────────────────────────────────

def build_model(num_classes: int, img_size: int = 224) -> keras.Model:
    """
    Transfer Learning dengan EfficientNet-B0.
    Fase 1 : Freeze semua layer base, train classifier head.
    Fase 2 : Unfreeze sebagian, fine-tune end-to-end.
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), name="input_image")

    # Base model — pretrained pada ImageNet
    base_model = EfficientNetB0(
        include_top    = False,
        input_tensor   = inputs,
        weights        = "imagenet",
        pooling        = None,
    )
    base_model.trainable = False  # Freeze untuk Fase 1

    # Head classifier
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    x = layers.Dense(512, activation="relu", name="dense_512")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="PlantDisease_EfficientNetB0")
    return model, base_model


model, base_model = build_model(NUM_CLASSES, IMG_SIZE)
model.summary()

print(f"\nTotal parameter      : {model.count_params():,}")
print(f"Parameter trainable  : {sum(tf.size(v).numpy() for v in model.trainable_variables):,}")

# ─────────────────────────────────────────────
# STEP 6 — FASE 1: Training Head Classifier
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("FASE 1 — Training Head Classifier (base frozen)")
print("="*60)

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=CONFIG["LEARNING_RATE"]),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
)

callbacks_phase1 = [
    ModelCheckpoint(
        CONFIG["MODEL_SAVE"],
        monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.3, patience=3, verbose=1, min_lr=1e-7
    ),
]

history_phase1 = model.fit(
    train_gen,
    epochs              = 15,
    validation_data     = val_gen,
    callbacks           = callbacks_phase1,
    verbose             = 1,
)

# ─────────────────────────────────────────────
# STEP 7 — FASE 2: Fine-tuning
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("FASE 2 — Fine-tuning (unfreeze top layers)")
print("="*60)

# Unfreeze layer dari CONFIG["FINE_TUNE_AT"] ke atas
base_model.trainable = True
for layer in base_model.layers[:CONFIG["FINE_TUNE_AT"]]:
    layer.trainable = False

trainable_count = sum(1 for l in model.layers if l.trainable)
print(f"Layer yang di-unfreeze: {trainable_count}")

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=CONFIG["FINE_TUNE_LR"]),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
)

callbacks_phase2 = [
    ModelCheckpoint(
        CONFIG["MODEL_SAVE"],
        monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.3, patience=4, verbose=1, min_lr=1e-8
    ),
]

history_phase2 = model.fit(
    train_gen,
    epochs              = CONFIG["EPOCHS"],
    initial_epoch       = len(history_phase1.history["accuracy"]),
    validation_data     = val_gen,
    callbacks           = callbacks_phase2,
    verbose             = 1,
)

# ─────────────────────────────────────────────
# STEP 8 — Visualisasi Kurva Training
# ─────────────────────────────────────────────
def plot_training_history(h1, h2):
    acc    = h1.history["accuracy"]    + h2.history["accuracy"]
    val_acc= h1.history["val_accuracy"]+ h2.history["val_accuracy"]
    loss   = h1.history["loss"]        + h2.history["loss"]
    val_loss=h1.history["val_loss"]    + h2.history["val_loss"]
    epochs = range(1, len(acc) + 1)
    phase_boundary = len(h1.history["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs, acc,     "b-o", markersize=3, label="Train Acc")
    ax1.plot(epochs, val_acc, "r-o", markersize=3, label="Val Acc")
    ax1.axvline(phase_boundary, color="gray", linestyle="--", label="Fine-tune start")
    ax1.set_title("Akurasi Training vs Validasi")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    ax2.plot(epochs, loss,     "b-o", markersize=3, label="Train Loss")
    ax2.plot(epochs, val_loss, "r-o", markersize=3, label="Val Loss")
    ax2.axvline(phase_boundary, color="gray", linestyle="--", label="Fine-tune start")
    ax2.set_title("Loss Training vs Validasi")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Kurva disimpan ke outputs/training_curves.png")
    print(f"Val Accuracy terbaik: {max(val_acc):.4f}")

plot_training_history(history_phase1, history_phase2)

# ─────────────────────────────────────────────
# STEP 9 — Evaluasi & Confusion Matrix
# ─────────────────────────────────────────────
print("\nMemuat model terbaik...")
best_model = keras.models.load_model(CONFIG["MODEL_SAVE"])

print("Evaluasi pada validation set...")
val_gen.reset()
y_pred_prob = best_model.predict(val_gen, verbose=1)
y_pred      = np.argmax(y_pred_prob, axis=1)
y_true      = val_gen.classes

# Laporan per kelas
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
print("\n" + report)
with open("outputs/classification_report.txt", "w") as f:
    f.write(report)

# Confusion Matrix (top 15 kelas agar terbaca)
cm_matrix = confusion_matrix(y_true, y_pred)
cm_norm   = cm_matrix.astype("float") / cm_matrix.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(20, 18))
sns.heatmap(
    cm_norm, annot=True, fmt=".2f", cmap="Blues",
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    linewidths=0.3, ax=ax
)
ax.set_title("Confusion Matrix (Normalized)", fontsize=14)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0,  fontsize=6)
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("Confusion matrix disimpan ke outputs/confusion_matrix.png")

overall_acc = np.sum(y_pred == y_true) / len(y_true)
print(f"\nOverall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
