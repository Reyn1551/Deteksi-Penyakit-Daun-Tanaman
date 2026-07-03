# ============================================================
# PROJECT: Deteksi Penyakit Daun Tanaman — OPTIMIZED VERSION
# Model   : EfficientNet-B0 (Transfer Learning)
# Dataset : PlantVillage (via Kaggle)
# Target  : >= 97% Accuracy, ~2-3x faster training
# ============================================================
#
# OPTIMASI untuk GPU NVIDIA (RTX series):
#   1. Mixed Precision (FP16) — 2x throughput di RTX Tensor Cores
#   2. tf.data pipeline — hapus CPU bottleneck dari ImageDataGenerator
#   3. XLA compilation — fused ops, less kernel launch overhead
#   4. Larger batch size — GPU utilization naik drastis
#   5. Prefetch + parallel I/O — data loading overlap dengan compute
#
# Speedup yang diharapkan: 7 menit/epoch → ~2-3 menit/epoch
# ============================================================

import os
import random
import time
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
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from sklearn.metrics import classification_report, confusion_matrix

# ════════════════════════════════════════════════════════════
# STEP 0 — GPU Detection & Setup
# ════════════════════════════════════════════════════════════
gpus = tf.config.list_physical_devices("GPU")
print(f"TensorFlow version : {tf.__version__}")
print(f"GPU detected       : {len(gpus)} GPU(s)")
for gpu in gpus:
    print(f"  → {gpu}")

if gpus:
    # Enable memory growth (hindari OOM, bagus untuk laptop)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✓ Memory growth enabled")

    # ─── OPTIMASI 1: Mixed Precision (FP16) ───
    # RTX series punya Tensor Cores yang 2x lebih cepat untuk FP16
    # Training tetap stabil karena loss computation pakai FP32
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"✓ Mixed precision enabled: {policy.name}")
    print("  → Compute dtype  : float16")
    print("  → Variable dtype : float32 (numerical stability)")
else:
    print("⚠ No GPU detected — running on CPU (will be slow)")
    print("  Install CUDA + cuDNN for NVIDIA GPU support")

# ─── OPTIMASI 2: XLA Compilation ───
# Fused ops, reduced memory, faster execution
tf.config.optimizer.set_jit(True)
print("✓ XLA JIT compilation enabled")

# ════════════════════════════════════════════════════════════
# STEP 1 — Konfigurasi
# ════════════════════════════════════════════════════════════
CONFIG = {
    "DATA_DIR"      : "data/plantvillage dataset/color",
    "MODEL_SAVE"    : "models/best_model.keras",
    "LABEL_MAP"     : "models/label_map.json",
    "IMG_SIZE"      : 224,
    # ─── OPTIMASI 3: Batch size lebih besar ───
    # RTX 4060/4070 laptop: 64-128 aman dengan FP16
    # RTX 3050/3060: 48-64
    # Kalau OOM, turunkan ke 48 atau 32
    "BATCH_SIZE"    : 48,  # RTX 4050 6GB VRAM: 48 aman, 64 bisa OOM. Coba naikin ke 64 kalau gak error.
    "EPOCHS"        : 30,
    "LEARNING_RATE" : 1e-3,
    "FINE_TUNE_LR"  : 1e-5,
    "FINE_TUNE_AT"  : 100,
    "SEED"          : 42,
    "VAL_SPLIT"     : 0.2,
    # ─── OPTIMASI 4: tf.data tuning ───
    "AUTOTUNE"      : tf.data.AUTOTUNE,
    "NUM_PARALLEL"  : tf.data.AUTOTUNE,  # parallel map calls
}

# Reproducibility
random.seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])
tf.random.set_seed(CONFIG["SEED"])

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ════════════════════════════════════════════════════════════
# STEP 2 — Data Loading dengan tf.data (bukan ImageDataGenerator)
# ════════════════════════════════════════════════════════════
IMG_SIZE    = CONFIG["IMG_SIZE"]
BATCH_SIZE  = CONFIG["BATCH_SIZE"]

# Scan semua file gambar & label
data_dir = Path(CONFIG["DATA_DIR"])

# ─── VERIFICATION: pastikan dataset ada & struktur benar ───
if not data_dir.exists():
    print(f"\n ERROR: Dataset tidak ditemukan di: {data_dir.absolute()}")
    print(" Download dari: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    print(" Extract ke folder: data/plantvillage dataset/color/")
    exit(1)

class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
NUM_CLASSES = len(class_names)

print(f"\nDataset ditemukan di: {data_dir.absolute()}")
print(f"Jumlah kelas: {NUM_CLASSES}")
print(f"Kelas pertama: {class_names[:3]}")

# Hitung total gambar per kelas
for cls in class_names[:3]:
    n = len(list((data_dir / cls).glob("*")))
    print(f"  {cls}: {n} gambar")

# Kumpulkan semua path + label
all_image_paths = []
all_labels = []
for class_name in class_names:
    class_dir = data_dir / class_name
    for img_path in class_dir.glob("*"):
        if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            all_image_paths.append(str(img_path))
            all_labels.append(class_to_idx[class_name])

all_image_paths = np.array(all_image_paths)
all_labels = np.array(all_labels)
total_samples = len(all_image_paths)
print(f"Total gambar: {total_samples}")

# Shuffle & split
indices = np.random.permutation(total_samples)
split_idx = int(total_samples * (1 - CONFIG["VAL_SPLIT"]))
train_idx, val_idx = indices[:split_idx], indices[split_idx:]

train_paths = all_image_paths[train_idx]
train_labels = all_labels[train_idx]
val_paths = all_image_paths[val_idx]
val_labels = all_labels[val_idx]

print(f"Training  : {len(train_paths)}")
print(f"Validasi  : {len(val_paths)}")

# Simpan label map
with open(CONFIG["LABEL_MAP"], "w") as f:
    json.dump(class_to_idx, f, indent=2)
print(f"Label map disimpan ke {CONFIG['LABEL_MAP']}")


# ─── tf.data Pipeline ───

def decode_and_preprocess(path, label, augment=False):
    """Load image, resize. Optional augmentation."""
    # Read & decode — gunakan decode_jpeg (shape known, lebih cepat)
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)  # ← FIXED: static shape
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)  # Keep in [0, 255] range for EfficientNet built-in rescaling

    if augment:
        # Augmentasi LEMBUT — jangan terlalu agresif biar pretrained features tetap relevan
        img = tf.image.random_flip_left_right(img)
        # Hapus: random_flip_up_down (daun biasanya orientasi normal)
        # Hapus: random_brightness/contrast/saturation terlalu kuat
        # Cukup sedikit variasi warna (skala [0, 255] -> max_delta=25.5):
        img = tf.image.random_brightness(img, 25.5)
        img = tf.image.random_contrast(img, 0.9, 1.1)

    # Clip ke [0, 255] setelah augmentasi
    img = tf.clip_by_value(img, 0.0, 255.0)

    # One-hot encode
    label = tf.one_hot(label, NUM_CLASSES)
    return img, label


def make_dataset(paths, labels, augment=False, shuffle=False):
    """Create optimized tf.data.Dataset."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 5000), seed=CONFIG["SEED"])

    # ─── OPTIMASI 5: Parallel map + prefetch ───
    ds = ds.map(
        lambda p, l: decode_and_preprocess(p, l, augment=augment),
        num_parallel_calls=CONFIG["NUM_PARALLEL"],
        deterministic=False,  # lebih cepat, order gak penting
    )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    # Prefetch: overlap data loading dengan training compute
    ds = ds.prefetch(CONFIG["AUTOTUNE"])

    return ds


train_ds = make_dataset(train_paths, train_labels, augment=True, shuffle=True)
val_ds = make_dataset(val_paths, val_labels, augment=False, shuffle=False)

print(f"\ntf.data pipeline siap")
print(f"  Batch size  : {BATCH_SIZE}")
print(f"  Train steps : {len(train_paths) // BATCH_SIZE + 1}")
print(f"  Val steps   : {len(val_paths) // BATCH_SIZE + 1}")

# ─── VERIFICATION: pastikan data pipeline benar ───
print("\n" + "="*50)
print("VERIFICATION — cek data pipeline")
print("="*50)
for images, labels in train_ds.take(1):
    print(f"  Image batch shape : {images.shape}")
    print(f"  Label batch shape : {labels.shape}")
    print(f"  Image dtype       : {images.dtype}")
    print(f"  Image range       : [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")
    print(f"  Label sample (5)  : {tf.argmax(labels[:5], axis=1).numpy()}")
    # Sanity check: labels shouldn't be all the same
    unique_labels = len(set(tf.argmax(labels, axis=1).numpy()))
    print(f"  Unique labels in batch: {unique_labels}")
    if unique_labels < 3:
        print("  ⚠ WARNING: labels may be shuffled incorrectly!")
    else:
        print("  ✓ Labels look OK")
    break

# ════════════════════════════════════════════════════════════
# STEP 3 — Visualisasi Sample Data
# ════════════════════════════════════════════════════════════
def visualize_samples(dataset, class_names, n=9):
    images, labels = next(iter(dataset))
    idx_to_class = {i: name for i, name in enumerate(class_names)}

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # Cast ke uint8 agar matplotlib dapat memvisualisasikan data [0, 255] dengan benar
            ax.imshow(tf.cast(images[i], tf.uint8).numpy())
            label_idx = np.argmax(labels[i].numpy())
            class_name = idx_to_class[label_idx].replace("___", "\n").replace("_", " ")
            ax.set_title(class_name, fontsize=9, pad=4)
        ax.axis("off")
    plt.suptitle("Sample Gambar Training", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("outputs/sample_images.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Sampel gambar disimpan ke outputs/sample_images.png")

visualize_samples(train_ds, class_names)

# ════════════════════════════════════════════════════════════
# STEP 4 — Membangun Model EfficientNet-B0
# ════════════════════════════════════════════════════════════

def build_model(num_classes: int, img_size: int = 224) -> keras.Model:
    """
    Transfer Learning dengan EfficientNet-B0.
    Output layer pakai float32 untuk numerical stability
    saat mixed precision training.
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), name="input_image")

    base_model = EfficientNetB0(
        include_top    = False,
        input_tensor   = inputs,
        weights        = "imagenet",
        pooling        = None,
    )
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    x = layers.Dense(512, activation="relu", name="dense_512")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)

    # Output layer: float32 untuk mixed precision stability
    outputs = layers.Dense(
        num_classes, activation="softmax", name="predictions",
        dtype="float32"  # ← PENTING untuk mixed precision!
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="PlantDisease_EfficientNetB0")
    return model, base_model


model, base_model = build_model(NUM_CLASSES, IMG_SIZE)
model.summary()

total_params = model.count_params()
trainable_params = sum(tf.size(v).numpy() for v in model.trainable_variables)
print(f"\nTotal parameter      : {total_params:,}")
print(f"Parameter trainable  : {trainable_params:,}")

# ════════════════════════════════════════════════════════════
# STEP 5 — FASE 1: Training Head Classifier
# ════════════════════════════════════════════════════════════
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

# Timing
t_start = time.time()

history_phase1 = model.fit(
    train_ds,
    epochs              = 15,
    validation_data     = val_ds,
    callbacks           = callbacks_phase1,
    verbose             = 1,
)

t_phase1 = time.time() - t_start
print(f"\n⏱ Fase 1 selesai dalam {t_phase1/60:.1f} menit")
print(f"  Rata-rata per epoch: {t_phase1/max(len(history_phase1.history['accuracy']),1):.1f} detik")

# ════════════════════════════════════════════════════════════
# STEP 6 — FASE 2: Fine-tuning
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FASE 2 — Fine-tuning (unfreeze top layers)")
print("="*60)

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

t_start = time.time()

history_phase2 = model.fit(
    train_ds,
    epochs              = CONFIG["EPOCHS"],
    initial_epoch       = len(history_phase1.history["accuracy"]),
    validation_data     = val_ds,
    callbacks           = callbacks_phase2,
    verbose             = 1,
)

t_phase2 = time.time() - t_start
total_time = t_phase1 + t_phase2
print(f"\n⏱ Fase 2 selesai dalam {t_phase2/60:.1f} menit")
print(f"⏱ Total training: {total_time/60:.1f} menit")

# ════════════════════════════════════════════════════════════
# STEP 7 — Visualisasi Kurva Training
# ════════════════════════════════════════════════════════════
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

# ════════════════════════════════════════════════════════════
# STEP 8 — Evaluasi & Confusion Matrix
# ════════════════════════════════════════════════════════════
print("\nMemuat model terbaik...")
best_model = keras.models.load_model(CONFIG["MODEL_SAVE"])

print("Evaluasi pada validation set...")
y_pred_prob = best_model.predict(val_ds, verbose=1)
y_pred      = np.argmax(y_pred_prob, axis=1)
y_true      = val_labels[:len(y_pred)]  # match length

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\n" + report)
with open("outputs/classification_report.txt", "w") as f:
    f.write(report)

cm_matrix = confusion_matrix(y_true, y_pred)
cm_norm   = cm_matrix.astype("float") / cm_matrix.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(20, 18))
sns.heatmap(
    cm_norm, annot=True, fmt=".2f", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names,
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
print(f"Total training time: {total_time/60:.1f} minutes")
