# Panduan Penjelasan Baris Kode Pelatihan Model (`01_train_efficientnet.py`)
## *(Bahan Belajar & Pendalaman Persiapan Sidang Skripsi)*

Dokumen ini membedah seluruh baris kode program pelatihan model yang ada pada [notebooks/01_train_efficientnet.py](../notebooks/01_train_efficientnet.py). Setiap blok kode, parameter, hingga **angka-angka terkecil (magic numbers)** dijelaskan tujuan fungsinya secara mendalam, kegunaannya, serta cara mengungkapkannya secara akademis di depan dosen penguji.

---

## 1. Blok Impor Library (Baris 3-19)

```python
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from sklearn.metrics import classification_report, confusion_matrix
```

### A. Penjelasan Parameter & Pustaka:
* **`os` & `Path`**: Pustaka untuk mengelola folder dan file pada sistem operasi (Linux).
* **`random` & `numpy` (sebagai `np`)**: Mengelola keacakan dan manipulasi array matriks gambar multi-dimensi.
* **`time`**: Menghitung lama waktu komputasi training.
* **`matplotlib.pyplot` (sebagai `plt`) & `seaborn` (sebagai `sns`)**: Membuat grafik kurva pelatihan dan heatmap confusion matrix.
* **`cv2` (OpenCV)**: Pustaka pemrosesan citra komputer untuk resizing dan manipulasi gambar.
* **`json`**: Membaca dan menulis pemetaan label kelas ke format teks terstruktur JSON.
* **`tensorflow` & `keras`**: Framework utama pembuat saraf tiruan (*neural network*).
* **`EfficientNetB0`**: Arsitektur dasar CNN (*Convolutional Neural Network*) pre-trained yang dikembangkan oleh Google.
* **`ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`**: Callbacks pengontrol training otomatis.
* **`classification_report` & `confusion_matrix`**: Metrik evaluasi statistik dari Scikit-Learn.

### B. Cara Menjelaskannya Saat Sidang:
> *"Pada bagian awal kode, kami mengimpor beberapa pustaka utama. Kami menggunakan **TensorFlow** dan **Keras** sebagai core framework untuk membangun dan melatih model deep learning. Untuk penanganan data citra secara efisien, kami memanfaatkan **OpenCV** dan **Pathlib**. Di bagian evaluasi akhir, kami menggunakan pustaka **Scikit-Learn** untuk memanggil fungsi pembuat Classification Report dan Confusion Matrix secara otomatis."*

---

## 2. Setup GPU & Optimasi Akselerasi (Baris 21-44)

```python
gpus = tf.config.list_physical_devices("GPU")
print(f"Versi TensorFlow: {tf.__version__}")
print(f"GPU Terdeteksi  : {len(gpus)} GPU(s)")
for gpu in gpus:
    print(f"  → {gpu}")

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Memory growth diaktifkan")

    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"Mixed precision diaktifkan: {policy.name}")
else:
    print("GPU tidak ditemukan - berjalan menggunakan CPU")

tf.config.optimizer.set_jit(True)
print("Kompilasi XLA diaktifkan")
```

### A. Penjelasan Parameter & Angka:
* **`list_physical_devices("GPU")`**: Memindai hardware kartu grafis NVIDIA yang terpasang pada komputer/laptop.
* **`set_memory_growth(gpu, True)`**: Mengatur VRAM GPU agar teralokasi secara bertahap sesuai kebutuhan model. Defaultnya, TensorFlow akan langsung mengunci 100% kapasitas VRAM yang memicu crash jika ada aplikasi lain berjalan.
* **`Policy("mixed_float16")`**: Kebijakan pencampuran tipe data numerik. Perhitungan perkalian matriks (*forward pass*) dilakukan menggunakan tipe **float16** (16-bit) yang sangat cepat di Tensor Core GPU, namun bobot utama (*master weights*) tetap disimpan menggunakan **float32** (32-bit) agar aman dari masalah kehilangan data akibat terlalu kecilnya nilai (*underflow*).
* **`set_jit(True)`**: Mengaktifkan XLA (*Accelerated Linear Algebra*) untuk menyatukan operasi matematika kecil menjadi satu instruksi besar agar transfer data di memori GPU berjalan efisien.

### B. Cara Menjelaskannya Saat Sidang:
> *"Kami membangun sistem ini dengan optimasi hardware tingkat lanjut. Pertama, kami mengaktifkan **Memory Growth** agar TensorFlow hanya mengambil VRAM GPU sesuai kebutuhan saat training secara dinamis, sehingga mencegah terjadinya crash OOM. Kedua, kami menerapkan **Mixed Precision (float16)** untuk mempercepat operasi matriks pada Tensor Core GPU tanpa mengurangi akurasi model. Terakhir, kami mengaktifkan compiler **XLA JIT (Just-In-Time)** untuk melakukan optimasi fusi operasi matematika langsung pada tingkat instruksi GPU."*

---

## 3. Kamus Konfigurasi & Pengunci Seed (Baris 46-70)

```python
CONFIG = {
    "DATA_DIR"      : "data/plantvillage dataset/color",
    "MODEL_SAVE"    : "models/best_model.keras",
    "LABEL_MAP"     : "models/label_map.json",
    "IMG_SIZE"      : 224,
    "BATCH_SIZE"    : 48,
    "EPOCHS"        : 30,
    "LEARNING_RATE" : 1e-3,
    "FINE_TUNE_LR"  : 1e-5,
    "FINE_TUNE_AT"  : 100,
    "SEED"          : 42,
    "VAL_SPLIT"     : 0.2,
    "AUTOTUNE"      : tf.data.AUTOTUNE,
    "NUM_PARALLEL"  : tf.data.AUTOTUNE,
}

random.seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])
tf.random.set_seed(CONFIG["SEED"])
```

### A. Penjelasan Parameter & Angka Konfigurasi:
* **`"DATA_DIR"`**: Path folder dataset gambar penyakit daun.
* **`"MODEL_SAVE"`**: File output `.keras` untuk menyimpan model terbaik. Format `.keras` adalah format standar penyimpanan model Keras terbaru yang menyimpan arsitektur, bobot, dan konfigurasi optimizer sekaligus.
* **`"LABEL_MAP"`**: File output pemetaan indeks kelas ke teks nama penyakit.
* **`"IMG_SIZE" : 224`**: Gambar di-resize ke resolusi $224 \times 224$ piksel. Angka **224** adalah resolusi asli saat model EfficientNet-B0 dilatih pada dataset ImageNet oleh Google. Menggunakan angka ini menjamin transfer pengetahuan berjalan optimal.
* **`"BATCH_SIZE" : 48`**: Model membaca 48 gambar sekaligus dalam satu iterasi. Angka **48** dipilih untuk memaksimalkan utilitas VRAM kartu grafis RTX 3060 tanpa memicu OOM (jika diatur ke 64, VRAM berisiko meluap).
* **`"EPOCHS" : 30`**: Batas maksimal iterasi pelatihan untuk Fase 2. Angka **30** dipilih karena model biasanya sudah konvergen penuh (tidak ada lagi penurunan loss) sebelum mencapai epoch ke-30 berkat bantuan callbacks.
* **`"LEARNING_RATE" : 1e-3 (0.001)`**: Kecepatan belajar awal untuk Fase 1. Angka **0.001** adalah nilai default ideal bagi optimizer Adam untuk melatih bobot acak pada classifier head kustom yang baru ditambahkan.
* **`"FINE_TUNE_LR" : 1e-5 (0.00001)`**: Kecepatan belajar Fase 2. Dibuat **100 kali lebih kecil** agar bobot layer backbone yang dibuka tidak bergeser terlalu jauh, menghindari kerusakan fitur ImageNet.
* **`"FINE_TUNE_AT" : 100`**: Angka indeks layer backbone yang mulai dilatih. Layer 0-99 dibekukan, layer 100-237 di-train ulang.
* **`"SEED" : 42`**: Angka pemandu generator keacakan. Nilai **42** adalah angka universal dalam komunitas pemrograman yang digunakan untuk mengunci keacakan.
* **`"VAL_SPLIT" : 0.2`**: Rasio pembagian data validasi sebesar 20%, menyisakan 80% data untuk training.
* **`AUTOTUNE`**: Konstanta internal TensorFlow untuk menentukan alokasi sumber daya CPU secara dinamis selama training.

### B. Cara Menjelaskannya Saat Sidang:
> *"Seluruh parameter training kami kumpulkan dalam satu struktur data konfigurasi terpusat. Untuk menjamin aspek **reproduksibilitas**—yaitu memastikan hasil akurasi model dapat dibuktikan kembali dengan hasil yang sama persis oleh peneliti lain—kami mengunci *random seed* sistem pada nilai **42** di tingkat Python, NumPy, dan TensorFlow."*

---

## 4. Pemindaian Direktori & Split Dataset (Baris 71-150)

```python
class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
NUM_CLASSES = len(class_names)

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

indices = np.random.permutation(total_samples)
split_idx = int(total_samples * (1 - CONFIG["VAL_SPLIT"]))
train_idx, val_idx = indices[:split_idx], indices[split_idx:]

train_paths = all_image_paths[train_idx]
train_labels = all_labels[train_idx]
val_paths = all_image_paths[val_idx]
val_labels = all_labels[val_idx]
```

### A. Penjelasan Parameter & Fungsi:
* **`sorted()`**: Mengurutkan nama folder kelas penyakit secara alfabetis agar pemetaan indeks label (0, 1, 2, dst.) tidak berubah-ubah di komputer yang berbeda.
* **`glob("*")`**: Memindai seluruh isi file di dalam folder kelas secara rekursif.
* **`suffix.lower()`**: Menyaring ekstensi file gambar agar hanya format gambar populer (.jpg, .jpeg, .png, .bmp) saja yang dimuat, menghindari file sistem sampah seperti `.DS_Store` masuk ke pipeline data.
* **`np.random.permutation()`**: Menghasilkan daftar indeks angka acak sepanjang total sampel gambar. Fungsi ini mengacak posisi data secara keseluruhan agar model tidak mempelajari urutan folder kelas tertentu secara berurutan.

### B. Cara Menjelaskannya Saat Sidang:
> *"Program kami membaca folder dataset secara rekursif untuk mengumpulkan path dari setiap file citra dan memetakan nama kelas penyakit ke format indeks numerik. Setelah itu, kami melakukan pengacakan indeks sampel (*shuffling*) secara acak terkontrol dan membagi data menjadi dua bagian: **80% untuk data latih** dan **20% untuk data validasi**. Kami juga menyimpan relasi label ke dalam file JSON agar bisa digunakan kembali saat tahap deployment web app."*

---

## 5. Fungsi Preprocessing & tf.data Dataset Pipeline (Baris 152-202)

```python
def decode_and_preprocess(path, label, augment=False):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 25.5)
        img = tf.image.random_contrast(img, 0.9, 1.1)

    img = tf.clip_by_value(img, 0.0, 255.0)
    label = tf.one_hot(label, NUM_CLASSES)
    return img, label

def make_dataset(paths, labels, augment=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 5000), seed=CONFIG["SEED"])

    ds = ds.map(
        lambda p, l: decode_and_preprocess(p, l, augment=augment),
        num_parallel_calls=CONFIG["NUM_PARALLEL"],
        deterministic=False,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(CONFIG["AUTOTUNE"])
    return ds
```

### A. Penjelasan Parameter & Angka:
* **`channels=3`**: Mendekode gambar ke format matriks 3 saluran warna (RGB). Informasi warna sangat penting untuk diagnosis penyakit tanaman (misal bercak karat cokelat atau daun menguning). Jika diisi `1`, gambar akan dikonversi ke hitam-putih (Grayscale) yang akan menghilangkan informasi warna esensial.
* **`random_brightness(..., 25.5)`**: Mengubah kecerahan gambar secara acak dengan rentang maksimal perubahan **25.5** (atau setara dengan ±10% dari nilai piksel maksimal 255).
* **`random_contrast(..., 0.9, 1.1)`**: Mengubah tingkat kekontrasan gambar secara acak dengan skala pengali antara **90% (0.9)** hingga **110% (1.1)**.
* **`clip_by_value(img, 0.0, 255.0)`**: Memotong nilai piksel agar tetap berada di antara rentang **0 hingga 255**. Proses penambahan kecerahan dan kontras sebelumnya berpotensi menghasilkan nilai piksel di atas 255 atau di bawah 0 yang dapat menyebabkan eror visual (piksel rusak).
* **`one_hot(label, NUM_CLASSES)`**: Mengubah indeks kelas (misal kelas `2`) menjadi vektor biner beraliran nol dan satu: `[0, 0, 1, 0, 0, ...]`.
* **`buffer_size=5000`**: Menentukan ukuran antrean pengacakan gambar dalam RAM. Alih-alih memuat seluruh dataset ke RAM (yang bisa menyebabkan crash komputer), TensorFlow hanya akan memuat 5000 gambar ke buffer, mengacaknya, lalu mengambil gambar berikutnya secara bergantian.
* **`prefetch` & `AUTOTUNE`**: Menyuruh CPU untuk memuat dan mempersiapkan batch gambar berikutnya selagi GPU sibuk memproses batch saat ini, menghilangkan waktu tunggu *idle* pada GPU.

### B. Cara Menjelaskannya Saat Sidang:
> *"Untuk mengatasi masalah bottleneck performa pemrosesan gambar, kami membangun data pipeline menggunakan **tf.data.Dataset API**. Fungsi preprocessing bertugas mendekode gambar JPEG ke nilai matriks RGB, meresize dimensi citra ke ukuran standar $224 \times 224$, serta menerapkan teknik augmentasi (seperti horizontal flip, random brightness, dan contrast) hanya pada data training. Data label kemudian dikonversi menjadi representasi vektor biner menggunakan **One-Hot Encoding**. Seluruh proses pemuatan gambar ini dijalankan secara paralel melalui fungsi `prefetch` sehingga GPU tidak perlu menunggu CPU selesai membaca gambar."*

---

## 6. Konstruksi Arsitektur Model (Baris 218-245)

```python
def build_model(num_classes: int, img_size: int = 224) -> tuple:
    inputs = keras.Input(shape=(img_size, img_size, 3), name="input_image")
    
    base_model = EfficientNetB0(
        include_top=False,
        input_tensor=inputs,
        weights="imagenet",
        pooling=None,
    )
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    x = layers.Dense(512, activation="relu", name="dense_512")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)

    outputs = layers.Dense(
        num_classes, activation="softmax", name="predictions",
        dtype="float32"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="PlantDisease_EfficientNetB0")
    return model, base_model
```

### A. Penjelasan Parameter & Angka Arsitektur:
* **`include_top=False`**: Memotong lapisan kepala klasifikasi bawaan EfficientNet-B0 (yang bertugas menebak 1000 objek ImageNet) agar kita bisa menaruh kepala klasifikasi kustom baru di atasnya.
* **`weights="imagenet"`**: Memuat bobot parameter yang sudah terlatih pada dataset ImageNet (jutaan gambar objek umum) sebagai modal pengetahuan awal model.
* **`GlobalAveragePooling2D`**: Merata-ratakan nilai spasial fitur gambar dari dimensi $7 \times 7 \times 1280$ menjadi hanya $1280$. Meminimalisasi parameter latih dibanding menggunakan layer Flatten.
* **`Dropout(0.4)` & `Dropout(0.3)`**: Memutus secara acak **40%** dan **30%** koneksi neuron saat training untuk regulasi ganda mencegah overfitting.
* **`Dense(512)`**: Lapisan tersembunyi (*hidden layer*) dengan **512** neuron aktif beraktivasi ReLU untuk mempelajari kombinasi pola non-linear dari fitur daun.
* **`Dense(num_classes)`**: Lapisan keluaran sebanyak **38** neuron (sesuai jumlah kelas tanaman).
* **`activation="softmax"`**: Mengubah skor output numerik mentah menjadi persentase keyakinan probabilitas untuk masing-masing kelas.
* **`dtype="float32"`**: Mengunci presisi output tetap di tipe float 32-bit agar perhitungan probabilitas Softmax tidak mengalami kegagalan numerik akibat kebijakan *mixed precision float16* pada layer-layer sebelumnya.

### B. Cara Menjelaskannya Saat Sidang:
> *"Model kami dibangun menggunakan konsep **Transfer Learning**. Kami mengambil fitur ekstraksi dari arsitektur **EfficientNet-B0** yang telah dilatih pada dataset ImageNet. Karena kami tidak memakai kepala klasifikasi bawaan ImageNet, kami memotong lapisan atasnya (`include_top=False`) dan menggantinya dengan kepala klasifikasi kustom. Kepala kustom ini terdiri dari **Global Average Pooling** untuk merangkum fitur, lapisan **Batch Normalization** untuk stabilisasi nilai, dua buah lapisan **Dropout (0.4 dan 0.3)** untuk menekan risiko overfitting, satu lapisan **Dense 512** dengan fungsi aktivasi **ReLU**, serta diakhiri dengan lapisan **Dense 38** menggunakan aktivasi **Softmax** dengan presisi `float32` demi kestabilan perhitungan probabilitas."*

---

## 7. Fase 1: Melatih Kepala Klasifikasi (Baris 255-275)

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG["LEARNING_RATE"]),
    loss="categorical_crossentropy",
    metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
)

callbacks_phase1 = [
    ModelCheckpoint(CONFIG["MODEL_SAVE"], monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1, min_lr=1e-7),
]

history_phase1 = model.fit(
    train_ds,
    epochs=15,
    validation_data=val_ds,
    callbacks=callbacks_phase1,
    verbose=1,
)
```

### A. Penjelasan Parameter & Angka Training:
* **`loss="categorical_crossentropy"`**: Fungsi kerugian standar untuk kasus klasifikasi multi-kelas di mana target output berupa matriks satu kolom biner (*one-hot encoded*).
* **`TopKCategoricalAccuracy(k=5)`**: Metrik evaluasi tambahan untuk mengukur apakah label asli berada dalam **top-5** tebakan model dengan persentase probabilitas terbesar.
* **`save_best_only=True`**: Menginstruksikan ModelCheckpoint agar hanya menimpa file model jika performa akurasi validasi (`val_accuracy`) saat ini lebih tinggi dari pencapaian terbaik sebelumnya.
* **`patience=5` (EarlyStopping)**: Menghentikan proses latihan otomatis jika dalam **5 epoch** berturut-turut nilai akurasi data validasi tidak mengalami kenaikan sedikit pun.
* **`factor=0.3` & `patience=3` (ReduceLROnPlateau)**: Mengalikan learning rate dengan faktor pengali **0.3** (menurunkannya sebesar 70%) jika dalam **3 epoch** berturut-turut nilai loss data validasi mengalami stagnasi.
* **`epochs=15`**: Latihan dibatasi maksimal selama **15 epoch** saja karena melatih kepala klasifikasi baru biasanya tidak memerlukan waktu lama untuk mencapai konvergensi awal.

### B. Cara Menjelaskannya Saat Sidang:
> *"Pada **Fase 1**, kami mengunci bobot arsitektur dasar (*backbone*) dan hanya melatih bagian kepala klasifikasi yang baru. Kami mengompilasi model menggunakan **Adam Optimizer** dengan learning rate `1e-3`. Kami menyertakan tiga buah callback pengaman utama selama proses latih: **ModelCheckpoint** untuk menyimpan bobot terbaik secara berkala, **EarlyStopping** dengan toleransi 5 epoch untuk menghindari pemborosan waktu training, serta **ReduceLROnPlateau** untuk memotong learning rate jika kurva loss mengalami kebuntuan."*

---

## 8. Fase 2: Fine-Tuning Sebagian Layer (Baris 277-316)

```python
base_model.trainable = True
for layer in base_model.layers[:CONFIG["FINE_TUNE_AT"]]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG["FINE_TUNE_LR"]),
    loss="categorical_crossentropy",
    metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
)

callbacks_phase2 = [
    ModelCheckpoint(CONFIG["MODEL_SAVE"], monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=4, verbose=1, min_lr=1e-8),
]

history_phase2 = model.fit(
    train_ds,
    epochs=CONFIG["EPOCHS"],
    initial_epoch=len(history_phase1.history["accuracy"]),
    validation_data=val_ds,
    callbacks=callbacks_phase2,
    verbose=1,
)
```

### A. Penjelasan Parameter & Angka Fine-Tuning:
* **`base_model.trainable = True`**: Mengaktifkan kembali status latihan pada seluruh layer backbone EfficientNet-B0 agar bobot parameternya bisa disesuaikan kembali.
* **`layers[:100]`**: Mengunci layer backbone indeks 0 sampai indeks ke-99. Layer-layer bawah ini bertugas mengekstrak fitur visual mendasar yang sudah sangat kuat dipelajari dari ImageNet (seperti garis tepi dan kontras gelap-terang). Kita hanya melatih layer 100 ke atas yang mendeteksi visual lebih rumit (seperti warna karat daun dan bentuk bercak).
* **`patience=8` (EarlyStopping Fase 2)**: Batas toleransi dihentikannya training ditingkatkan menjadi **8 epoch** karena penyesuaian bobot pada fine-tuning berjalan lambat dan membutuhkan waktu observasi kurva validasi lebih lama.
* **`patience=4` (ReduceLROnPlateau Fase 2)**: Menurunkan learning rate jika loss validasi tidak membaik selama **4 epoch**.
* **`initial_epoch`**: Melanjutkan pencatatan grafik epoch dari titik akhir Fase 1 agar kurva pelatihan terhubung secara linier dan berkesinambungan.

### B. Cara Menjelaskannya Saat Sidang:
> *"Setelah kepala klasifikasi stabil pada Fase 1, kami beralih ke **Fase 2 (Fine-tuning)**. Di sini, kami membuka seluruh layer backbone, tetapi mengunci kembali layer indeks 0 sampai 99 karena layer bawah bertugas mengenali bentuk garis dan warna umum yang sudah matang dari ImageNet. Kami melatih layer 100 ke atas dengan **learning rate yang jauh lebih kecil (1e-5)** agar penyesuaian bobot berjalan secara halus dan tidak merusak fitur bawaan yang berguna. Proses ini dilatih maksimal selama 30 epoch."*

---

## 9. Evaluasi & Visualisasi Heatmap Akhir (Baris 318-384)

```python
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\n" + report)

cm_matrix = confusion_matrix(y_true, y_pred)
cm_norm = cm_matrix.astype("float") / cm_matrix.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(20, 18))
sns.heatmap(
    cm_norm, annot=True, fmt=".2f", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names,
    linewidths=0.3, ax=ax
)
```

### A. Penjelasan Parameter Visualisasi:
* **`digits=4`**: Menginstruksikan `classification_report` agar menampilkan hasil presisi, recall, dan F1-score dengan ketelitian **4 digit di belakang koma** (misal `0.9944`) agar kita bisa mendeteksi perbedaan performa minor pada kelas-kelas yang bersaing ketat.
* **`cm_matrix.sum(axis=1, keepdims=True)`**: Membagi setiap angka di dalam sel confusion matrix dengan total jumlah gambar pada kelas baris terkait. Ini menghasilkan **Confusion Matrix Ternormalisasi** (skala 0.00 hingga 1.00), yang jauh lebih objektif dibaca dibandingkan angka absolut jika jumlah gambar tiap kelas tidak seimbang.
* **`figsize=(20, 18)`**: Mengatur ukuran kanvas gambar confusion matrix selebar **20 inci** dan setinggi **18 inci**. Resolusi besar ini wajib digunakan agar nama-nama kelas yang panjang (38 kelas) serta teks persentase di dalam sel-sel kecil tidak menumpuk dan tetap terbaca dengan jelas.
* **`cmap="Blues"`**: Memilih palet warna biru untuk heatmap. Sel dengan akurasi 100% akan berwarna biru pekat, sedangkan sel yang bernilai 0% akan berwarna putih bersih.
* **`linewidths=0.3`**: Memberikan garis putih pembatas setebal **0.3** antar sel matriks agar tampilan heatmap terlihat rapi dan tidak melebur menjadi satu warna.

### B. Cara Menjelaskannya Saat Sidang:
> *"Tahap terakhir adalah evaluasi model. Kami memuat file bobot model terbaik yang disimpan saat training, lalu memprediksi seluruh data validasi. Dari hasil tebakan tersebut, kami membuat **Classification Report** untuk menampilkan detail F1-score tiap kelas dengan ketelitian 4 digit desimal. Terakhir, kami menghitung **Confusion Matrix** yang kemudian dinormalisasi untuk memvisualisasikan persebaran kesalahan prediksi model ke dalam bentuk grafik heatmap gradasi warna biru."*
