# Penjelasan Konsep, Istilah, & Makna Angka pada Model (Bahasa Sederhana)
## *(Panduan Belajar Lengkap & Persiapan Ujian Sidang)*

Dokumen ini menjelaskan istilah-istilah teknis arsitektur model deep learning yang digunakan dalam proyek ini dengan bahasa yang mudah dipahami, lengkap dengan analogi sederhana, arti angka konfigurasi, dan daftar pertanyaan kritis (SAQ) untuk ujian sidang.

---

## 1. Penjelasan Istilah & Konsep Dasar

Berikut adalah penjelasan fungsi setiap istilah teknis yang sering muncul dalam proyek ini:

### A. Backbone (EfficientNet-B0)
* **Maksudnya:** Bagian utama atau "pondasi" model yang bertugas mengekstrak ciri-ciri visual dari gambar daun (seperti warna daun, bentuk tepi daun, pola bercak, garis-garis serat, dll.).
* **Mengapa menggunakan pre-trained?** Kami menggunakan arsitektur EfficientNet-B0 yang sudah dilatih sebelumnya pada dataset raksasa (ImageNet). Keuntungannya, model ini sudah pintar dalam melihat objek, garis, dan bentuk dasar gambar sejak awal. Kita tidak perlu melatih model dari nol, melainkan cukup menyesuaikannya agar fokus mengenali ciri penyakit daun tanaman saja (metode ini disebut *Transfer Learning*).

### B. Global Average Pooling (GAP)
* **Maksudnya:** Proses penyusutan atau merangkum informasi gambar dari bentuk peta dua dimensi ($7 \times 7 \times 1280$) menjadi satu baris angka ($1280$). GAP bekerja dengan cara mengambil nilai rata-rata dari setiap area fitur gambar.
* **Mengapa penting?** Jika kita menggunakan metode lama seperti `Flatten` (mengubah data gambar menjadi satu baris panjang tanpa dirangkum), ukuran data akan membengkak drastis. Hal ini membuat model menjadi sangat lambat dan rentan mengalami *overfitting* (hanya menghafal data latihan). GAP menyederhanakan data dengan cerdas tanpa kehilangan informasi penting.

### C. Batch Normalization (BN)
* **Maksudnya:** Proses standarisasi atau penyeragaman nilai data hasil ekstraksi gambar.
* **Mengapa penting?** Saat gambar diproses melalui berbagai layer, nilai angka di dalamnya bisa berubah drastis (ada yang menjadi sangat besar, ada yang sangat kecil). BN bertugas menyeimbangkan kembali angka-angka tersebut agar berada di rentang yang stabil. Hal ini membuat proses pelatihan model berjalan jauh lebih cepat dan tidak mudah eror secara perhitungan.

### D. Dropout
* **Maksudnya:** Teknik mematikan koneksi beberapa neuron (unit pemroses) secara acak selama latihan (misalnya dinonaktifkan sebesar 30% atau 40%).
* **Mengapa penting?** Tujuannya agar model tidak malas dan tidak hanya mengandalkan beberapa neuron tertentu saja untuk mengenali penyakit. Dengan mematikan neuron secara acak, model dipaksa untuk menggunakan jalur alternatif lain. Hasilnya, model menjadi lebih cerdas dan tidak mudah mengalami *overfitting* (bisa mengenali gambar baru di luar data latihan dengan baik).

### E. Dense Layer (Fully Connected Layer)
* **Maksudnya:** Lapisan saraf pembuat keputusan akhir. Di bagian ini, semua ciri-ciri gambar yang telah dirangkum sebelumnya digabungkan dan dihubungkan satu sama lain untuk menarik kesimpulan.
* **Mengapa penting?** Lapisan inilah yang menghubungkan ciri visual daun (misalnya: warna kuning + ada bercak cokelat membulat) dengan nama penyakit tanaman yang sesuai.

### F. Softmax (Output Layer)
* **Maksudnya:** Fungsi matematika di bagian paling akhir model yang mengubah angka mentah hasil tebakan model menjadi nilai persentase (probabilitas).
* **Mengapa penting?** Softmax memastikan hasil prediksi akhir berupa persentase keyakinan model untuk masing-masing penyakit (contoh: 95% terkena penyakit A, 3% terkena penyakit B, 2% tanaman sehat) dengan total penjumlahan semua persentase pas 100%.

---

## 2. Perumpamaan & Analogi Sederhana

Untuk membantu visualisasi presentasi, berikut beberapa perumpamaan sederhana mengenai cara kerja sistem:

* **Grad-CAM (Kaca Pembesar yang Meninggalkan Jejak):** 
  Bayangkan Grad-CAM seperti sebuah kaca pembesar detektif yang meninggalkan jejak tinta warna merah di atas foto daun. Daerah yang berwarna merah tebal menunjukkan di bagian mana mata detektif (model) tersebut paling lama mengamati gambar sebelum ia menyimpulkan jenis penyakitnya. Jika warna merah menyala tepat di bercak penyakit, model kita terbukti bekerja dengan benar.
* **Augmentasi Data (Latihan Menembak di Berbagai Medan):** 
  Bayangkan Anda sedang melatih penembak jitu. Jika ia hanya dilatih menembak pada siang hari yang cerah dengan target tegak lurus, ia akan gagal saat bertempur di malam hari yang berangin dengan target bergerak. Dengan *Data Augmentation*, kita memutar gambar daun sedikit (flip) dan mengubah tingkat cahayanya (brightness/contrast) agar model tangguh mengenali penyakit daun di segala kondisi luar ruangan yang tidak ideal.

---

## 3. Arti Angka & Parameter Konfigurasi (Mengapa Nilai Ini Dipilih?)

Berikut penjelasan ilmiah mengapa angka-angka tertentu dipilih dalam program:

* **`IMG_SIZE = 224`**: Ukuran resolusi standar input **EfficientNet-B0** agar sesuai dengan fitur pre-trained ImageNet. Ukuran ini menghemat memori GPU namun tetap mempertahankan detail bercak penyakit daun.
* **`BATCH_SIZE = 48`**: Jumlah gambar yang diproses sekaligus dalam satu iterasi. Angka 48 disesuaikan agar performa GPU RTX 3060 berjalan maksimal tanpa memicu eror kehabisan memori (*Out of Memory*).
* **`LEARNING_RATE = 1e-3 (0.001)`**: Kecepatan belajar awal untuk melatih bagian klasifikasi akhir (*classifier head*) yang masih kosong. Dipilih karena merupakan nilai standar yang cepat dan stabil bagi optimizer Adam.
* **`FINE_TUNE_LR = 1e-5 (0.00001)`**: Kecepatan belajar saat melakukan fine-tuning pada layer backbone. Dibuat **100x lebih lambat** agar model tidak merusak bobot representasi visual ImageNet yang sudah matang (*catastrophic forgetting*).
* **`FINE_TUNE_AT = 100`**: Membekukan layer 1-99 (mendeteksi bentuk garis dan warna umum) dan melatih ulang layer 100 ke atas (mendeteksi tekstur penyakit spesifik tanaman) agar pelatihan lebih cepat.
* **`VAL_SPLIT = 0.2 (20%)`**: Pembagian dataset di mana 80% digunakan untuk melatih model, dan 20% disimpan khusus sebagai instrumen evaluasi berkala untuk memantau kemajuan akurasi.
* **`Dense(512)`**: Lapisan transisi dengan 512 neuron untuk menyusutkan dimensi secara bertahap dari 1280 (fitur backbone) menuju 38 (kelas output).
* **`Dropout(0.4) & Dropout(0.3)`**: Tingkat pemutusan koneksi saraf (40% setelah GAP dan 30% setelah Dense 512) sebagai pengaman ganda agar model tidak menghafal gambar latihan (*overfitting*).
* **`Dense(38)`**: Menyesuaikan output secara mutlak dengan 38 jumlah variasi penyakit dan daun sehat pada dataset PlantVillage.
* **`cv2.addWeighted (0.55 & 0.45)`**: Pengaturan opasitas visualisasi Grad-CAM (55% gambar asli, 45% heatmap warna) agar detail fisik daun dan fokus area penyakit terlihat seimbang saat ditampilkan di web app.

---

## 4. Tanya Jawab Kritis & Singkat (SAQ) untuk Sidang Skripsi

### Q1: Mengapa Anda memilih EfficientNet-B0 sebagai arsitektur backbone dibandingkan ResNet-50 atau VGG-16?
* **Jawaban:** 
  "EfficientNet-B0 dipilih karena menerapkan metode **Compound Scaling** (penskalaan seimbang pada dimensi kedalaman, lebar jaringan, dan resolusi gambar). Arsitektur ini memiliki parameter yang jauh lebih sedikit (~5,3 juta parameter) dibanding ResNet-50 (~25 juta parameter), namun mampu menghasilkan akurasi yang setara atau bahkan lebih tinggi. Ini membuatnya sangat efisien untuk di-deploy pada server lokal atau aplikasi web."

### Q2: Mengapa Anda membagi proses training menjadi dua fase (Fase 1: Freeze, Fase 2: Fine-Tuning)? Apa yang terjadi jika langsung di-train semua dari awal?
* **Jawaban:** 
  "Fase 1 digunakan untuk melatih *classifier head* baru yang bobotnya masih acak sementara bobot *backbone* dibekukan (*frozen*) agar fitur universal dari ImageNet tidak rusak oleh gradien loss yang tidak stabil di awal training.
  Setelah *classifier head* stabil, di Fase 2 kita melakukan *fine-tuning* pada layer atas backbone dengan **Learning Rate yang sangat kecil (1e-5)**. Jika langsung dilatih dari awal tanpa pembekuan, model akan mengalami *catastrophic forgetting* di mana fitur penting bawaan ImageNet akan rusak."

### Q3: Akurasi test set Anda mencapai 99.78%. Apakah ini tidak mengindikasikan adanya overfitting? Bagaimana Anda membuktikannya?
* **Jawaban:**
  "Akurasi tinggi ini valid untuk kondisi laboratorium dataset PlantVillage yang seragam. Kami membuktikan model tidak sekadar menghafal latar belakang melalui dua hal:
  1. **Grad-CAM:** Membuktikan visualisasi area merah (fokus utama model) tepat berada pada objek bercak penyakit di daun, bukan pada latar belakang.
  2. **Evaluasi Robustness:** Menguji model dengan gangguan buatan (*Gaussian Noise* dan variasi *Brightness*). Model terbukti tetap kokoh mempertahankan akurasi di atas 88-99% meskipun diberi gangguan visual."

### Q4: Mengapa Anda menggunakan Global Average Pooling (GAP) daripada Flattening sebelum Dense Layer?
* **Jawaban:**
  "Penggunaan `Flatten` akan mengubah matriks 2D menjadi vektor 1D dengan ukuran sangat besar ($62.720$ dimensi), yang memicu ledakan parameter latih dan rawan overfitting. Sementara `GAP` merata-ratakan nilai spasial menjadi hanya $1280$ dimensi. GAP menghemat parameter, memiliki sifat *translation invariance*, dan menjaga keselarasan spasial untuk Grad-CAM."

### Q5: Bagaimana program Anda mendeteksi bahwa model mengalami Overfitting saat training?
* **Jawaban:**
  "Kami memantau kurva **Loss** dan **Accuracy** data latihan dibanding data validasi. Jika nilai loss latihan terus turun mendekati nol tetapi loss validasi malah naik kembali atau akurasinya stagnan, itu indikasi overfitting. Jika terjadi, Callback **EarlyStopping** akan langsung menghentikan proses training secara otomatis untuk mengambil bobot terbaik."
