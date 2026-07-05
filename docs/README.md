# Dokumentasi Proyek Deteksi Penyakit Daun Tanaman

Dokumen berikut merangkum arsitektur, alur kerja, evaluasi, dan deployment sistem.

## Daftar Dokumen

- [Arsitektur Sistem](ARCHITECTURE.md) — penjelasan lengkap tentang model, data pipeline, training, inference, dan komponen UI.
- [Deployment & Evaluasi](DEPLOYMENT_AND_EVALUATION.md) — panduan deployment, benchmark, GPU, dan metrik evaluasi.

## Ringkasan Singkat

Proyek ini menggunakan transfer learning dengan EfficientNet-B0 untuk mengklasifikasikan penyakit daun tanaman dari dataset PlantVillage. Arsitektur sistem mencakup:

- pipeline preprocessing data dengan tf.data
- model classifier berbasis EfficientNet-B0
- explainability dengan Grad-CAM
- antarmuka web berbasis Flask
- evaluasi dan benchmarking otomatis
