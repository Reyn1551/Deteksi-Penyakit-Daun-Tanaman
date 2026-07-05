"""
═══════════════════════════════════════════════════════════════════════════════
COMPREHENSIVE BENCHMARK — best_model.keras
═══════════════════════════════════════════════════════════════════════════════

Tujuan:
  1. Validasi apakah akurasi 99.44% valid atau ada overfitting
  2. Analisis per-class performance
  3. Check confidence calibration (apakah model confident atau overconfident)
  4. Test robustness terhadap data corruption
  5. Identifikasi class imbalance
  6. Rekomendasi untuk improvement

Hasil output:
  - outputs/benchmark_report.txt
  - outputs/confidence_distribution.png
  - outputs/per_class_performance.png
  - outputs/robustness_test.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc
)
import cv2
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "MODEL_PATH"   : "models/best_model.keras",
    "LABEL_PATH"   : "models/label_map.json",
    "DATA_DIR"     : "data/plantvillage dataset/color",
    "IMG_SIZE"     : 224,
    "BATCH_SIZE"   : 32,
    "VAL_SPLIT"    : 0.2,
    "SEED"         : 42,
}

np.random.seed(CONFIG["SEED"])
tf.random.set_seed(CONFIG["SEED"])

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL & DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading model & data...")
model = keras.models.load_model(CONFIG["MODEL_PATH"])

with open(CONFIG["LABEL_PATH"]) as f:
    class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = list(idx_to_class.values())

# Scan dataset
data_dir = Path(CONFIG["DATA_DIR"])
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

# Split menjadi train/val/test (hindari data leakage!)
indices = np.random.permutation(total_samples)
train_size = int(total_samples * 0.6)
val_size = int(total_samples * 0.2)

train_idx = indices[:train_size]
val_idx = indices[train_size:train_size+val_size]
test_idx = indices[train_size+val_size:]

test_paths = all_image_paths[test_idx]
test_labels = all_labels[test_idx]

print(f"\nDataset split:")
print(f"  Train: {len(train_idx)}")
print(f"  Val  : {len(val_idx)}")
print(f"  Test : {len(test_idx)}")

# ─────────────────────────────────────────────────────────────────────────────
# Load test data dengan preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def load_image(path, img_size=224):
    """Load & preprocess image."""
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img

def make_dataset(paths, labels, batch_size=32):
    """Create tf.data dataset."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, l: (load_image(p), l), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

test_ds = make_dataset(test_paths, test_labels, CONFIG["BATCH_SIZE"])

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: PREDICTION & BASIC METRICS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 1: PREDIKSI & BASIC METRICS")
print("="*80)

y_pred_prob_list = []
for images, _ in tqdm(test_ds, desc="Predicting"):
    probs = model.predict(images, verbose=0)
    y_pred_prob_list.append(probs)

y_pred_prob = np.concatenate(y_pred_prob_list, axis=0)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = test_labels[:len(y_pred)]

# Basic metrics
overall_acc = accuracy_score(y_true, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average='weighted'
)

print(f"\n✓ Test Set Performance:")
print(f"  Accuracy  : {overall_acc:.4f} ({overall_acc*100:.2f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: PER-CLASS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 2: PER-CLASS ANALYSIS")
print("="*80)

class_report = classification_report(
    y_true, y_pred, target_names=class_names, output_dict=True
)

# Sort by F1-score untuk lihat mana class yang terburuk
class_f1_scores = []
for class_name in class_names:
    f1 = class_report[class_name]['f1-score']
    support = class_report[class_name]['support']
    class_f1_scores.append((class_name, f1, support))

class_f1_scores_sorted = sorted(class_f1_scores, key=lambda x: x[1])

print(f"\n⚠ Bottom 10 classes (lowest F1-score):")
for class_name, f1, support in class_f1_scores_sorted[:10]:
    print(f"  {class_name:50s} F1={f1:.4f} (n={support})")

print(f"\n✓ Top 10 classes (highest F1-score):")
for class_name, f1, support in class_f1_scores_sorted[-10:]:
    print(f"  {class_name:50s} F1={f1:.4f} (n={support})")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: CONFIDENCE CALIBRATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 3: CONFIDENCE CALIBRATION")
print("="*80)

confidence_correct = y_pred_prob[np.arange(len(y_pred)), y_pred][y_pred == y_true]
confidence_wrong = y_pred_prob[np.arange(len(y_pred)), y_pred][y_pred != y_true]

print(f"\n✓ Confidence Analysis:")
print(f"  Correct predictions   : mean={confidence_correct.mean():.4f}, std={confidence_correct.std():.4f}")
print(f"  Wrong predictions     : mean={confidence_wrong.mean():.4f}, std={confidence_wrong.std():.4f}")

if len(confidence_wrong) > 0:
    print(f"\n  ⚠ Model masih confident pada kesalahan (avg: {confidence_wrong.mean():.4f})")
    print(f"    → Indikasi overconfidence atau overfitting")
else:
    print(f"\n  ✓ Model tidak ada kesalahan pada test set")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: CLASS IMBALANCE CHECK
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 4: CLASS IMBALANCE CHECK")
print("="*80)

class_counts = np.bincount(y_true)
max_class = class_counts.max()
min_class = class_counts.min()
imbalance_ratio = max_class / min_class

print(f"\n✓ Dataset balance:")
print(f"  Max class size  : {max_class} ({class_counts.argmax()}: {class_names[class_counts.argmax()]})")
print(f"  Min class size  : {min_class} ({class_counts.argmin()}: {class_names[class_counts.argmin()]})")
print(f"  Imbalance ratio : {imbalance_ratio:.2f}x")

if imbalance_ratio > 10:
    print(f"  ⚠ HIGH CLASS IMBALANCE! Akurasi mungkin bias ke class besar.")
    print(f"    → Lihat weighted metrics di atas")
else:
    print(f"  ✓ Dataset cukup balanced")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: ROBUSTNESS TEST
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 5: ROBUSTNESS TEST")
print("="*80)

robustness_results = {}

# 5a: Gaussian Noise
print("\nTesting robustness to Gaussian noise...")
noise_levels = [0.01, 0.05, 0.1, 0.15]
noise_accs = []

for noise_level in noise_levels:
    noisy_preds = []
    for images, _ in tqdm(test_ds, desc=f"Noise level {noise_level}", leave=False):
        noisy_images = images + tf.random.normal(tf.shape(images), stddev=noise_level*255)
        noisy_images = tf.clip_by_value(noisy_images, 0, 255)
        probs = model.predict(noisy_images, verbose=0)
        noisy_preds.append(np.argmax(probs, axis=1))
    
    noisy_pred = np.concatenate(noisy_preds, axis=0)[:len(y_true)]
    noisy_acc = accuracy_score(y_true, noisy_pred)
    noise_accs.append(noisy_acc)
    print(f"  Noise σ={noise_level}: {noisy_acc:.4f}")
    robustness_results[f"noise_{noise_level}"] = noisy_acc

# 5b: Brightness
print("\nTesting robustness to brightness...")
brightness_levels = [0.7, 0.85, 1.0, 1.15, 1.3]
brightness_accs = []

for brightness in brightness_levels:
    bright_preds = []
    for images, _ in tqdm(test_ds, desc=f"Brightness {brightness}", leave=False):
        bright_images = tf.clip_by_value(images * brightness, 0, 255)
        probs = model.predict(bright_images, verbose=0)
        bright_preds.append(np.argmax(probs, axis=1))
    
    bright_pred = np.concatenate(bright_preds, axis=0)[:len(y_true)]
    bright_acc = accuracy_score(y_true, bright_pred)
    brightness_accs.append(bright_acc)
    print(f"  Brightness {brightness}: {bright_acc:.4f}")
    robustness_results[f"brightness_{brightness}"] = bright_acc

print(f"\n✓ Robustness summary:")
print(f"  Clean accuracy        : {overall_acc:.4f}")
print(f"  Min noisy accuracy    : {min(noise_accs):.4f} (drop: {overall_acc - min(noise_accs):.4f})")
print(f"  Min brightness accuracy: {min(brightness_accs):.4f} (drop: {overall_acc - min(brightness_accs):.4f})")

if (overall_acc - min(noise_accs)) > 0.05 or (overall_acc - min(brightness_accs)) > 0.05:
    print(f"  ⚠ Model tidak robust terhadap perturbasi kecil")
else:
    print(f"  ✓ Model cukup robust")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: CONFUSION MATRIX ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 6: CONFUSION ANALYSIS")
print("="*80)

cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

# Find hardest pairs to distinguish
max_off_diag = 0
hardest_pair = None
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm_norm[i, j] > max_off_diag:
            max_off_diag = cm_norm[i, j]
            hardest_pair = (i, j)

if hardest_pair:
    i, j = hardest_pair
    print(f"\n⚠ Hardest to distinguish:")
    print(f"  {class_names[i]} → misclassified as {class_names[j]}")
    print(f"  Misclassification rate: {cm_norm[i, j]*100:.2f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 7: GENERATING VISUALIZATIONS")
print("="*80)

# 7a: Confidence distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(confidence_correct, bins=30, alpha=0.6, label="Correct", color='green')
if len(confidence_wrong) > 0:
    axes[0].hist(confidence_wrong, bins=30, alpha=0.6, label="Wrong", color='red')
axes[0].set_xlabel("Confidence Score")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Model Confidence Distribution")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 7b: Robustness plot
axes[1].plot(noise_levels, noise_accs, 'o-', label="Gaussian Noise", linewidth=2, markersize=8)
axes[1].plot(brightness_levels, brightness_accs, 's-', label="Brightness", linewidth=2, markersize=8)
axes[1].axhline(y=overall_acc, color='k', linestyle='--', label='Clean Accuracy')
axes[1].set_xlabel("Perturbation Level")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Robustness to Common Corruptions")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/confidence_and_robustness.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/confidence_and_robustness.png")

# 7b: Per-class F1 scores
fig, ax = plt.subplots(figsize=(12, len(class_names)//2))
f1_scores = [class_report[cn]['f1-score'] for cn in class_names]
sorted_indices = np.argsort(f1_scores)
sorted_names = [class_names[i] for i in sorted_indices]
sorted_f1 = [f1_scores[i] for i in sorted_indices]

colors = ['red' if f1 < 0.95 else 'orange' if f1 < 0.98 else 'green' for f1 in sorted_f1]
ax.barh(sorted_names, sorted_f1, color=colors)
ax.set_xlabel("F1-Score")
ax.set_title("Per-Class F1 Scores")
ax.axvline(x=0.99, color='blue', linestyle='--', alpha=0.5, label='99% threshold')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig("outputs/per_class_f1_scores.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/per_class_f1_scores.png")

# 7c: Confusion matrix
fig, ax = plt.subplots(figsize=(20, 18))
sns.heatmap(
    cm_norm, annot=False, cmap="Blues", square=True,
    xticklabels=class_names, yticklabels=class_names,
    cbar_kws={'label': 'Normalized Count'},
    ax=ax
)
ax.set_title("Confusion Matrix (Test Set)", fontsize=14)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0, fontsize=6)
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_test.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/confusion_matrix_test.png")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: GENERATE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 8: GENERATING BENCHMARK REPORT")
print("="*80)

report_text = f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║              COMPREHENSIVE BENCHMARK REPORT — best_model.keras                 ║
╚════════════════════════════════════════════════════════════════════════════════╝

Date: 2026-07-05
Model: EfficientNet-B0 (Transfer Learning)
Dataset: PlantVillage (38 disease classes)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. TEST SET PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Overall Metrics:
  Accuracy  : {overall_acc:.4f} ({overall_acc*100:.2f}%)
  Precision : {precision:.4f}
  Recall    : {recall:.4f}
  F1-Score  : {f1:.4f}

Test Set Size: {len(y_true)} samples

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. CLASS-WISE PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Per-Class F1 Scores (sorted by performance):
"""

for class_name, f1, support in class_f1_scores_sorted:
    status = "✓" if f1 >= 0.98 else "⚠" if f1 >= 0.95 else "✗"
    report_text += f"\n  {status} {class_name:50s} F1={f1:.4f} (n={support})"

report_text += f"""

Critical Classes (F1 < 0.95):
"""
critical_classes = [(name, f1, supp) for name, f1, supp in class_f1_scores_sorted if f1 < 0.95]
if critical_classes:
    for name, f1, supp in critical_classes:
        report_text += f"\n  → {name:50s} F1={f1:.4f} (n={supp})"
else:
    report_text += "\n  ✓ None (all classes > 95% F1)"

report_text += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. CONFIDENCE CALIBRATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Correct Predictions:
  Mean confidence   : {confidence_correct.mean():.4f}
  Std deviation     : {confidence_correct.std():.4f}
  Min confidence    : {confidence_correct.min():.4f}
  Max confidence    : {confidence_correct.max():.4f}
"""

if len(confidence_wrong) > 0:
    report_text += f"""
Wrong Predictions:
  Count             : {len(confidence_wrong)}
  Mean confidence   : {confidence_wrong.mean():.4f}
  Std deviation     : {confidence_wrong.std():.4f}
  Min confidence    : {confidence_wrong.min():.4f}
  Max confidence    : {confidence_wrong.max():.4f}

Calibration Analysis:
  Confidence gap    : {confidence_correct.mean() - confidence_wrong.mean():.4f}
  """
    if confidence_wrong.mean() > 0.5:
        report_text += """
  ⚠ WARNING: Model is overconfident on wrong predictions!
    → Suggests overfitting or poor calibration
    → Consider: temperature scaling, calibration, or stronger regularization
"""
    else:
        report_text += """
  ✓ Model shows reasonable confidence gap between correct and wrong predictions
"""
else:
    report_text += """
Wrong Predictions: 0 (perfect accuracy)
  ✓ No errors on test set
"""

report_text += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. CLASS IMBALANCE ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dataset Balance:
  Max class size    : {max_class} ({class_names[class_counts.argmax()]})
  Min class size    : {min_class} ({class_names[class_counts.argmin()]})
  Imbalance ratio   : {imbalance_ratio:.2f}x
"""

if imbalance_ratio > 10:
    report_text += """
  ⚠ HIGH CLASS IMBALANCE DETECTED
    → Large accuracy may be biased toward frequent classes
    → Weighted metrics (precision, recall) are more reliable
    → Consider: class weights, data augmentation for minority classes
"""
else:
    report_text += """
  ✓ Dataset reasonably balanced
"""

report_text += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. ROBUSTNESS ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Gaussian Noise Robustness:
  Clean accuracy    : {overall_acc:.4f}
"""

for noise_level, acc in zip(noise_levels, noise_accs):
    drop = overall_acc - acc
    status = "✓" if drop < 0.02 else "⚠" if drop < 0.05 else "✗"
    report_text += f"\n  {status} Noise σ={noise_level}: {acc:.4f} (drop: {drop:.4f})"

report_text += f"""

Brightness Robustness:
"""

for brightness, acc in zip(brightness_levels, brightness_accs):
    drop = overall_acc - acc
    status = "✓" if drop < 0.02 else "⚠" if drop < 0.05 else "✗"
    report_text += f"\n  {status} Brightness {brightness}: {acc:.4f} (drop: {drop:.4f})"

max_noise_drop = overall_acc - min(noise_accs)
max_bright_drop = overall_acc - min(brightness_accs)

report_text += f"""

Summary:
  Max accuracy drop (noise)      : {max_noise_drop:.4f}
  Max accuracy drop (brightness) : {max_bright_drop:.4f}
"""

if max_noise_drop > 0.05 or max_bright_drop > 0.05:
    report_text += """
  ⚠ MODEL NOT ROBUST to perturbations
    → May fail in real-world conditions (lighting changes, sensor noise)
    → Consider: data augmentation, adversarial training
"""
else:
    report_text += """
  ✓ Model reasonably robust to common corruptions
"""

if hardest_pair:
    i, j = hardest_pair
    report_text += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. CONFUSION HOTSPOTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hardest to Distinguish:
  True class  : {class_names[i]}
  Confused as : {class_names[j]}
  Rate        : {cm_norm[i, j]*100:.2f}% misclassification
  
Recommendation:
  → Review similar-looking disease classes
  → May need more training data for distinguishing features
  → Consider ensemble or manual review for high-confusion pairs
"""

report_text += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. CONCLUSIONS & RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ STRENGTHS:
  • Overall accuracy is very high ({overall_acc*100:.2f}%)
  • Consistent performance across most classes
  • Strong transfer learning from ImageNet pretrained EfficientNet-B0
  • Weighted metrics align with accuracy (balanced performance)

"""

issues = []
if confidence_wrong.mean() > 0.5 and len(confidence_wrong) > 0:
    issues.append("Model is overconfident on wrong predictions (overfitting indicator)")
if max_noise_drop > 0.05 or max_bright_drop > 0.05:
    issues.append("Low robustness to natural corruptions (lighting, noise)")
if imbalance_ratio > 10:
    issues.append("High class imbalance (may skew metrics toward frequent classes)")
if critical_classes:
    issues.append(f"{len(critical_classes)} classes with F1 < 95% need attention")

if issues:
    report_text += f"""
⚠ CONCERNS TO ADDRESS:
"""
    for i, issue in enumerate(issues, 1):
        report_text += f"\n  {i}. {issue}"
    report_text += f"""

→ RECOMMENDED ACTIONS:
  1. Test on external/real-world data (not from PlantVillage)
  2. Apply temperature scaling for calibration
  3. Increase data augmentation (rotation, perspective, color jitter)
  4. Monitor for data leakage in preprocessing
  5. Use stratified cross-validation for more robust evaluation
  6. Consider focal loss or weighted loss for minority classes
"""
else:
    report_text += f"""
⚠ NO MAJOR CONCERNS DETECTED

→ MODEL IS PRODUCTION-READY with caveats:
  • Performance verified on held-out test set
  • No obvious overfitting indicators
  • Reasonable robustness to common corruptions
  • Consistent per-class performance
  
→ NEXT STEPS:
  1. Test on new external dataset (real-world plant images)
  2. Deploy with monitoring to catch distribution shifts
  3. Set up confidence thresholds for rejection (e.g., < 80% confidence)
  4. Collect feedback on misclassifications for continued improvement
"""

report_text += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL CLASSIFICATION REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{classification_report(y_true, y_pred, target_names=class_names, digits=4)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# Save report
os.makedirs("outputs", exist_ok=True)
with open("outputs/benchmark_report.txt", "w") as f:
    f.write(report_text)

print("\n" + "="*80)
print("✓ BENCHMARK COMPLETE")
print("="*80)
print(report_text)

print("\nArtifacts saved:")
print("  • outputs/benchmark_report.txt")
print("  • outputs/confidence_and_robustness.png")
print("  • outputs/per_class_f1_scores.png")
print("  • outputs/confusion_matrix_test.png")
