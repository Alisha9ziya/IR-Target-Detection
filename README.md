# 🔍 IR Target Detection (IRSTD-1K)

This project implements an **infrared small target detection pipeline** and evaluates its performance on the **IRSTD-1K dataset** using standard metrics such as **Probability of Detection (PD)** and **False Alarm Rate (FA)**.

---

## 🚀 Overview

Detecting small infrared targets is challenging due to:

* Low contrast
* Background clutter
* Noise interference

This project applies image processing techniques to accurately detect targets and minimize false alarms.

---

## ⚙️ Pipeline

1. Load infrared images
2. Apply detection algorithm (`detect_targets`)
3. Generate binary mask of detected regions
4. Compare with Ground Truth labels
5. Compute evaluation metrics (PD, FA)
6. Save detection outputs and results

---

## 📊 Final Results (IRSTD-1K Dataset)

**Average Performance:**

* **PD (Probability of Detection):** 0.5343
* **FA (False Alarm Rate):** 0.010155

---

## 📁 Results

* 📄 [Final Metrics](results/final_metrics.txt)
* 📄 [Per-Image Metrics](results/per_image_metrics.txt)
* 📊 [CSV File](results/metrics.csv)
* 🖼️ [Detection Outputs](results/outputs/)

---

## 🧪 Dataset

* **Dataset Used:** IRSTD-1K
* Ground truth masks are used for evaluation

---

## ▶️ How to Run

```bash
python src/main.py
```

Make sure to update dataset paths in `main.py` before running.

---

## 🛠️ Tech Stack

* Python
* OpenCV
* NumPy
---




