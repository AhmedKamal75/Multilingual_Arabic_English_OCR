# Multilingual Arabic-English SVTR OCR Recognizer

This project explores **Scene Text Recognition (STR)** using the **SVTR (Scene Visual Transformer)** architecture with **deformable attention** for multilingual OCR in **Arabic and English**.  

The work was conducted on **Kaggle GPUs (Tesla P100-PCIE-16GB)** under resource limits of **30 hours per week** and **12 hours per session**. Training was done in multiple experiments to evaluate dataset complexity, model scaling, and augmentation strategies.

---

## 📑 Contents
- [Motivation](#-motivation)
- [Datasets](#-datasets)
- [Model Architecture](#-model-architecture)
- [Experiments & Results](#-experiments--results)
  - [First Attempt](#1-first-attempt-easy-dataset)
  - [Second Attempt](#2-second-attempt-small-model--harder-dataset)
  - [Third Attempt](#3-third-attempt-large-model--scaling)
- [Key Learnings](#-key-learnings)
- [Future Work](#-future-work)
- [References](#-references)

---

## 🎯 Motivation

The goal was to build a **robust OCR system** capable of recognizing both **Arabic** and **English** text, trained from scratch on synthetic datasets.  

Initial results showed that simple datasets lead to overfitting and artificially high scores, motivating the creation of **harder synthetic datasets** and **larger models**.

---

## 📂 Datasets

| Dataset | Source | Samples Used | Notes |
|---------|--------|------|-------|
| [Arabic-English OCR Dataset](https://www.kaggle.com/datasets/ahmedkamal75/arabic-english-ocr-dataset) | TRDG synthetic | 80,000 | Easy, clean — model overfits easily |
| [Arabic-English OCR Synthetic Dataset V2](https://www.kaggle.com/datasets/ahmedkamal75/arabic-english-ocr-synthatic-dataset-v2) | TRDG synthetic | 100,000 – 120,000 | Noisy, varied, more challenging |

---

## 🏗 Model Architecture

**SVTR with deformable attention**  
- Input: `(H, W, C)` = (64 or 32, 256–512, 3)  
- Embedding dims: `[128, 256, 384]`  
- Heads: `[4, 8, 12]`  
- MLP ratio: `2`, Dropout: `0.1`  
- Local types: `non_overlapping`, `deformable`
- Deformable attention: `n_points=9`, `offset_scale=4.0`  

Two main sizes tested:
- **SMALL**: `num_blocks=[3, 6, 3]`  
- **LARGE**: `num_blocks=[3, 12, 3]`

---

## 🔬 Experiments & Results

### 1. First Attempt: Easy Dataset
- **Date**: Sep 4, 2025  
- **Runtime**: 10h 49m  
- **Dataset**: [Arabic-English OCR Dataset](https://www.kaggle.com/datasets/ahmedkamal75/arabic-english-ocr-dataset) (80k samples)  
- **Config**: Input (64×512×3), Large model, 20 epochs  

**Validation Metrics**
| Metric | Value |
|--------|-------|
| Sequence Accuracy | 0.8287 |
| Character Accuracy | 0.9187 |
| Word Accuracy | 0.9422 |
| CER | 0.0193 |
| WER | 0.0663 |

**Test Metrics**
| Metric | Value |
|--------|-------|
| Sequence Accuracy | 0.8275 |
| Character Accuracy | 0.9214 |
| Word Accuracy | 0.9424 |
| CER | 0.0185 |
| WER | 0.0667 |

> ⚠️ **Observation**: Excellent results, but dataset was too easy → not realistic, high overfitting.

---

### 2. Second Attempt: Small Model + Harder Dataset
- **Date**: Sep 10, 2025  
- **Runtime**: 19h 29m  
- **Dataset**: [Synthetic Dataset V2](https://www.kaggle.com/datasets/ahmedkamal75/arabic-english-ocr-synthatic-dataset-v2) (100k samples)  
- **Config**: Input (64×256×3), SMALL model, 40 epochs + continued training  

**Best Continuous Training Model** → [Epoch 42](https://www.kaggle.com/models/ahmedkamal75/svtr_deformable_epoch_42_best)  

**Final Test Metrics**
| Metric | Value |
|--------|-------|
| Sequence Accuracy | 0.6194 |
| Character Accuracy | 0.8335 |
| Word Accuracy | 0.8419 |
| CER | 0.0655 |
| WER | 0.1915 |

> ✅ More realistic, but lower performance. Demonstrated the need for a **larger model** and more samples.

---

### 3. Third Attempt: Large Model + Scaling
- **Date**: Sep 10–12, 2025  
- **Dataset**: [Synthetic Dataset V2](https://www.kaggle.com/datasets/ahmedkamal75/arabic-english-ocr-synthatic-dataset-v2) (120k samples)  
- **Config**: Input (64×256×3 → 32×256×3), LARGE model, multiple training stages  

**Key Checkpoints**
### Key Checkpoints

- **[Epoch 46](https://www.kaggle.com/models/akamalkaggle117511/svtr_deformable_large_epoch_46)**: Used height 64
- **[Epoch 56](https://www.kaggle.com/models/ahmedkamal75/svtr_deformable_large_epoch_56)**: Used height 32
- **[Epoch 58](https://www.kaggle.com/models/ahmedkamal75/svtr_deformable_large_epoch_58)**: Used height 32

**Best Continuous Training Model (Epoch 58)**

**Final Test Metrics**
| Metric | Value |
|--------|-------|
| Sequence Accuracy | 0.6637 |
| Character Accuracy | 0.8476 |
| Word Accuracy | 0.8606 |
| CER | 0.0491 |
| WER | 0.1615 |

> 🚀 **Observation**: Larger model + more samples significantly improved robustness. Lower CER/WER compared to Small version.

---

## 📌 Key Learnings

1. **Dataset matters more than model size**:  
   - Easy datasets → inflated scores (Attempt 1).  
   - Harder synthetic datasets → realistic but challenging results.

2. **Need for a comprehensive dataset**:  
   - Synthetic datasets are not enough to capture the variability of real-world text images.  
   - A **handwritten dataset** is necessary to achieve good performance on handwritten text images.  
   - Preferably, a **dataset with a mix of synthetic and real-world images** can also be used to leverage the strengths of both.  
   

3. **Model scaling works**:  
   - Large model (Attempt 3) consistently outperformed Small model (Attempt 2).  

4. **Aggressive augmentation helps**:  
   - Higher augmentation strength (0.8) improved generalization.  

5. **Input resolution affects performance**:  
   - Reducing input height (64 → 32) balanced performance and training time reduced by half.  

---

## 🔮 Future Work

- Incorporate **real-world Arabic-English text images** (not just synthetic).  
- Explore **semi-supervised learning** with unlabeled text images.  
- Benchmark against **other STR models** (eg. SVTRv2).  

---

## 📚 References

- [SVTR: Scene Text Recognition with a Single Visual Model](https://arxiv.org/abs/2205.00159)  
- [Kaggle Notebook (Logs)](https://www.kaggle.com/code/ahmedkamal75/multilingual-arabic-english-svtr-ocr-recongnizer)  
- [Dataset V1](https://www.kaggle.com/datasets/ahmedkamal75/arabic-english-ocr-dataset)  
- [Dataset V2](https://www.kaggle.com/datasets/ahmedkamal75/arabic-english-ocr-synthatic-dataset-v2)  
- [Best Large Model (Epoch 58)](https://www.kaggle.com/models/ahmedkamal75/svtr_deformable_large_epoch_58)  

---

✨ **Final Note**:  
This project demonstrated how **dataset complexity**, **augmentation strength**, and **model scaling** interact in multilingual OCR. The final Large model reached a **CER of ~0.05 and WER of ~0.16** on a challenging synthetic dataset — a strong baseline for future multilingual OCR research.
