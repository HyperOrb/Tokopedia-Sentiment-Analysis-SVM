# 🛒 Tokopedia Sentiment Analysis: Advanced SVM & XAI Implementation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikitlearn)
![XAI](https://img.shields.io/badge/AI-Explainable--AI-red)
![Accuracy](https://img.shields.io/badge/Accuracy-95%25-green?style=for-the-badge)

## 📌 Project Overview
Projek ini mengklasifikasikan sentimen ulasan produk Tokopedia (Positif, Netral, Negatif) menggunakan algoritma **Support Vector Machine (SVM)** dengan akurasi **95%**. Projek ini menerapkan pendekatan **Explainable AI (XAI)** untuk memberikan transparansi pada prediksi model.

---

## 🚀 Key Technical Highlights
* **Explainable AI (XAI)**: Menggunakan analisis koefisien pada *Linear Kernel* untuk mengidentifikasi kata kunci utama yang paling berpengaruh terhadap hasil klasifikasi sentimen.
* **Data Augmentation**: Teknik *Synonym Replacement* dan *Random Insertion/Deletion* untuk meningkatkan variasi data dan ketahanan model.
* **Custom Feature Engineering**: Integrasi fitur linguistik (panjang teks, intensitas tanda baca) ke dalam representasi TF-IDF.
* **Class Balancing**: Menerapkan teknik *resampling* pada 500 ulasan dari 5 kategori produk berbeda (Fashion, Electronics, Food, dll.).

---

## 💻 Tech Stack

| Category | Tools & Libraries |
| :--- | :--- |
| **Language** | Python 3.8+ |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (SVC, TF-IDF, Resampling) |
| **Explainable AI (XAI)** | SVM Coefficient Analysis |
| **NLP (Indonesian)** | Sastrawi, RegEx |
| **Visualization** | Matplotlib, Seaborn |

---

## 🛠️ Methodology & Workflow

### 1. Preprocessing Pipeline
* **Text Cleaning**: Menghapus URL, karakter non-alfabet, dan angka.
* **Normalization**: Case folding dan Stemming menggunakan library **Sastrawi**.

### 2. Feature Extraction
Model menggabungkan **TF-IDF Vectorizer** dengan **Behavioral Features** seperti pola penggunaan tanda seru (!) dan tanda tanya (?) sebagai indikator emosi.

### 3. Model Architecture & XAI
Menggunakan **SVM** dengan *Linear Kernel* yang memungkinkan interpretasi model secara langsung:
$$f(x) = \text{sign}(w \cdot x + b)$$
Analisis pada bobot $w$ dilakukan untuk memastikan transparansi keputusan model (XAI).

---

## 📈 Evaluation Results

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **95.00%** |
| **Precision** | 0.94 - 0.96 |
| **Recall** | 0.94 - 0.96 |
| **F1-Score** | 0.95 |

---

## 📂 Project Structure
```text
.
├── data/                   # Raw & Processed Datasets
├── notebook/               # Jupyter Notebook
└──  models/                # Saved SVM Models (.pkl)
