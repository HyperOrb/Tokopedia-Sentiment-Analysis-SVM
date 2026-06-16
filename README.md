# 🛒 Tokopedia Sentiment Analysis: Advanced SVM & XAI Implementation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikitlearn)
![XAI](https://img.shields.io/badge/AI-Explainable--AI-red)
![Accuracy](https://img.shields.io/badge/Accuracy-96.22%25-green?style=for-the-badge)

## 📌 Project Overview
Projek ini mengklasifikasikan sentimen ulasan produk Tokopedia (Positif, Netral, Negatif) menggunakan algoritma **Support Vector Machine (SVM)** dengan akurasi akhir mencapai **96.22%** setelah hyperparameter tuning. Projek ini menerapkan pendekatan **Explainable AI (XAI)** menggunakan LIME dan SHAP untuk memberikan transparansi dan interpretabilitas penuh pada prediksi model.

---

## 🚀 Key Technical Highlights
* **Explainable AI (XAI)**: Menggunakan framework **LIME** (Local Interpretable Model-agnostic Explanations) dan **SHAP** (SHapley Additive exPlanations) untuk memvisualisasikan kontribusi kata-kata spesifik terhadap prediksi model RBF SVM. Analisis koefisien bobot juga dilakukan pada model baseline linear SVM.
* **Data Augmentation**: Menerapkan teknik *Synonym Replacement* menggunakan kamus sinonim kustom serta *Random Insertion* dan *Random Deletion* untuk meningkatkan jumlah dataset sebesar 200% (dari 2.775 baris menjadi 8.325 baris) demi melatih model yang lebih kuat.
* **Custom Feature Engineering**: Mengintegrasikan 4 fitur tambahan ke representasi TF-IDF, yaitu: panjang karakter teks, jumlah kata (*word count*), intensitas penggunaan tanda baca (`!`, `?`, `.`), dan jumlah huruf kapital.
* **Class Balancing**: Menyelesaikan masalah ketidakseimbangan kelas (kelas mayoritas positif memiliki ~38k ulasan, sedangkan negatif hanya 925 ulasan) dengan melakukan *downsampling* kelas positif dan netral menjadi masing-masing **925 sampel**, sehingga seimbang pada dataset dasar sebanyak 2.775 ulasan.
* **Hyperparameter Grid Search**: Melakukan fine-tuning model menggunakan `GridSearchCV` dengan **3-fold Cross Validation** untuk mengoptimalkan parameter `C` dan `gamma` pada kernel RBF.

---

## 💻 Tech Stack

| Category | Tools & Libraries |
| :--- | :--- |
| **Language** | Python 3.8+ |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (SVC, TF-IDF Vectorizer, StandardScaler, GridSearchCV, Resampling) |
| **Explainable AI (XAI)** | LIME, SHAP, SVM Coefficient Analysis |
| **NLP (Indonesian)** | RegEx Text Preprocessing |
| **Visualization** | Matplotlib, Seaborn |

---

## 🛠️ Methodology & Workflow

### 1. Preprocessing Pipeline
* **Text Cleaning**: Menghapus URL menggunakan RegEx kustom.
* **Normalization**: Case folding (mengubah semua teks menjadi huruf kecil), menghapus karakter non-alfabet (menyisakan huruf a-z dan spasi), serta menghapus spasi berlebih (*whitespace collapsing*).

### 2. Feature Extraction
Model menggabungkan **TF-IDF Vectorizer** (maksimal 5000 fitur) dengan **Behavioral Features** (panjang karakter, jumlah kata, kemunculan tanda seru/tanya/titik, dan jumlah huruf besar) yang telah dinormalisasi menggunakan `StandardScaler` menggunakan fungsi `hstack` dari SciPy.

### 3. Model Architecture & XAI
Model terbaik menggunakan **SVM dengan Kernel Radial Basis Function (RBF)** dengan fungsi keputusan:
$$f(x) = \text{sign}\left(\sum_{i \in SV} \alpha_i y_i \exp(-\gamma \|x_i - x\|^2) + b\right)$$

Interpretasi model ditangani menggunakan model penjelasan lokal:
* **LIME**: Menjelaskan kontribusi kata-kata secara lokal untuk ulasan individual.
* **SHAP**: Menyediakan visualisasi nilai Shapley untuk mengukur signifikansi fitur secara global maupun lokal.

---

## 📈 Evaluation Results

Berikut adalah hasil perbandingan model pada dataset pengujian (test set):

| Model Stage | Accuracy | Improvement |
| :--- | :--- | :--- |
| **Model 1: TF-IDF Only** | **63.78%** | Baseline |
| **Model 2: TF-IDF + Feature Engineering** | **64.14%** | +0.56% |
| **Model 3: TF-IDF + FE + Data Augmentation (Linear)** | **85.41%** | +33.90% dari Baseline |
| **Model Tuned: RBF Kernel (C=100, gamma=scale)** | **96.22%** | +12.66% dari Model 3 |

### Classification Report (Model Terbaik - Tuned RBF):

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Negative** | 0.97 | 0.99 | 0.98 | 564 |
| **Neutral** | 0.96 | 0.94 | 0.95 | 564 |
| **Positive** | 0.96 | 0.96 | 0.96 | 537 |
| **Accuracy** | | | **0.96** | 1665 |
| **Macro Avg** | 0.96 | 0.96 | 0.96 | 1665 |
| **Weighted Avg** | 0.96 | 0.96 | 0.96 | 1665 |

---

## 📂 Project Structure
```text
.
├── data/                   # Raw & Processed Datasets
├── notebook/               # Jupyter Notebook
└── models/                 # Saved SVM Models (.pkl)
