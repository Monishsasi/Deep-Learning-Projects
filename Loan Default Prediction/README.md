# 🏦 Loan Default Prediction – Machine Learning Project

This project builds a **binary classification model** to predict whether a loan applicant is likely to **default** or **repay**, based on financial, demographic, and loan-related attributes.  
The goal is to assist organizations in making data-driven lending decisions by detecting high-risk borrowers early.

---

## 📘 Project Overview

Loan defaults are critical risks for banks and lending agencies. Inaccurate default predictions can lead to:

- Financial losses  
- Higher interest rates  
- Increased risk exposure  

This project covers an end-to-end machine learning pipeline including:

- Data understanding & analysis  
- Preprocessing & feature engineering  
- Class imbalance handling  
- Deep learning model training  
- Hyperparameter tuning using **Keras Tuner**  
- Final evaluation and insights  

---

## 📂 Project Structure

Loan Default.ipynb → Complete machine learning notebook
README.md → Documentation
requirements.txt → Python dependencies (optional)


---

## 📊 Dataset Information

The dataset includes various borrower-related features such as:

- Financial background  
- Income information  
- Loan details  
- Payment history  
- Demographics  

The target variable indicates whether the applicant **defaulted (1)** or **repaid (0)**.

The dataset shows:

- Missing values in select columns  
- Several categorical features  
- A significant **class imbalance** (defaulters are much fewer)

---

## 🔧 Workflow Summary

### 1️⃣ Importing Dependencies & Loading Data
The notebook loads essential libraries:

- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Imbalanced-learn  
- TensorFlow / Keras  
- Keras Tuner  

Initial inspection includes checking shape, missing values, datatypes, and sample records.

---

### 2️⃣ Basic Information Analysis

Performed tasks:

- `.info()`, `.describe()`  
- Missing value analysis  
- Target distribution check  
- Data type classification  
- Basic statistical overview  

---

### 3️⃣ Exploratory Data Analysis (EDA)

The notebook visualizes patterns using:

- Histograms  
- Count plots  
- Boxplots for outlier detection  
- Heatmap for correlation  
- Loan default distributions across categorical features  

EDA highlights relationships that influence loan default behavior.

---

### 4️⃣ Data Preprocessing

Preprocessing steps include:

- Handling missing values  
- Encoding categorical features  
- Scaling numerical values  
- Splitting features and target  
- Converting data into arrays for model compatibility  

---

### 5️⃣ Handling Class Imbalance

Applied **RandomUnderSampler** from Imbalanced-learn:

- Balances the dataset  
- Reduces majority class samples  
- Helps model learn minority class patterns better  

---

### 6️⃣ Train-Test Split

Dataset is split into:

- **Training set**  
- **Testing set**

Balanced training data is used for model learning.

---

### 7️⃣ Model Development

A **Deep Neural Network (DNN)** is built using Keras.  
Key components include:

- Dense layers  
- ReLU activation  
- Dropout regularization  
- Adam optimizer  

To optimize performance, **Keras RandomSearch Tuner** is used for:

- Number of layers  
- Neurons per layer  
- Learning rate  
- Dropout percentage  

---

### 8️⃣ Model Training & Evaluation

The best model from hyperparameter tuning is trained and evaluated.

Metrics analyzed:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

### 📌 Insight  
The model shows strong performance for the majority class (non-defaulters).  
However, predicting minority class (defaulters) remains challenging due to inherent class imbalance.

---

## 🚀 Future Improvements

- Use **SMOTE** to oversample defaulters  
- Try **XGBoost / LightGBM** for comparison  
- Experiment with cost-sensitive learning  
- Add advanced feature engineering  
- Deploy using **FastAPI / Streamlit**  

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| Programming | Python |
| Deep Learning | TensorFlow, Keras |
| Optimization | Keras Tuner |
| Visualization | Matplotlib, Seaborn |
| Data Handling | Pandas, NumPy |
| Imbalance Handling | Imbalanced-learn |
| Evaluation | Scikit-learn |

---

## 📁 How to Run This Project

### 1. Clone the repository

```bash
git clone https://github.com/Monishsasi/Deep-Learning-Projects/tree/main/Loan%20Default%20Prediction
