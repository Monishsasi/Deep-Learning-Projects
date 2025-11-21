# 🩺 Breast Cancer Prediction  
A machine learning project that classifies breast tumors as **Malignant** or **Benign** using the *Wisconsin Breast Cancer Dataset*.  
This repository contains the complete workflow — from data loading to preprocessing, model building, evaluation, and comparison.

---

## ⭐ About the Project
Breast cancer diagnosis plays a crucial role in early detection and patient survival.  
This project builds a supervised ML model that predicts tumor type using various clinical measurements from digitized images of breast masses.

The notebook covers:
- Data preprocessing  
- Exploratory data analysis (EDA)  
- Feature scaling  
- Model building  
- Model comparison  
- Final evaluation  

---

## 📂 Dataset Information
The dataset used is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, available **inbuilt in Scikit-learn**.

### 🔹 How to Load the Built-in Dataset
```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load dataset
data = load_breast_cancer()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column
df['target'] = data.target

# View dataset
df.head()
```

## 🔹 Features Include:

Mean radius

Texture

Perimeter

Area

Smoothness

Compactness

Concavity

Symmetry

Fractal dimension

...and many more

## 🔹 Target Values:

0 → Malignant (Cancerous)

1 → Benign (Non-Cancerous)

## 🧰 Technologies Used

Python

NumPy

Pandas

Matplotlib / Seaborn

Scikit-learn

Jupyter Notebook

## 🧪 Steps Performed in the Notebook
1️⃣ Load & Explore Data

Import dataset

Check structure, missing values, summary statistics

Explore basic distributions

2️⃣ Data Preprocessing

Encode target

Drop redundant columns (if any)

Feature scaling using StandardScaler

3️⃣ Exploratory Data Analysis

Correlation heatmap

Pairplots

Feature distribution plots

Important features exploration

4️⃣ Train-Test Split

80% training

20% testing

5️⃣ Model Building

Models evaluated:

Logistic Regression

Support Vector Machine (SVM)

Random Forest

KNN

Decision Tree

6️⃣ Model Evaluation

Metrics measured:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

ROC Curve (optional)

## 🎯 Outcome

The top-performing model achieves 95–99% accuracy, suitable for early diagnostic assistance.

## 📁 Repository Structure

 📦 Breast-Cancer-Prediction
 ├── Breast Cancer.ipynb
 ├── README.md
 └── requirements.txt

## ▶️ How to Run This Project
1. Clone the Repository
```bash
git clone https://github.com/your-username/Breast-Cancer-Prediction.git
cd Breast-Cancer-Prediction
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Launch Jupyter Notebook
```bash
jupyter notebook
```

4. Open the Notebook
```bash
Breast Cancer.ipynb
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.
Feel free to fork this repository and submit a pull request.

## 📜 License

This project is licensed under the MIT License — free to use, modify, and distribute.

## 👤 Contact

If you have questions or suggestions, feel free to reach out.

Author: Monish Sasikumar
GitHub: https://github.com/Monishsasi