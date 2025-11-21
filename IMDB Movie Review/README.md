# 🎬 IMDB Movie Review Sentiment Analysis  
A natural language processing (NLP) project that classifies IMDB movie reviews as **Positive** or **Negative** using machine learning / deep learning techniques.  
This repository contains the full workflow — dataset loading, text preprocessing, model building, training, and evaluation.

---

## ⭐ About the Project
Sentiment analysis plays a significant role in understanding public opinions and improving user experience.  
This project analyzes movie reviews and predicts whether the sentiment is positive or negative.

The notebook includes:
- Text preprocessing  
- Tokenization  
- Padding  
- Model building  
- Model training  
- Evaluation  
- Prediction on new reviews  

---

## 📂 Dataset Information
This project uses the **IMDB Movie Reviews Dataset**, which is **inbuilt in Keras**.

### 🔹 How to Load the Built-in Dataset

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Pad sequences
X_train = pad_sequences(X_train, maxlen=200)
X_test = pad_sequences(X_test, maxlen=200)

# Display sample
print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
```

## 🔹 Labels

0 → Negative Review

1 → Positive Review

🔹 Example Features

Each review is converted into sequences of word indexes:

Word index 1 → most frequent word

Word index 2 → second most frequent

The top 10,000 most common words are used.

##🧰 Technologies Used

Python

TensorFlow / Keras

NumPy

Pandas

Matplotlib / Seaborn

NLP tokenizers

Jupyter Notebook

##🧪 Steps Performed in the Notebook
1️⃣ Load & Explore Dataset

Load IMDB dataset

Check reviews and sentiment distribution

Inspect encoded sequences

2️⃣ Text Preprocessing

Tokenization

Sequence padding

Limiting vocabulary size

Converting sequences to fixed length

3️⃣ Model Building

Possible models include:

Embedding + LSTM

Embedding + GRU

Embedding + 1D CNN

Fully connected classifier

Layers used:

Embedding layer

LSTM / GRU

Dense layers

Sigmoid output

4️⃣ Model Training

Use binary crossentropy

Adam optimizer

Train for multiple epochs

Track loss & accuracy

5️⃣ Model Evaluation

Metrics measured:

Accuracy

Loss

Confusion matrix (optional)

Training graphs

6️⃣ Predicting New Reviews

Convert raw text → tokenized → padded → prediction.

## 🎯 Outcome

The trained model typically achieves 85%–90% accuracy, making it effective for sentiment classification tasks.

## 📁 Repository Structure
📦 IMDB-Movie-Review-Sentiment-Analysis
├── IMDB Movie Review Sentiment Analysis.ipynb
├── README.md
└── requirements.txt

## ▶️ How to Run This Project
```bash
1. Clone the Repository
git clone https://github.com/Monishsasi/Deep-Learning-Projects/tree/main/IMDB%20Movie%20Review

2. Install Dependencies
pip install -r requirements.txt

3. Launch Jupyter Notebook
jupyter notebook

4. Open the Notebook
IMDB Movie Review Sentiment Analysis.ipynb
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.
Feel free to fork this repository and submit a pull request.

## 📜 License

This project is licensed under the MIT License — free to use, modify, and distribute.

## 👤 Contact

Author: Monish Sasikumar
GitHub: https://github.com/Monishsasi