# 😷 Face Mask Detection  
A deep learning project that classifies whether a person is **Wearing a Mask** or **Not Wearing a Mask** using a custom dataset of facial images.  
This repository includes the entire pipeline — from loading images to preprocessing, model building, training, evaluation, and real-time prediction.

---

## ⭐ About the Project
Face mask detection became an essential computer vision application during the COVID-19 pandemic.  
This project uses a convolutional neural network (CNN) to automatically identify if a person in an image is wearing a mask.

The notebook covers:
- Dataset loading  
- Image preprocessing  
- Data augmentation  
- CNN model creation  
- Training and validation  
- Evaluation metrics  
- Predictions on new images  

---

## 📂 Dataset Information
The dataset contains two categories of images:

### 🔹 Classes Included:
- **With Mask**
- **Without Mask**

Each image is preprocessed using:
- Resizing  
- Normalization  
- Data augmentation (rotation, zoom, flip, shifts)

---

## 💻 Code Snippet — Loading Dataset
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    "dataset/", 
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    "dataset/", 
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)
```

## 🧰 Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

Jupyter Notebook

## 🧪 Steps Performed in the Notebook
1️⃣ Dataset Preparation

Import and organize images

Apply augmentation

Set up training & validation generators

2️⃣ Image Preprocessing

Scaling

Resizing

Normalization

3️⃣ CNN Model Architecture

Layers include:

Convolution layers

MaxPooling layers

Flatten layer

Dense layers

Dropout layers

Output layer with sigmoid activation

4️⃣ Model Training

Train on augmented images

Use validation data for monitoring

Loss & accuracy tracking

5️⃣ Model Evaluation

Metrics used:

Accuracy

Loss curves

Confusion matrix (optional)

Prediction on test samples

## 🎯 Outcome

The trained model achieves strong accuracy in identifying masked and unmasked faces, making it suitable for real-time applications such as:

Surveillance systems

Workplace monitoring

Public safety checkpoints

## 📁 Repository Structure
📦 Face-Mask-Detection
├── Face Mask Detection.ipynb
├── dataset/
│   ├── with_mask/
│   └── without_mask/
├── README.md
└── requirements.txt

▶️ How to Run This Project
```bash
1. Clone the Repository
git clone https://github.com/Monishsasi/Deep-Learning-Projects/tree/main/Face%20Mask%20Detection
cd Face-Mask-Detection

2. Install Dependencies
pip install -r requirements.txt

3. Launch Jupyter Notebook
jupyter notebook

4. Open the Notebook
Face Mask Detection.ipynb
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.
Feel free to fork this repository and submit a pull request.

## 📜 License

This project is licensed under the MIT License — free to use, modify, and distribute.

##👤 Contact

If you have questions or suggestions, feel free to reach out.

Author: Monish Sasikumar
GitHub: https://github.com/Monishsasi