# 🌿 AI Plant Disease Detector

An intelligent web application that detects plant diseases using Deep Learning.

---

## 🚀 Features

- 🌱 Detect plant diseases from leaf images  
- 🧠 Deep Learning model (MobileNetV2)  
- 📊 Confidence score & visualization  
- 🛠 Solution and prevention tips  
- ⚡ Fast and user-friendly interface  

---

## 🧠 Model Details

- Model: Transfer Learning (MobileNetV2)
- Accuracy: ~95%
- Classes:
  - Potato Early Blight
  - Potato Healthy
  - Tomato Early Blight
  - Tomato Late Blight

---

## 🖼️ How It Works

1. Upload a leaf image  
2. AI analyzes the image  
3. Predicts disease  
4. Shows:
   - Prediction
   - Confidence
   - Description
   - Solution
   - Prevention  

---

## 🛠 Tech Stack

- Python  
- TensorFlow  
- Streamlit  
- NumPy  
- Matplotlib  

---

## 📂 Project Structure
app.py best_model.h5 requirements.txt
---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
