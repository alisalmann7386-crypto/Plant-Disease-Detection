# 🌿 AgroVision AI - Plant Disease Detection System

<p align="center">
  <img src="assets/logo.png" alt="AgroVision AI Logo" width="180">
</p>

  🚀 <a href="https://plant-disease-detection-s.streamlit.app/">Live Demo</a>
</p>
<p align="center">
  <b>AI-Powered Plant Disease Detection using Deep Learning</b><br>
  Built with TensorFlow, Keras, and Streamlit
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---


## 📌 Overview

**AgroVision AI** is an AI-powered web application that detects plant diseases from leaf images using a Convolutional Neural Network (CNN). The application provides accurate disease predictions along with confidence scores, disease descriptions, symptoms, treatment recommendations, prevention measures, and a downloadable PDF diagnostic report.

This project is built using the **PlantVillage dataset** and demonstrates the practical application of Deep Learning in smart agriculture.

---

## ✨ Features

- 🌿 Detects **38 plant diseases**
- 🤖 TensorFlow CNN-based prediction
- 📊 Confidence score visualization
- 📖 Detailed disease description
- 🍂 Symptoms identification
- 💊 Treatment recommendations
- 🌱 Prevention methods
- 📈 Top-5 prediction confidence chart
- 📄 Downloadable PDF diagnostic report
- 🌙 Modern responsive Streamlit interface
- ⚡ Fast and lightweight deployment

---


## 🌱 Supported Crops

- 🍎 Apple
- 🫐 Blueberry
- 🍒 Cherry
- 🌽 Corn
- 🍇 Grape
- 🍊 Orange
- 🍑 Peach
- 🫑 Bell Pepper
- 🥔 Potato
- 🍓 Strawberry
- 🍅 Tomato
- 🌱 Soybean
- 🌿 Raspberry
- 🎃 Squash

---

## 🧠 Deep Learning Model

| Property | Details |
|----------|---------|
| Model | Convolutional Neural Network (CNN) |
| Framework | TensorFlow / Keras |
| Dataset | PlantVillage |
| Disease Classes | 38 |
| Input Image Size | 224 × 224 |
| Output | Disease Prediction + Confidence |

---

## 🛠️ Tech Stack

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pandas
- Plotly
- Pillow
- ReportLab

---

## 📂 Project Structure

```
AgroVision-AI/
│
├── app.py
├── disease_info.py
├── class_names.json
├── plant_disease_model.keras
├── style.css
├── requirements.txt
│
├── assets/
│   ├── logo.png
│   └── doctor.png
│
└── README.md
```

---

## 📊 How It Works

1. Upload a plant leaf image.
2. The image is resized to **224 × 224** pixels.
3. The CNN model processes the image.
4. The disease class is predicted.
5. The confidence score is calculated.
6. Disease information is retrieved.
7. The application displays:
   - Disease name
   - Confidence score
   - Description
   - Symptoms
   - Treatment
   - Prevention
8. A professional PDF report can be downloaded.

---

## 📄 PDF Report Includes

- Prediction Result
- Confidence Score
- Disease Description
- Symptoms
- Treatment
- Prevention
- Date & Time of Analysis

---

## 🎯 Future Improvements

- Grad-CAM visualization
- Disease severity estimation
- Multi-language support
- Farmer chatbot integration
- Weather-aware disease prediction
- Fertilizer recommendations
- Mobile application

---

## 👨‍💻 Developer

**Md Salman Ali**

B.Tech Computer Science & Engineering (Data Science)

Jamia Millia Islamia

LinkedIn: https://www.linkedin.com/in/md-salman-ali-8a301a324/

---

## ⭐ If You Like This Project

If you found this project helpful, consider giving it a ⭐ on GitHub.

---

## 📜 License

This project is licensed under the MIT License.
