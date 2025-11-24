# 🧠 Face Recognition System

This project is a complete face recognition pipeline that integrates:
- ✔ Face Identification  
- ✔ Age & Gender Estimation  
- ✔ Face Mask Detection  
- ✔ Anti-Spoofing Detection  

It ensures secure and accurate face recognition in real-time.

---

## 🎥 Demo Video
👉 **[Demo Video](https://drive.google.com/file/d/1GrZni8NgTZKUI0Arhmc_81rZM96HASZ7/view?usp=drive_link)**

---

# 📌 Features

## 1️⃣ Face Identity Recognition
Uses the **face_recognition** Python library for:
- Face detection  
- Face embedding  
- Identity matching  

---

## 2️⃣ Age & Gender Estimation
Powered by a custom CNN model.

### 🔍 Model Architecture
![Age/Gender Model Architecture](training%20Models/Age_Gen_estimate/model_architecture.jpg)

---

## 3️⃣ Face Mask Detection
Detects:
- Mask 😷  
- No Mask 🙂  

### 🔍 Model Architecture
![Mask Detection Architecture](training%20Models/MaskDetect/model_architecture.jpg)

---

## 4️⃣ Anti-Spoofing Detection
Prevents unauthorized access by detecting fake faces using liveness checks (e.g., printed photos, screens, or replay attacks).

---

## 🚀 Technologies
- Python  
- face_recognition (dlib)  
- TensorFlow / Keras  
- OpenCV  

---

