# 🧠 Brain Cancer Classification

This project was chosen out of my passion for combining healthcare and artificial intelligence to create meaningful solutions. Brain cancer is a critical global health issue, and by building a neural network to classify tumor types from MRI images, this project showcases the power of machine learning to assist in early diagnosis and improve patient outcomes.

---

## 💡 Project Motivation

Brain tumors are among the most life-threatening forms of cancer, often requiring early and precise detection for effective treatment. However, interpreting MRI scans can be challenging, time-consuming, and subject to human error.  

This project was motivated by the desire to:

✅ Apply deep learning to real-world medical challenges  
✅ Support healthcare professionals with AI-powered tools  
✅ Contribute to open-source solutions in medical imaging  
✅ Deepen my own expertise in computer vision and healthcare AI  

By using machine learning to classify brain tumor types, I aim to explore how technology can play a transformative role in improving diagnostic accuracy and ultimately saving lives.

---

## 📊 Dataset

The training dataset comes from the following Kaggle project:

https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c

---

### 🖼️ Sample MRI Images

<table>
  <tr>
    <th style="text-align: center">Example MRI Scans</th>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/05240ef8-e105-4bc0-84e0-2d3e0ace636c" /></td>
  </tr>
</table>

---

### 📈 Class Distribution (90% Train Dataset)

<table>
  <tr>
    <th colspan="2" style="text-align: center">Class Distribution</th>
  </tr>
  <tr>
    <td style="width: 66%">
      <img src="https://github.com/user-attachments/assets/568dff8f-4624-40cd-a77c-44aa6f08d420" style="width: 100%"/>
    </td>
    <td style="width: 34%; vertical-align: top">
      <table>
        <tr><th>Class</th><th>Count</th></tr>
        <tr><td>Meningioma</td><td>874</td></tr>
        <tr><td>Astrocitoma</td><td>580</td></tr>
        <tr><td>Normal</td><td>522</td></tr>
        <tr><td>Schwannoma</td><td>465</td></tr>
        <tr><td>Neurocitoma</td><td>457</td></tr>
        <tr><td>Carcinoma</td><td>251</td></tr>
        <tr><td>Papiloma</td><td>237</td></tr>
        <tr><td>Oligodendroglioma</td><td>224</td></tr>
        <tr><td>Glioblastoma</td><td>204</td></tr>
        <tr><td>Ependimoma</td><td>150</td></tr>
        <tr><td>Tuberculoma</td><td>145</td></tr>
        <tr><td>Meduloblastoma</td><td>131</td></tr>
        <tr><td>Germinoma</td><td>100</td></tr>
        <tr><td>Granuloma</td><td>78</td></tr>
        <tr><td>Ganglioglioma</td><td>61</td></tr>
      </table>
    </td>
  </tr>
</table>

---

## ⚙️ Model Architecture

The model uses **ResNet50V2**, a deep convolutional neural network pre-trained on ImageNet, as the feature extractor.

- Last 7 layers are unfrozen for fine-tuning on MRI data.
- Output features are average pooled, flattened, and passed into a fully connected layer of size 15, matching the 15 brain cancer classes.

---

<table>
  <tr>
    <th style="text-align: center">Training Accuracy Over Epochs</th>
    <th style="text-align: center">Confusion Matrix on Validation Set</th>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/7ec8c9dc-3304-450c-82c7-19ab7c0f12f9" /></td>
    <td><img src="https://github.com/user-attachments/assets/f756acf9-22ae-43f1-99e3-2cecd5ae1610" /></td>
  </tr>
</table>

---

## 🚀 Project Highlights

✅ Advanced CNN with transfer learning  
✅ Fine-tuned on domain-specific MRI data  
✅ Strong class separation and high accuracy  
✅ Practical use case in healthcare AI

---

## 🔧 How to Run

1️⃣ Clone the repository:
```bash
git clone https://github.com/dubboy000/brain_cancer_classifier.git
cd brain_cancer_classifier
```

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

3️⃣ Start training:
```bash
python train.py
```

4️⃣ Evaluate results:
```bash
python evaluate.py
```

---

## 🌟 Future Work

- Incorporate data augmentation to expand variability  
- Test alternative architectures like EfficientNet or DenseNet  
- Implement explainability tools (e.g., Grad-CAM)  
- Deploy as a web-based diagnostic tool for clinics and researchers

---
