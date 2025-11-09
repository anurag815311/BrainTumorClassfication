

---

````markdown
# 🧠 Brain Tumor Classification (MRI) – ResNet18 + Grad-CAM

This project focuses on classifying MRI brain images into two classes:
- **Tumor**
- **No Tumor**

It uses transfer learning with **ResNet18**, along with basic image preprocessing and augmentation.  
To make the model explainable, **Grad-CAM** is integrated to visualize which parts of the MRI the model focuses on during prediction.

---

## 📂 Dataset
The dataset contains **MRI images**:
- images with tumor  
- images without tumor  

All images were resized to **224×224** and normalized before training.

---

## ⚙️ Model
The model is based on **ResNet18** pretrained on **ImageNet**.  
Only the final layer was modified to predict two classes.  
It was trained for a few epochs on CPU and achieved around **94% test accuracy**.

---

## 🚀 Features
- Simple training pipeline using **PyTorch**
- **Single-image prediction script**
- **Grad-CAM** visualization for model explainability
- **Streamlit** web app for interactive predictions

---

## 🧩 How to Run

### 1. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On macOS/Linux
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run prediction on an image

```bash
python predict.py --image path_to_image.jpg
```

### 4. Launch Streamlit app

```bash
streamlit run app.py
```

The web app will open in your browser.
You can upload an MRI scan to view the model’s prediction and corresponding **Grad-CAM heatmap**.

---

## 📁 Project Structure

```
├── app.py                     # Streamlit web app
├── predict.py                 # CLI prediction script
├── resnet18_brain_tumor.pth   # Trained model weights
├── brain_tumor_dataset/       # Dataset (train/test images)
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

---

## 🔍 Grad-CAM

Grad-CAM is used to visualize **where the model is focusing** in the MRI image.
It helps interpret predictions and verify whether the network attends to the tumor region.

---

## ⚠️ Notes

* This project is for **educational and research purposes only**.
* It should **not** be used for actual medical diagnosis.

---

## 👤 Author

**Anurag Kumar Singh**


