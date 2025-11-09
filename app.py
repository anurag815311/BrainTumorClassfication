import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
import cv2
from PIL import Image


# App Styling

st.set_page_config(
    page_title="Brain Tumor Detector",
    layout="centered",
)

st.markdown("""
    <style>
        .main {background-color: #f5f7fa;}
        .stButton>button {
            border-radius: 10px;
            height: 3rem;
            width: 12rem;
            background-color: #4F46E5;
            color: white;
            font-size: 16px;
        }
        .title-text {
            text-align: center;
            font-size: 32px;
            font-weight: 700;
            color: #111827;
        }
        .sub-text {
            text-align: center;
            font-size: 18px;
            color: #4B5563;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<p class='title-text'>🧠 Brain Tumor Detection</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Upload an MRI scan to detect tumor and view the Grad-CAM heatmap.</p>", unsafe_allow_html=True)



# Model Setup

CLASSES = ["no_tumor", "yes_tumor"]
MODEL_PATH = "resnet18_brain_tumor.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()



# Preprocess

def preprocess(img_pil):
    img = np.array(img_pil)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    x = transform(img_resized).unsqueeze(0).to(device)
    return img_resized, x



# Grad-CAM

def generate_gradcam(model, x):
    model.eval()

    final_conv = model.layer4[1].conv2
    gradients = []
    activations = []

    def save_grad(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activation(module, input, output):
        activations.append(output)

    final_conv.register_forward_hook(save_activation)
    final_conv.register_backward_hook(save_grad)

    # forward
    logits = model(x)
    pred_class = logits.argmax().item()

    # backward
    model.zero_grad()
    logits[0, pred_class].backward()

    # ✅ FIX — detach tensors before numpy()
    grad = gradients[0].detach().cpu().numpy()[0]
    act = activations[0].detach().cpu().numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for w, a in zip(weights, act):
        cam += w * a

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    heatmap = np.uint8(cam * 255)
    return heatmap



def overlay_heatmap(img, heatmap):
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)
    return overlay


#UI for Streamlit

uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")

    st.markdown("""
        <h2 style="text-align:center; font-weight:700; color:#1e293b; margin-top:10px;">
            Brain Tumor Analysis
        </h2>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Image", "Grad-CAM", "Model Info"])

    with tab1:
        st.markdown("#### Uploaded Image")
        st.image(img_pil, use_container_width=True)

    img_resized, x = preprocess(img_pil)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = np.argmax(probs)
    label = CLASSES[pred_idx].upper()
    conf = float(probs[pred_idx])
    conf_percent = conf * 100

    st.markdown(f"""
        <div style="
            background: rgba(240, 253, 244, 0.9);
            padding: 18px;
            border-radius: 14px;
            margin-top: 20px;
            border-left: 10px solid #22c55e;">
            <h3 style="margin:0; color:#14532d;">Prediction: {label}</h3>
            <p style="margin:6px 0 0; font-size:18px;">
                Confidence: <b>{conf_percent:.2f}%</b>
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.progress(conf)

    with tab2:
        st.markdown("#### Grad-CAM Heatmap")
        heatmap = generate_gradcam(model, x)
        overlay = overlay_heatmap(img_resized, heatmap)
        st.image(overlay, use_container_width=True)

    with tab3:
        st.markdown("""
            #### Model Details
            • Architecture: ResNet18  
            • Transfer learning with custom final layer  
            • Input size: 224×224  
            • Outputs: Tumor / No Tumor  
            • Grad-CAM used for visualization  

            This model is built for learning and demonstration purposes.
        """)
