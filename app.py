import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
import cv2
from PIL import Image

# ---------------------------
# SETTINGS
# ---------------------------
CLASSES = ["no_tumor", "yes_tumor"]
MODEL_PATH = "resnet18_brain_tumor.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


model = load_model()


# ---------------------------
# IMAGE PREPROCESSING
# ---------------------------
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


# ---------------------------
# GRAD-CAM IMPLEMENTATION
# ---------------------------
def generate_gradcam(model, x):
    model.eval()

    # get final conv layer
    final_conv = model.layer4[1].conv2

    gradients = []
    activations = []

    def save_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activations(module, input, output):
        activations.append(output)

    final_conv.register_forward_hook(save_activations)
    final_conv.register_backward_hook(save_gradients)

    # forward
    logits = model(x)
    pred_class = logits.argmax(dim=1).item()

    # backward
    model.zero_grad()
    logits[0, pred_class].backward()

    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for w, a in zip(weights, act):
        cam += w * a

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / cam.max()

    heatmap = (cam * 255).astype(np.uint8)
    return heatmap


def overlay_heatmap(img, heatmap):
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)
    return overlay


# -----------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------
st.title("🧠 Brain Tumor Detection (ResNet18 + Grad-CAM)")
st.write("Upload an MRI scan to detect tumor presence and visualize model attention.")

uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img_pil = Image.open(uploaded).convert("RGB")

    st.subheader("Original Image")
    st.image(img_pil, width=300)

    # preprocess
    img_resized, x = preprocess(img_pil)

    # prediction
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = np.argmax(probs)
    label = CLASSES[pred_idx]
    conf = probs[pred_idx]

    # display prediction
    st.subheader("Prediction")
    st.write(f"**Tumor:** `{label}`")
    st.write(f"**Confidence:** `{conf:.4f}`")

    # generate gradcam
    heatmap = generate_gradcam(model, x)
    overlay = overlay_heatmap(img_resized, heatmap)

    # show heatmap
    st.subheader("Grad-CAM Heatmap")
    st.image(overlay, caption="Model Focus Area", width=300)
