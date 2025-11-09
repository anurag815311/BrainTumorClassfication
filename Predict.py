import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
import numpy as np
import os

# ---------------------------
# 1. Classes
# ---------------------------
CLASSES = ["no_tumor", "yes_tumor"]


# ---------------------------
# 2. Image Preprocessing
# ---------------------------
def preprocess_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"cv2 cannot read image: {path}")

    # handle RGBA
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    tensor = transform(img_resized).unsqueeze(0)
    return img_resized, tensor


# ---------------------------
# 3. Load Model
# ---------------------------
def load_model(model_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ---------------------------
# 4. Predict
# ---------------------------
def predict(model, tensor, device):
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)

    return CLASSES[pred_idx], float(probs[pred_idx]), probs


# ---------------------------
# 5. Main function
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Tumor Prediction Script")
    parser.add_argument("--image", type=str, required=True, help="Path to MRI image")
    parser.add_argument("--model", type=str, default="resnet18_brain_tumor.pth",
                        help="Path to trained model (.pth file)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nLoading model...")
    model = load_model(args.model, device)
    print("Model loaded successfully.")

    print("\nPreprocessing image...")
    img, tensor = preprocess_image(args.image)

    print("Running prediction...\n")
    label, conf, all_probs = predict(model, tensor, device)

    print("✅ Prediction:", label)
    print(f"✅ Confidence: {conf:.4f}")
    print("\nProbabilities (no_tumor, yes_tumor):", all_probs)
    print("\nImage tested:", args.image)
