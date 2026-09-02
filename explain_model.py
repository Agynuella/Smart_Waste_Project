import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget # <-- NEW IMPORT
from PIL import Image
import urllib.request
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = 'waste_classifier_mobilenet.pth'
NUM_CLASSES = 6
CLASSES = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash'] 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. DOWNLOAD A TEST IMAGE
# ==========================================
IMAGE_PATH = 'test_bottle.jpg'
if not os.path.exists(IMAGE_PATH):
    print("Downloading a test image of a plastic bottle...")
    urllib.request.urlretrieve(
        "https://images.unsplash.com/photo-1528323273322-d81458248d40?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60", 
        IMAGE_PATH
    )

# ==========================================
# 3. LOAD THE MODEL
# ==========================================
def load_model():
    print("Loading the trained AI brain...")
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() 
    return model

# ==========================================
# 4. RUN EXPLAINABLE AI (GRAD-CAM)
# ==========================================
def run_xai():
    model = load_model()
    
    # Prepare the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_pil = Image.open(IMAGE_PATH).convert('RGB')
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    img_vis = cv2.resize(np.array(img_pil), (224, 224))
    img_vis = np.float32(img_vis) / 255.0

    # 1. GET THE PREDICTION FIRST (NEW LOGIC)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        predicted_label = CLASSES[predicted_idx.item()]
        print(f"✅ AI Predicted: {predicted_label}")

    # 2. ATTACH GRAD-CAM
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # 3. PASS THE PREDICTION AS THE TARGET (THE FIX)
    targets = [ClassifierOutputTarget(predicted_idx.item())]
    
    print("Generating XAI Heatmap...")
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(img_vis, grayscale_cam, use_rgb=True)

    # Plot the results side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_vis)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(visualization)
    axes[1].set_title(f"XAI Heatmap ({predicted_label})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('xai_result.png')
    print("✅ Success! Open 'xai_result.png' in VS Code to see your AI's brain in action!")

if __name__ == '__main__':
    run_xai()