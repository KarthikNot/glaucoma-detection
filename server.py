import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import streamlit as st
from PIL import Image
import numpy as np
from collections import OrderedDict

data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class GlaucomaCNN(nn.Module):
    def __init__(self):
        super(GlaucomaCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GlaucomaCNN()
print(device)
checkpoint = torch.load("./models/glaucoma_model_90TA.pth", map_location=device)

new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in checkpoint.items())

model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

st.set_page_config(page_title="Glaucoma Detection", page_icon="🩺", layout="centered")

st.title("🔍 Glaucoma Detection System")
st.markdown(
    "Upload an eye image to check for glaucoma. The model will predict whether **glaucoma** is present or not."
)

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", width=500)

    image_tensor = data_transforms(image).unsqueeze(0).to(device)

    if st.button("🔎 Analyze"):
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)

        class_names = ["🟢 No Glaucoma", "🔴 Glaucoma Detected"]
        result = class_names[predicted.item()]

        if predicted.item() == 0:
            st.success(f"**Prediction:** {result}")
        else:
            st.error(f"**Prediction:** {result}")
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        st.write(f"🟢 No Glaucoma: {probabilities[0]:.2%}")
        st.write(f"🔴 Glaucoma Detected: {probabilities[1]:.2%}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: rgba(255, 255, 255, 0.7);'>🚀 Developed with PyTorch & Streamlit</div>", unsafe_allow_html=True)
