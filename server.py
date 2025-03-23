import torch
import torch.nn as nn
import torchvision.transforms as transforms
import streamlit as st
from PIL import Image
import numpy as np
from collections import OrderedDict
from spikingjelly.clock_driven import neuron, functional, surrogate

@st.cache_resource
def load_model():
    model = GlaucomaSNN()
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image).unsqueeze(0)

class GlaucomaSNN(nn.Module):
    def __init__(self, T=4):
        super(GlaucomaSNN, self).__init__()
        self.T = T  # Number of time steps

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),  # Spiking activation
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            nn.AdaptiveAvgPool2d((8, 8)),  # Adaptive pooling
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        mem = 0
        for t in range(self.T):  # Iterate over time steps
            out = self.conv_layers(x)
            out = self.fc_layers(out)
            mem += out  # Accumulate membrane potential
        
        return mem / self.T  # Average across time steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GlaucomaSNN()
print(device)
state_dict = torch.load('./models/glaucoma_model_SNN_86TA.pth')

new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

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

    image_tensor = preprocess_image(image)

    if st.button("🔎 Analyze"):
        with torch.no_grad():
            image_tensor = image_tensor.to(device)  
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
