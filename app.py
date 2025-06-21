import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100

# Define the generator model (same architecture as in training)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.ReLU(True)
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.shape[0], 128, 7, 7)
        img = self.upsample(out)
        return img

# Load generator model from saved checkpoint
@st.cache_resource
def load_generator():
    model = Generator().to(device)
    model.load_state_dict(torch.load("generator_model.pth", map_location=device))
    model.eval()
    return model

# Generate images using the generator
def generate_images(generator, num_images):
    z = torch.randn(num_images, latent_dim).to(device)
    with torch.no_grad():
        gen_imgs = generator(z)
    gen_imgs = gen_imgs.cpu().numpy()
    images = [gen_imgs[i][0] for i in range(num_images)]  # Extract channel 0
    return images

# Streamlit UI
st.title("MNIST Digit Generator")
st.markdown("Generate handwritten digit images using a trained with Thoky generator.")

num_images = st.slider("Number of images", 1, 10, 5)

if st.button("Generate"):
    try:
        generator = load_generator()
        with st.spinner("Generating images..."):
            images = generate_images(generator, num_images)

        st.success("Generated Images")
        cols = st.columns(num_images)
        for i, col in enumerate(cols):
            col.image(images[i], width=100, clamp=True, channels="GRAY")
    except Exception as e:
        st.error("\u26A0\uFE0F Failed to generate images.")
        st.exception(e)
