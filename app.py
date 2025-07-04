import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model settings
latent_dim = 100
img_shape = (1, 28, 28)
num_classes = 10

# Generator definition (must match training script exactly)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.GELU(),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((noise, self.label_emb(labels)), dim=1)
        out = self.model(gen_input)
        return out.view(out.size(0), *img_shape)

# Load the trained generator
@st.cache_resource
def load_generator():
    model = Generator().to(device)
    model.load_state_dict(torch.load("generator2.pth", map_location=device))
    model.eval()
    return model

# Image generation function
def generate_images(generator, digit, num_images):
    noise = torch.randn(num_images, latent_dim).to(device)
    labels = torch.full((num_images,), digit, dtype=torch.long).to(device)
    with torch.no_grad():
        generated = generator(noise, labels)
    return generated.cpu().numpy()

# App UI
st.title("Tahmeed Thoky Conditional GAN - MNIST Digit Generator")
st.markdown("Generate MNIST-style handwritten digits using a trained CGAN model.")

# User input section
st.subheader("Choose a Digit to Generate")
use_manual_input = st.checkbox("Enter digit manually", value=False)

if use_manual_input:
    digit = st.number_input("Enter digit (0–9):", min_value=0, max_value=9, step=1)
else:
    digit = st.selectbox("Select digit", list(range(10)))

num_images = st.slider("Number of images to generate", 1, 10, 5)

# Generate button
if st.button("Generate"):
    try:
        gen = load_generator()
        images = generate_images(gen, int(digit), num_images)

        st.success(f"Generated {num_images} image(s) for digit '{digit}'")
        cols = st.columns(num_images)
        for i in range(num_images):
            img = (images[i].squeeze() + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
            cols[i].image(img, width=100, clamp=True, channels="GRAY")

    except Exception as e:
        st.error("❌ Failed to generate images.")
        st.exception(e)
