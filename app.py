
import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Generator class definition must match the training script
class Generator(torch.nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.gen = torch.nn.Sequential(
            torch.nn.Linear(z_dim + 10, 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(True),
            torch.nn.Linear(1024, 28*28),
            torch.nn.Tanh()
        )

    def forward(self, z, labels):
        one_hot = F.one_hot(labels, num_classes=10).float()
        x = torch.cat([z, one_hot], dim=1)
        return self.gen(x).view(-1, 1, 28, 28)

# App title and setup
st.set_page_config(page_title="Digit Generator", layout="centered")
st.title("🖊️ Handwritten Digit Generator")
st.markdown("Select a digit (0–9) and generate 5 synthetic handwritten images using a GAN trained on MNIST.")

# Load model
@st.cache_resource
def load_generator():
    model = Generator(z_dim=100)
    model.load_state_dict(torch.load("checkpoint.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

generator = load_generator()

# User input
digit = st.number_input("Choose a digit to generate (0–9):", min_value=0, max_value=9, step=1)

if st.button("✨ Generate Images"):
    with st.spinner("Generating images..."):
        torch.manual_seed(42)
        z = torch.randn(5, 100)
        labels = torch.tensor([digit] * 5)
        with torch.no_grad():
            images = generator(z, labels)

        # Prepare plot
        fig, axes = plt.subplots(1, 5, figsize=(12, 3))
        for i, ax in enumerate(axes):
            ax.imshow(images[i].squeeze(), cmap="gray_r")
            ax.axis("off")
        st.pyplot(fig)
        st.success("Done! These are synthetic samples of the digit {}.".format(digit))
