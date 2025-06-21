import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Config
z_dim = 100
device = torch.device("cpu")

# Generator Model
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim + 10, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        one_hot = F.one_hot(labels, num_classes=10).float()
        x = torch.cat([z, one_hot], dim=1)
        return self.gen(x).view(-1, 1, 28, 28)

@st.cache_resource
def load_generator():
    model = Generator(z_dim).to(device)
    model.load_state_dict(torch.load("checkpoint.pth", map_location=device))
    model.eval()
    return model

generator = load_generator()

st.title("MNIST Digit Generator")
digit = st.selectbox("Choose a digit (0–9):", list(range(10)))

if st.button("Generate Samples"):
    labels = torch.full((5,), digit, dtype=torch.long).to(device)
    z = torch.randn(5, z_dim).to(device)
    with torch.no_grad():
        samples = generator(z, labels).cpu().squeeze()

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axes[i].imshow(samples[i], cmap="gray")
        axes[i].axis("off")
    st.pyplot(fig)
