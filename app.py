import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cpu")  # For Streamlit Cloud

# Your original model architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__()
        layers = []
        in_features = input_size

        for hidden in hidden_layers:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            in_features = hidden

        layers.append(nn.Linear(in_features, output_size))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_generator():
    checkpoint = torch.load("checkpoint.pth", map_location=device)
    model = NeuralNetwork(
        input_size=checkpoint["input_size"],
        output_size=checkpoint["output_size"],
        hidden_layers=checkpoint["hidden_layers"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model

def generate_images(model, digit, num_images=5):
    z_dim = checkpoint["input_size"] - 10  # Infer z_dim from input size
    z = torch.randn(num_images, z_dim)
    labels = torch.full((num_images,), digit, dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(labels, num_classes=10).float()
    input_vec = torch.cat([z, one_hot], dim=1)

    with torch.no_grad():
        outputs = model(input_vec).view(-1, 28, 28)

    return outputs.numpy()

# Streamlit UI
st.title("🧠 MNIST Digit Generator")
st.write("Select a digit (0-9) to generate 5 handwritten-style images!")

digit = st.number_input("Choose a digit", min_value=0, max_value=9, step=1)

if st.button("Generate"):
    try:
        generator = load_generator()
        checkpoint = torch.load("checkpoint.pth", map_location=device)  # needed to infer z_dim
        images = generate_images(generator, digit)

        fig, axs = plt.subplots(1, 5, figsize=(15, 3))
        for i in range(5):
            axs[i].imshow(images[i], cmap="gray")
            axs[i].axis("off")

        st.pyplot(fig)
    except Exception as e:
        st.error(f"⚠️ Failed to generate images.\n\n{e}")
