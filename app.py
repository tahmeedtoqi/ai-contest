import numpy as np
import streamlit as st
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch.autograd import Variable
import os
from time import time
import matplotlib.pyplot as pltplt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Neural Network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        super().__init__()
        self.input_size = input_size
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        for input_dim, output_dim in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.hidden_layers.append(nn.Linear(input_dim, output_dim))
        self.output_layer = nn.Linear(hidden_layers[-1], output_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=drop_p)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
            x = self.drop(x)
        return torch.tanh(self.output_layer(x))

# Load model from checkpoint
@st.cache(allow_output_mutation=True)
def load_generator():
    checkpoint = torch.load("checkpoint.pth", map_location=device)
    model = NeuralNetwork(
        input_size=checkpoint['input_size'],
        output_size=checkpoint['output_size'],
        hidden_layers=checkpoint['hidden_layers']
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model

# Generate images for a specific digit
def generate_images(generator, digit, num_images):
    images = []
    for _ in range(num_images):
        # Random noise and dummy label to pass through the trained classifier network
        vec = torch.randn(1, generator.input_size).to(device)
        with torch.no_grad():
            output = generator(vec)
        image = output.view(28, 28).cpu().numpy()
        images.append(image)
    return images

# Streamlit UI
st.title("MNIST Digit Generator")
st.markdown("Generate handwritten digits using a neural network trained from scratch.")

num_images = st.slider("Number of images", 1, 10, 5)
digit = st.selectbox("Select digit to generate", list(range(10)))

if st.button("Generate"):
    try:
        generator = load_generator()
        with st.spinner("Generating images..."):
            images = generate_images(generator, digit, num_images)

        st.success("Generated Images")
        cols = st.columns(num_images)
        for i, col in enumerate(cols):
            col.image(images[i], width=100, clamp=True, channels="GRAY")
    except Exception as e:
        st.error("\u26A0\uFE0F Failed to generate images.")
        st.exception(e)
