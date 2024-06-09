import streamlit as st
import numpy as np
import cv2
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from PIL import Image


def gaussian_derivative_kernels(sigma, order=1):
    size = int(np.ceil(3 * sigma))
    x, y = np.meshgrid(np.arange(-size, size + 1), np.arange(-size, size + 1))

    gaussian = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    gaussian /= (2 * np.pi * sigma ** 2)

    if order == 1:
        gx = -x * gaussian / sigma ** 2
        gy = -y * gaussian / sigma ** 2
        return gx, gy
    else:
        raise NotImplementedError("Higher order derivatives not implemented.")


def apply_steerable_filters(image, sigma):
    gx, gy = gaussian_derivative_kernels(sigma)

    Ix = convolve(image, gx)
    Iy = convolve(image, gy)

    G0 = Ix
    G90 = Iy

    return G0, G90


def plot_results(image, G0, G90):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(np.abs(G0), cmap='gray')
    axes[1].set_title('G0 (0 degrees)')
    axes[1].axis('off')

    axes[2].imshow(np.abs(G90), cmap='gray')
    axes[2].set_title('G90 (90 degrees)')
    axes[2].axis('off')

    st.pyplot(fig)


# Streamlit app
st.title("Steerable Filters - Freeman and Adelson (1991)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = np.array(image)

    st.image(image, caption='Original Image', use_column_width=True)

    sigma = st.sidebar.slider("Sigma (scale of Gaussian)", 0.1, 10.0, 1.0, 0.1)

    G0, G90 = apply_steerable_filters(image, sigma)

    plot_results(image, G0, G90)
