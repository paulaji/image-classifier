import numpy as np  # to work with arrays
import matplotlib.pyplot as plt  # to plot
import streamlit as st  # to build UI
import tensorflow as tf
from PIL import Image  # to work with images

def main():
    st.title("CIFAR_10 Image Classifier")
    st.write("Upload an image and let the prediction happen!")

    file = st.file_uploader("upload an image", type=['jpg', 'png'])
    if file:
        image = Image.open(file)
    else:
        st.text("Please upload an image file.")


