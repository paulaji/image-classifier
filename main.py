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
        # to recognize/read and load the file uploaded as an image into a variable image
        image = Image.open(file)
        # to actually display the image uploaded | second parameter is to correctly fit the image
        st.image(image, use_column_width=True)

        # now we have to resize the image so that the ML model can take it as an input
        # we specified the image input as 32x32 and 3 color channels in the model, therefore:
        resized_image = image.resize((32, 32))

        # refer info.txt for explanation
        # to normalise the resized image
        image_array = np.array(resized_image) / 255
        # now that we normalised and resized, we need the input to be in proper format, so,
        image_array = image_array.reshape((1, 32, 32, 3))  # 1 image 32x32 size 3 color channels

        # now that we have uploaded and reshaped our image for giving it as input to the model, let's load the model
        model = tf.keras.models.load_model('cifar10_model.h5')

        # using the in-built predict function of tf and keras to make predictions using the model
        predictions = model.predict(image_array)
        cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # using matplotlib to plot bar graphs showing the confidence of prediction in each class
        fig, ax = plt.subplots()  # fig for figure, ax for axes
        y_pos = np.arrange(len(cifar10_classes))
        # barh: horizontal bargraph
        ax.barh(y_pos, predictions[0], align="center")
    else:
        st.text("Please upload an image file.")


