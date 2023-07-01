import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

def process_image(image_file):
   
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    logo_sizes = np.array([image_rgb.shape[:2]])

    plt.figure(figsize=(8, 6))
    plt.scatter(logo_sizes[:, 1], logo_sizes[:, 0])
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Distribution of Logo Sizes')
    plt.tight_layout()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    height, width, _ = image_rgb.shape
    aspect_ratio = width / height

    plt.figure(figsize=(8, 6))
    plt.hist([aspect_ratio], bins=30, edgecolor='black')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Frequency')
    plt.title('Distribution of Logo Aspect Ratios')
    plt.tight_layout()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    color_mean = np.mean(image_rgb, axis=(0, 1))

    plt.figure(figsize=(8, 6))
    plt.scatter(color_mean[0], color_mean[1], c=color_mean[2] / 255.0, cmap='viridis')
    plt.xlabel('Red')
    plt.ylabel('Green')
    plt.title('Distribution of Logo Colors')
    plt.colorbar(label='Blue')
    plt.tight_layout()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

def run():
    st.title("Logo Detection")
    st.subheader('Exploratory Data Analysis (EDA)')
    st.write('Created By: **Mangara Haposan Immanuel Siagian**')
    image = Image.open('deployment/LD.png')
    st.image(image, caption=' ')
    
    st.markdown('---')
    
    st.write('### Definition')
    st.write('Logo detection is a computer vision and image processing technology that focuses on identifying and localizing instances of logos in digital images and videos.')
    st.write('[Reference](https://en.wikipedia.org/wiki/Object_detection)')
    
    st.markdown('---')

    st.write('Enter an image file to see the data')
    image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        process_image(image_file)

if __name__ == "__main__":
    run()