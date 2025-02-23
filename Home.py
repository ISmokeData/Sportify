import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import pickle
from pathlib import Path


# Home Page
def main():
    st.title("Sportify: AI That Knows Your Game")
    st.image("gameee.jpg",  use_container_width=True)
    st.write("""
    This app uses a deep learning model to classify images of sports activities. 
    Whether it's a thrilling game of baseball, the intensity of arm wrestling, or the elegance of balance beam routines, 
    our model can identify and categorize the sport in an image with remarkable accuracy.
    """)
    st.write("### How It Works:")
    st.write("1. **Upload an Image**: Go to the **Sports Classification** page and upload an image of a sports activity.")
    st.write("2. **Get Predictions**: The app will analyze the image and predict the sport.")
    st.write("3. **Explore Results**: See the predicted sport and learn more about it!")
    st.write("### Ready to Try It Out?")
    st.write("Navigate to the **Sports Classification** page using the sidebar or the link below.")

    # Link to the classification page
    st.page_link("pages/Sportify.py", label="Go to Sports Classification Page", icon="🏈")

# Run the app
if __name__ == "__main__":
    main()
