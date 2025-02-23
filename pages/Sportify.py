import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import pickle
from pathlib import Path


# Load the model
def load_model():
    # Define the model architecture (must match the one used during training)
    model_path = Path('C:/Users/dhanr/Desktop/sports_classification/model.pth')
    model = torch.load(model_path, map_location=torch.device('cpu'),weights_only=False)
    model.eval() # Set the model to evaluation mode
    return model

# Preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Class labels (replace with your actual class labels)
with open("pretrain/class_labels.pkl", "rb") as f:
    class_labels = pickle.load(f)


# Sports Classification Page
def main():
    st.title("Sportify!")
    st.write("Upload an image of a sports activity, and the app will predict the sport.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image",  use_container_width=True)

        # Preprocess the image
        input_tensor = preprocess_image(image)

        # Load the model
        model = load_model()

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_class = torch.max(output, 1)
            predicted_index = predicted_class.item()
            predicted_label = next((key for key, value in class_labels.items() if value == predicted_index), "Unknown")

        # Display the prediction
        st.write(f"**Predicted Sport:** {predicted_label}")
        st.write("### Want to try another image?")
        st.write("Upload a new image or click the link in the sidebar to go back to the Home page.")
        st.page_link("Home.py", label="Go to Home Page", icon="🏈")

# Run the app
if __name__ == "__main__":
    main()