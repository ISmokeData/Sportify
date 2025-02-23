# Sportify: AI That Knows Your Game

🚀 **Welcome to the Sports Classification Project!**  
This project uses **deep learning** to classify images of various sports activities. Whether it's a thrilling game of baseball, the intensity of arm wrestling, or the elegance of balance beam routines, this model can identify and categorize the sport in an image with remarkable accuracy.

---

## **Key Features**
- **Image Classification**: Predicts the sport from an uploaded image.
- **Deep Learning Model**: Built using **PyTorch** and a pre-trained **ResNet50** architecture.
- **User-Friendly Interface**: Deployed as a **Streamlit web app** for easy interaction.
- **Multi-Class Classification**: Supports classification of multiple sports (e.g., archery, baseball, hockey, etc.).

---

## **How It Works**
1. **Upload an Image**: Users can upload an image of a sports activity.
2. **Model Prediction**: The deep learning model processes the image and predicts the sport.
3. **Display Results**: The predicted sport is displayed along with the uploaded image.

---

## **Technologies Used**
- **Python**: Core programming language.
- **PyTorch**: For building and training the deep learning model.
- **Streamlit**: For creating the web interface.
- **Torchvision**: For image preprocessing and transformations.
- **Pillow**: For image handling.

---

## **Repository Structure**
```
sports_classification/
│
├── app.py                  # Main Streamlit app (home page)
├── pages/                  # Folder for additional pages
│   └── classification.py   # Sports classification page
├── model/                  # Model and related files
│   └── model_state_dict.pth  # Trained model's state dictionary
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
```

---

## **Getting Started**
### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/sports-classification.git
cd sports-classification
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Streamlit App**
```bash
streamlit run app.py
```

### **4. Open the App**
- Go to `http://localhost:8501` in your browser.
- Upload an image and see the prediction!

---

## **Contributing**
Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeatureName`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeatureName`).
5. Open a pull request.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**
- **PyTorch**: For providing the deep learning framework.
- **Streamlit**: For making it easy to build interactive web apps.
- **Torchvision**: For pre-trained models and image transformations.


