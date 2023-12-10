import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
# Load the pre-trained model
model = load_model('FedaratedModel.h5')
page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("https://cdn.wallpapersafari.com/51/42/hjZ3E5.gif");
        background-size: 100%;
        background-position: top left;
        background-attachment: local;
    }}
    [data-testid="stSidebar"] {{
        background-image: url("https://cdn.dribbble.com/users/189524/screenshots/2103470/01-black-cat_800x600_v1.gif");
        background-size: 470%;
        background-position: top left;
        background-attachment: local;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0, 0, 0, 0);
    }}
    [data-testid="stToolbar"] {{
        right: 2rem;
    }}
    </style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
def convert_and_denoise(input_image_path, threshold=128):
    try:
        grayscale_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
        _, black_white_image = cv2.threshold(grayscale_image, threshold, 255, cv2.THRESH_BINARY)
        inverted_image = cv2.bitwise_not(black_white_image)
        kernel = np.ones((3, 3), np.uint8)
        denoised_image = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, kernel)
        return denoised_image
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def predict(img):
    grayscale_image = Image.fromarray(img.astype(np.uint8))
    resized_image = grayscale_image.resize((28, 28))
    normalized_image = np.array(resized_image) / 255.0
    normalized_image = normalized_image.reshape(1, 28, 28, 1)
    predicted_class = np.argmax(model.predict(normalized_image))
    confidence_score = model.predict(normalized_image).max()
    return predicted_class, confidence_score

# Custom CSS styles (same as your original CSS)
st.markdown(
    """
    <style>
        .title {
            font-size: 36px;
            text-align: center;
            color: #FF5733;
        }
        .upload-btn-wrapper {
            position: relative;
            overflow: hidden;
            display: inline-block;
        }
        .btn {
            border: 2px solid gray;
            color: gray;
            background-color: white;
            padding: 8px 20px;
            border-radius: 8px;
            font-size: 20px;
            font-weight: bold;
        }
        .upload-btn-wrapper input[type=file] {
            font-size: 100px;
            position: absolute;
            left: 0;
            top: 0;
            opacity: 0;
        }
        .prediction {
            font-size: 24px;
            text-align: center;
            color: #28a745;
            margin-top: 20px;
        }
        .loader {
            border: 8px solid #f3f3f3;
            border-radius: 50%;
            border-top: 8px solid #3498db;
            width: 40px;
            height: 40px;
            animation: spin 2s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Streamlit UI
st.title("MNIST Digit Recognition")
uploaded_image = st.file_uploader("Choose a digit image (jpg, png, or jpeg)", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    col1, col2, col3 = st.columns(3)

# Column 1: Display uploaded image
    col1.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Loading spinner while processing
    with st.spinner('Predicting...'):
        # Save the uploaded image
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_image.read())

        # Perform denoising and prediction
        denoised_image = convert_and_denoise("temp_image.jpg")
        if denoised_image is not None:
            # Column 2: Display denoised image
            col2.image(denoised_image, caption="Denoised Image", use_column_width=True)

            # Predict the digit and confidence score
            predicted_class, confidence_score = predict(denoised_image)
            # Column 3: Display prediction with confidence score
            col3.markdown(f"<div class='prediction'>Predicted digit: {predicted_class}</div>", unsafe_allow_html=True)
            #col3.markdown(f"<div class='prediction'>Confidence: {confidence_score:.2f}</div>", unsafe_allow_html=True
try:
    os.remove("temp_image.jpg")
except:
    pass
