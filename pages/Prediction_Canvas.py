import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('FedaratedModel.h5')
page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("https://i.makeagif.com/media/1-13-2023/_3qu79.gif");
        background-size: 100%;
        background-position: top left;
        background-attachment: local;
    }}
    [data-testid="stSidebar"] {{
                background-image: url("https://cdn.dribbble.com/users/189524/screenshots/2103470/01-black-cat_800x600_v1.gif");
        background-size: 350%;
        background-position: top ;
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
st.title("Drawing to Image Saver")

# Function to save the drawn image
def save_drawing_as_image(drawing, image_path):
    cv2.imwrite(image_path, drawing)

# Create a canvas for drawing with the provided settings
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=20,
    stroke_color="White",
    background_color="Black",
    height=400,
    width=400,  # Set canvas width
    drawing_mode="freedraw",  # Allow free drawing
    key="canvas",
)

image_filename = "output_drawing.png"

if st.button("Submit"):
    # Get the drawing from the canvas
    drawing = np.array(canvas_result.image_data)

    if drawing is not None:
        # Save the drawn image
        save_drawing_as_image(drawing, image_filename)

        # Display the saved image with a styled caption
        image = Image.open('output_drawing.png')
        image = image.convert('L')
        image = image.resize((28, 28))
        image = np.array(image) / 255.0  # Normalize the pixel values
        image = image.reshape(1, 28, 28, 1)
        prediction = model.predict(image)

        st.subheader("Prediction")
        st.write(f'<p style="color: green; font-size: 36px;">{np.argmax(prediction)}</p>', unsafe_allow_html=True)
