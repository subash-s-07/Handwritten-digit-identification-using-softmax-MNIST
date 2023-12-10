import plotly.express as px
import pandas as pd
from sklearn.datasets import fetch_openml
import streamlit as st
from sklearn.decomposition import PCA
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go


import warnings
warnings.filterwarnings('ignore')
page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("https://archive.org/download/wp2757875-wallpaper-gif/wp2757875-wallpaper-gif.gif");
        background-size: 100%;
        background-position: top left;
        background-attachment: local;
    }}
    [data-testid="stSidebar"] {{
        background-image: url("https://cdn.dribbble.com/users/189524/screenshots/2103470/01-black-cat_800x600_v1.gif");
        background-size: 350%;
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
# Load the MNIST dataset
mnist = fetch_openml('mnist_784')
X, y = mnist.data, mnist.target.astype(int)

# Create a DataFrame for the data
mnist_df = pd.DataFrame(X, columns=[f'Pixel_{i}' for i in range(X.shape[1])])
mnist_df['Label'] = y

def p1():
    st.title("MNIST Digit Distribution")

    # Display the MNIST digit distribution chart
    digit_counts = mnist_df['Label'].value_counts().sort_index()
    fig = px.bar(x=digit_counts.index, y=digit_counts.values, title='Digit Distribution in MNIST',labels={'x': 'Digit', 'y': 'Count'})
    
    st.plotly_chart(fig)

def p2():
    st.title("Pixel Intensity Changes Across Row 0")

    # Assuming X is a DataFrame where each row represents pixel intensity values for an image
    row = X.iloc[0].values
    fig = px.line(x=list(range(len(row))), y=row, title='Pixel Intensity Changes Across Row 0', labels={'x': 'Pixel Index', 'y': 'Pixel Intensity'})

    st.plotly_chart(fig)
def p3():
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    mnist_df = pd.DataFrame({'PC1': X_2d[:, 0], 'PC2': X_2d[:, 1], 'Label': y})
    st.title("MNIST 2D Scatter Plot with PCA")
    fig = px.scatter(mnist_df, x='PC1', y='PC2', color='Label',title='MNIST 2D Scatter Plot with PCA',labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'})

    st.plotly_chart(fig)
def p4():
    st.title("MNIST Image Heatmap")

    image_index = 0
    image_data = X.iloc[image_index].values.reshape(28, 28)
    
    # Scale the pixel values to the 0-255 range
    image_data *= 255

    # Create an interactive heatmap
    fig = px.imshow(image_data,
                   title=f'Heatmap of MNIST Image {image_index}',
                   labels={'x': 'Column', 'y': 'Row', 'color': 'Pixel Intensity'},
                   color_continuous_scale='Viridis')
    st.plotly_chart(fig)
def p5():
    st.title("Histogram of Pixel Intensities")

    image_index = st.number_input("Enter the image index (0 to 59999)", 0, len(X) - 1, 0)
    pixel_intensities = X.iloc[image_index]

    # Create a histogram of pixel intensities
    fig = px.histogram(pixel_intensities, nbins=20, title=f'Histogram of Pixel Intensities for Image {image_index}',
                       labels={'x': 'Pixel Intensity', 'y': 'Count'})
    
    st.plotly_chart(fig)
def p6():
    st.title("2D Density Contour Plot of Pixel Intensities")

    image_index = 0
    pixel_intensities = X.iloc[image_index]

    # Create a 2D density contour plot of pixel intensities
    fig = px.density_contour(x=pixel_intensities, y=pixel_intensities, title=f'2D Density Contour Plot of Pixel Intensities for Image {image_index}',
                            labels={'x': 'Column', 'y': 'Row'})
    
    st.plotly_chart(fig)

def p7():

    st.title("3D Surface Plot of Image Intensity")

    image_index = 0
    image_data = X.iloc[image_index].values.reshape(28, 28)

    x, y = np.meshgrid(range(28), range(28))
    fig = go.Figure(data=[go.Surface(z=image_data, colorscale='Viridis')])
    fig.update_layout(title=f'3D Surface Plot with Image Intensity (MNIST Image {image_index})', scene=dict(zaxis_title='Intensity'))

    st.plotly_chart(fig)
def p6():
    st.title("MNIST Digit Label Distribution")

    # Count the occurrences of each digit label
    digit_counts = mnist_df['Label'].value_counts().reset_index()
    digit_counts.columns = ['Digit', 'Count']

    # Create a treemap visualization
    fig = px.treemap(digit_counts, path=['Digit'], values='Count', title='MNIST Digit Label Distribution')

    st.plotly_chart(fig)


p1()
p2()
p3()
p4()
p5()
p6()
p7()

