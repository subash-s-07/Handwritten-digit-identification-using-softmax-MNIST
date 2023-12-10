import streamlit as st
import pandas as pd
import plotly.graph_objs as go
st.set_page_config(layout="wide")
# Streamlit app
import streamlit as st
page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("https://mir-s3-cdn-cf.behance.net/project_modules/fs/da9c16117113967.607068e32a564.gif");
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
st.title("Digit Recognition using MNIST Dataset")
# Set page to wide layout

st.markdown(
    """
    <style>
        .hi{
            color:black;
            }
        .reportview-container {
            background: linear-gradient(to right, #00FFFF, #FF4500);
            color: #FFFFFF;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(to right, #00FFFF, #FF4500);
            color: #FFFFFF;
        }
        .dataframe {
            background: linear-gradient(to right, #00FFFF, #FF4500);
            color: #FFFFFF;
            font-size: 18px;
        }
        .st-cy {
            color: #00FFFF;
        }
        .st-dz {
            color: #FF4500;
        }
        .custom-container {
            background: linear-gradient(to right, #00FFFF, #FF4500);
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 34px;
        }
        .description-container {
            background: linear-gradient(to right, #FFDF00,  #FFC300);
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 34px;
        }
        p{
        font-size: 29px;
        color=black
        }
        h1{
        color:#FFDF00
        }
        strong{
        color:#F62817
        }
    </style>
    """
    , unsafe_allow_html=True
)
# Description of MNIST dataset inside the container with ice and fire gradient theme
with st.container():
    st.markdown(
        """
        <div class="description-container">
        <h1 class="hi">About MNIST Dataset</h1>
        <p>
        The MNIST dataset is a large database of handwritten digits that is widely used for training and testing machine 
        learning models. It consists of 28x28 pixel grayscale images of handwritten digits (0 to 9) along with their labels. 
        This dataset is often used as a benchmark to evaluate various machine learning algorithms.</p></div>
        """
    ,unsafe_allow_html=True)

st.markdown("""<div class="custom-container">
    <h1>Softmax Function</h1>
    <p>The softmax function is a mathematical operation that converts a vector of real numbers into a probability distribution. It takes an input vector and squashes the values between 0 and 1, ensuring they sum up to 1. Softmax is often used in machine learning for multi-class classification problems. It assigns probabilities to different classes, making it useful for determining the most likely class prediction. Softmax is widely used in the output layer of neural networks to compute probabilities for multiple classes, enabling the selection of the class with the highest probability as the final prediction.</p>
</div>""",unsafe_allow_html=True)
st.markdown("""<div class="custom-container">
    <h1>Federated Learning Process</h1>
    <p><strong>Data Loading:</strong> The code loads the MNIST dataset containing handwritten digits.</p>
    <p><strong>Data Preprocessing:</strong> It reshapes and normalizes the input images, preparing them for model training.</p>
    <p><strong>Data Augmentation:</strong> An ImageDataGenerator is used for data augmentation, creating variations of training images to enhance the model's robustness.</p>
    <p><strong>Client Models Creation:</strong> Individual client models are created with different learning rates using grid search for hyperparameter tuning.</p>
    <p><strong>Federated Learning Setup:</strong> Five client models are instantiated with random subsets of training data, simulating a decentralized training scenario.</p>
    <p><strong>Training:</strong> Each client model is trained locally using its respective subset of data for a single epoch.</p>
    <p><strong>Model Aggregation:</strong> The weights of the client models are averaged to create a global model, aggregating knowledge from all clients.</p>
    <p><strong>Evaluation:</strong> The performance of each client model and the global model is evaluated on the test dataset.</p>
    <p><strong>Result Output:</strong> The evaluation results for each client model on the test data are printed.</p>
    <p><strong>Model Saving:</strong> The trained global model is saved as "FederatedModel.h5" for future use or deployment.</p>
</div>""",unsafe_allow_html=True)
st.markdown("""<div>
    <h1>Classification Report and Details</h1></div>""",unsafe_allow_html=True)
tabs=st.tabs(['Global','Client_1','Client_2','Client_3','Client_4','Client_5'])
with tabs[0]:
    report_data = {
    "Class": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Precision": [0.93, 0.96, 0.93, 0.90, 0.92, 0.90, 0.94, 0.93, 0.81, 0.88],
    "Recall": [0.98, 0.97, 0.86, 0.89, 0.91, 0.84, 0.94, 0.90, 0.89, 0.90],
    "F1-Score": [0.95, 0.97, 0.89, 0.90, 0.91, 0.87, 0.94, 0.92, 0.85, 0.89],
    "Support": [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
}

    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df.style.background_gradient(cmap="RdBu", subset=["Class","Precision", "Recall", "F1-Score","Support"]))
with tabs[1]:
    report_data = {
    "Class": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Precision": [0.93, 0.97, 0.92, 0.91, 0.91, 0.92, 0.93, 0.87, 0.79, 0.91],
    "Recall": [0.97, 0.96, 0.84, 0.88, 0.91, 0.82, 0.96, 0.89, 0.86, 0.85],
    "F1-Score": [0.95, 0.96, 0.88, 0.89, 0.91, 0.86, 0.93, 0.91, 0.84, 0.87],
    "Support": [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
}


    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df.style.background_gradient(cmap="RdBu", subset=["Class","Precision", "Recall", "F1-Score","Support"]))
with tabs[2]:
    report_data_model2 = {
    "Class": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Precision": [0.96, 0.96, 0.95, 0.89, 0.94, 0.78, 0.95, 0.95, 0.79, 0.83],
    "Recall": [0.97, 0.95, 0.84, 0.88, 0.86, 0.90, 0.89, 0.88, 0.86, 0.92],
    "F1-Score": [0.96, 0.96, 0.89, 0.89, 0.90, 0.83, 0.92, 0.91, 0.82, 0.87],
    "Support": [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
}


    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df.style.background_gradient(cmap="RdBu", subset=["Class","Precision", "Recall", "F1-Score","Support"]))
with tabs[3]:
    report_data_model3 = {
    "Class": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Precision": [0.92, 0.97, 0.92, 0.93, 0.94, 0.92, 0.92, 0.94, 0.70, 0.83],
    "Recall": [0.97, 0.97, 0.84, 0.84, 0.87, 0.75, 0.93, 0.87, 0.94, 0.92],
    "F1-Score": [0.94, 0.97, 0.88, 0.88, 0.91, 0.83, 0.93, 0.90, 0.80, 0.87],
    "Support": [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
}

    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df.style.background_gradient(cmap="RdBu", subset=["Class","Precision", "Recall", "F1-Score","Support"]))
with tabs[4]:
    report_data_model4 = {
    "Class": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Precision": [0.89, 0.95, 0.92, 0.91, 0.91, 0.92, 0.93, 0.87, 0.79, 0.91],
    "Recall": [0.98, 0.97, 0.85, 0.88, 0.91, 0.80, 0.94, 0.92, 0.90, 0.84],
    "F1-Score": [0.94, 0.96, 0.89, 0.90, 0.91, 0.85, 0.93, 0.89, 0.84, 0.87],
    "Support": [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
}

    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df.style.background_gradient(cmap="RdBu", subset=["Class","Precision", "Recall", "F1-Score","Support"]))
with tabs[5]:
    report_data_model5 = {
    "Class": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Precision": [0.93, 0.95, 0.92, 0.91, 0.91, 0.92, 0.93, 0.87, 0.79, 0.91],
    "Recall": [0.98, 0.97, 0.85, 0.88, 0.91, 0.80, 0.94, 0.92, 0.90, 0.84],
    "F1-Score": [0.95, 0.96, 0.88, 0.89, 0.91, 0.85, 0.93, 0.89, 0.84, 0.87],
    "Support": [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
}

metrics = ['accuracy', 'precision', 'recall', 'f1-score']
client_metrics = {
    'Client 1': [0.90, 0.93, 0.97, 0.95],
    'Client 2': [0.90, 0.96, 0.95, 0.96],
    'Client 3': [0.89, 0.90, 0.92, 0.89],
    'Client 4': [0.90, 0.91, 0.92, 0.90],
    'Client 5': [0.90, 0.93, 0.97, 0.95]
}


# Find the maximum value in the client_metrics
max_value = max(max(max(client_data) for client_data in client_metrics.values()), 1.0)

# Create a Plotly polar plot
fig = go.Figure()

# Set the gap between clients
total_clients = len(client_metrics)
dtheta = 360 / total_clients  # This divides the circle equally among the clients
theta_shift = 0

for client_name, client_data in client_metrics.items():
    fig.add_trace(go.Scatterpolar(r=client_data, theta=metrics, fill='toself', name=client_name))
    theta_shift += dtheta  # Increase the theta shift for the next client to create a gap

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, max_value], tickvals=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])),
    showlegend=True,
    title='Model Performance Metrics Comparison'
)

# Streamlit app
st.title('Model Performance Metrics Comparison')
st.plotly_chart(fig)

with st.container():

    st.markdown(
        """
        <div class="custom-container">
            <h1>Federative Learning</h1>
            <p>
                Federated learning is a machine learning approach that trains an algorithm across multiple devices or servers 
                holding local data samples without exchanging them. It enables model training without centralizing data.
            </p>
        </div>
        <div class="custom-container">
            <h1>PATE</h1>
            <p>
                Private Aggregation of Teacher Ensembles (PATE) is a method for training machine learning models on private 
                data. It protects the privacy of individual data points while allowing model training on the aggregate 
                information.
            </p>
        </div>
        <div class="custom-container">
            <h1>Grid Search</h1>
            <p>
                Grid search is a hyperparameter tuning technique that systematically tests a range of hyperparameters for a 
                machine learning model. It helps in finding the best combination of hyperparameters for optimal model performance.
            </p>
        </div>
        """
        , unsafe_allow_html=True
    )
