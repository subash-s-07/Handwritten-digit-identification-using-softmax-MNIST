client_models = []
import os
import tensorflow as tf
import plotly.express as px
import pandas as pd
from sklearn.datasets import fetch_openml
import streamlit as st
from sklearn.decomposition import PCA
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


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
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
train_labels = train_labels.astype('int')

directory_path = r'Models'
files = os.listdir(directory_path)

for i  in files:
    temp = tf.keras.models.load_model(directory_path+"/"+i)
    client_models.append(temp)

global_model = tf.keras.models.load_model("FedaratedModel (1).h5")



import streamlit as st

def v1():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=['Global Model'], y=[global_model.evaluate(test_images, test_labels)[1] * 100], mode='lines+markers', name='Global Model'))

    for i, model in enumerate(client_models):
        accuracy = model.evaluate(test_images, test_labels)[1] * 100
        fig.add_trace(go.Scatter(x=[f'Client {i+1}'], y=[accuracy], mode='lines+markers', name=f'Client {i+1}'))

    fig.update_layout(title='Model Accuracy Comparison', xaxis_title='Models', yaxis_title='Accuracy (%)', template='plotly_dark')

    # Streamlit app
    st.title('Model Accuracy Comparison')
    st.plotly_chart(fig)

def v2():
    num_clients = len(client_models)

    # Calculate the global and client losses
    global_loss = global_model.evaluate(train_images, train_labels, verbose=0)[0]
    client_losses = [model.evaluate(train_images, train_labels, verbose=0)[0] for model in client_models]
    client_names = ["Client " + str(i+1) for i in range(num_clients)]

    # Create a Plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=client_names, y=client_losses, mode='lines+markers', name='Client Models'))
    fig.add_trace(go.Scatter(x=['Global Model'], y=[global_loss], mode='markers', name='Global Model'))

    fig.update_layout(title='Training Loss Trend',
                    xaxis_title='Models',
                    yaxis_title='Loss')

    # Streamlit app
    st.title('Training Loss Trend')
    st.plotly_chart(fig)
def v3():
    layer_index = 1  # Index of the layer for which you want to visualize weights

    num_clients = len(client_models)

    # Get the global weights and client weights for the specified layer
    global_weights = global_model.layers[layer_index].get_weights()[0].flatten()
    client_weights = [model.layers[layer_index].get_weights()[0].flatten() for model in client_models]

    # Create a Plotly figure
    fig = go.Figure()

    fig.add_trace(go.Histogram(x=global_weights, name='Global Model'))
    for i in range(num_clients):
        fig.add_trace(go.Histogram(x=client_weights[i], name=f'Client {i+1}'))

        fig.update_layout(title=f'Weight Distribution of Layer {layer_index}',
                        xaxis_title='Weight Value',
                        yaxis_title='Count')

    # Streamlit app
    st.title(f'Weight Distribution of Layer {layer_index}')
    st.plotly_chart(fig)
def v5():
    # Predict test labels using the global model
    predictions = global_model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    # Generate confusion matrix
    cm = confusion_matrix(test_labels, predicted_labels)

    # Streamlit app
    st.title('Confusion Matrix')
    st.write('Confusion Matrix for Global Model Predictions:')
    st.write(cm)

    # Plot confusion matrix using Matplotlib and Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    st.pyplot(plt)
def v6():
    test_predictions = global_model.predict(test_images)
    predicted_labels = np.argmax(test_predictions, axis=1)

    # Confusion Matrix
    confusion_matrix = tf.math.confusion_matrix(test_labels, predicted_labels)
    confusion_matrix_fig = px.imshow(confusion_matrix, labels=dict(x="Predicted", y="True"), x=[str(i) for i in range(10)], y=[str(i) for i in range(10)])
    confusion_matrix_fig.update_layout(title="Confusion Matrix")

    # Accuracy per class
    class_accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    class_accuracy_percentage = class_accuracy * 100
    accuracy_per_class_fig = px.bar(x=[str(i) for i in range(10)], y=np.diag(confusion_matrix), labels={'x': 'Class', 'y': 'Correct Predictions'})
    accuracy_per_class_fig.update_layout(title=f"Accuracy per Class: {class_accuracy_percentage:.2f}%")

    # Sample Test Images and Predictions
    sample_indices = np.random.choice(len(test_images), 9, replace=False)
    sample_test_images = test_images[sample_indices]
    sample_test_labels = test_labels[sample_indices]
    sample_predictions = predicted_labels[sample_indices]

    sample_images_fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(sample_test_images[i].reshape(28, 28), cmap='binary')
        ax.set_title(f"True: {sample_test_labels[i]}, Predicted: {sample_predictions[i]}")
        ax.axis('off')

    # Distribution of Predictions
    predicted_classes = np.argmax(test_predictions, axis=1)
    predicted_classes_fig = px.histogram(x=predicted_classes, nbins=10, labels={'x': 'Predicted Class'}, title="Distribution of Predicted Classes")

    # Streamlit app
    st.title('Model Evaluation')
    st.subheader('Confusion Matrix')
    st.plotly_chart(confusion_matrix_fig)

    st.subheader('Accuracy per Class')
    st.plotly_chart(accuracy_per_class_fig)

    st.subheader('Sample Test Images and Predictions')
    st.pyplot(sample_images_fig)

def v8():
    # Simulated data for epochs and accuracies (replace with actual data)
    epochs = list(range(1, 6))
    accuracies = [0.85, 0.88, 0.92, 0.94, 0.96]

    # Create a Plotly line chart
    fig = px.line(x=epochs, y=accuracies, markers=True, labels={'x':'Epoch', 'y':'Accuracy'}, title='Accuracy Over Epochs')
    fig.update_traces(line_color='orange', marker_color='green')

    # Streamlit app
    st.title('Accuracy Over Epochs')
    st.plotly_chart(fig)

def v9():
    loss_values = [model.evaluate(test_images, test_labels)[0] for model in client_models]

    # Create a Plotly violin plot
    fig = px.violin(y=loss_values, box=True, points="all", labels={'y': 'Loss Value'}, title='Distribution of Loss Values among Clients')
    fig.update_traces(marker_color='pink')

    # Streamlit app
    st.title('Distribution of Loss Values among Clients')
    st.plotly_chart(fig)

def v10():
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    epochs = [1, 2, 3, 4, 5]  # Assuming 5 epochs for each client model
    accuracy_values = [0.92, 0.89, 0.94, 0.91, 0.93]  # Replace with actual accuracy values for each configuration

    # Create a 3D scatter plot
    fig = px.scatter_3d(x=learning_rates, y=epochs, z=accuracy_values, labels={'x': 'Learning Rate', 'y': 'Epochs', 'z': 'Accuracy'},
                        title='Accuracy vs Learning Rate vs Epochs', color=accuracy_values, size_max=10)
    fig.update_traces(marker=dict(color='green', size=8))

    # Streamlit app
    st.title('Accuracy vs Learning Rate vs Epochs')
    st.plotly_chart(fig)

def v11():
    st.image("Images\Figure_1.png")
    st.image("Images\Figure_2.png")
    st.image("Images\Figure_3.png")
    st.image(r"Images\newplot.png")

def v12():
    # Initialize the evaluation_metrics list
    evaluation_metrics = []

    # Iterate through different learning rates
    for learning_rate in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        client_results = []

        # Evaluate client models
        for client_model in client_models:
            predictions = client_model.predict(test_images)
            y_pred = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(test_labels, y_pred)
            precision = precision_score(test_labels, y_pred, average='weighted')
            recall = recall_score(test_labels, y_pred, average='weighted')
            f1 = f1_score(test_labels, y_pred, average='weighted')

            client_results.append({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

        # Evaluate global model
        global_predictions = global_model.predict(test_images)
        global_y_pred = np.argmax(global_predictions, axis=1)
        global_accuracy = accuracy_score(test_labels, global_y_pred)
        global_precision = precision_score(test_labels, global_y_pred, average='weighted')
        global_recall = recall_score(test_labels, global_y_pred, average='weighted')
        global_f1 = f1_score(test_labels, global_y_pred, average='weighted')

        # Store evaluation results in evaluation_metrics list
        evaluation_metrics.append({
            'learning_rate': learning_rate,
            'clients': client_results,
            'global': {
                'accuracy': global_accuracy,
                'precision': global_precision,
                'recall': global_recall,
                'f1_score': global_f1
            }
        })
        data = {'Learning Rate': [], 'Metric': [], 'Value': []}

    for entry in evaluation_metrics:
        learning_rate = entry['learning_rate']

        # Client models
        for i, client_result in enumerate(entry['clients']):
            data['Learning Rate'].append(learning_rate)
            data['Metric'].append(f'Client {i + 1}')
            data['Value'].append(client_result['accuracy'])

        # Global model
        data['Learning Rate'].append(learning_rate)
        data['Metric'].append('Global')
        data['Value'].append(entry['global']['accuracy'])

    df = pd.DataFrame(data)

    # Create a tree map using Plotly
    fig = px.treemap(df, path=['Learning Rate', 'Metric'], values='Value')

    # Streamlit app
    st.title('Tree Map of Evaluation Metrics')
    st.plotly_chart(fig)



v1()
v2()
v3()

v5()
v6()
v8()
v9()
v10()
v11()
v12()