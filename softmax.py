import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Define a custom learning rate
custom_learning_rate = 0.001  # You can set your desired learning rate here

# Create a custom optimizer with the desired learning rate
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=custom_learning_rate)

# Build the model
model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28, 1)))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model with the custom optimizer
model.compile(optimizer=custom_optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model and evaluate as usual
model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_split=0.2)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
