import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

# Preprocess CIFAR-10 data
def preprocess_data(images, labels):
    images = tf.image.resize(images, [32, 32])
    images = tf.cast(images, tf.float32) / 255.0  # Normalize pixel values
    return images, tf.one_hot(labels.flatten(), depth=10)

def load_and_prepare_cifar10():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images, train_labels = preprocess_data(train_images, train_labels)
    test_images, test_labels = preprocess_data(test_images, test_labels)
    return (train_images, train_labels), (test_images, test_labels)

(train_images, train_labels), (test_images, test_labels) = load_and_prepare_cifar10()

# Split CIFAR-10 data into federated client datasets
NUM_CLIENTS = 5
client_data = []
client_size = len(train_images) // NUM_CLIENTS

for i in range(NUM_CLIENTS):
    start = i * client_size
    end = (i + 1) * client_size
    client_data.append({
        'images': train_images[start:end],
        'labels': train_labels[start:end]
    })

# Define the model using Keras
def create_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Wrap the model for TFF
def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=(
            tf.TensorSpec(shape=[None, 32, 32, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 10], dtype=tf.float32)
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

# Create Federated Averaging process
iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
)

# Initialize training
state = iterative_process.initialize()

# Simulate federated training
NUM_ROUNDS = 10
for round_num in range(1, NUM_ROUNDS + 1):
    federated_data = [
        tff.simulation.ClientData.from_tensor_slices((client['images'], client['labels']))
        for client in client_data
    ]
    state, metrics = iterative_process.next(state, federated_data)
    print(f"Round {round_num}: {metrics}")

# Evaluate the trained model
def evaluate_model(state):
    keras_model = create_keras_model()
    keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    keras_model.set_weights(state.model.trainable)
    test_loss, test_accuracy = keras_model.evaluate(test_images, test_labels, verbose=0)
    return test_loss, test_accuracy

final_test_loss, final_test_accuracy = evaluate_model(state)
print(f"Final Test Loss: {final_test_loss:.4f}")
print(f"Final Test Accuracy: {final_test_accuracy:.4f}")
