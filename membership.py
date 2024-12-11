import tensorflow as tf
import tensorflow_federated as tff

# Preprocess CIFAR-10 data
def preprocess_data(images, labels):
    def augment(image, label):
        image = tf.image.resize(image, [32, 32])
        image = tf.image.random_flip_left_right(image)
        image = tf.cast(image, tf.float32) / 255.0  # Normalize
        label = tf.one_hot(tf.squeeze(label), depth=10)  # One-hot encode
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# Load and preprocess CIFAR-10
def load_and_prepare_cifar10():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    return train_images, train_labels, test_images, test_labels

train_images, train_labels, test_images, test_labels = load_and_prepare_cifar10()

# Create federated client datasets
def create_client_datasets_with_simulation(images, labels, num_clients):
    client_datasets = {}
    num_samples_per_client = len(images) // num_clients

    for i in range(num_clients):
        start_idx = i * num_samples_per_client
        end_idx = start_idx + num_samples_per_client

        # Create a dataset for each client
        client_images = images[start_idx:end_idx]
        client_labels = tf.squeeze(labels[start_idx:end_idx])  # Remove extra dimension

        dataset = tf.data.Dataset.from_tensor_slices((client_images, client_labels))
        dataset = dataset.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.one_hot(y, depth=10))
        ).batch(5)

        client_datasets[f'client_{i}'] = dataset

    return client_datasets


NUM_CLIENTS = 10
client_datasets = create_client_datasets_with_simulation(train_images, train_labels, NUM_CLIENTS)

# Convert dictionary of datasets to a list for TFF
federated_datasets = [dataset for _, dataset in client_datasets.items()]

# Define Keras model
def create_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Model function
def model_fn():
    keras_model = create_keras_model()
    loss = tf.keras.losses.CategoricalCrossentropy()
    input_spec = (
        tf.TensorSpec(shape=[None, 32, 32, 3], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 10], dtype=tf.float32),
    )
    return tff.learning.models.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=loss,
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

# Define optimizers
def client_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

def server_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=1.0, momentum=0.9)

# Federated learning process
iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn=model_fn,
    client_optimizer_fn=client_optimizer_fn,
    server_optimizer_fn=server_optimizer_fn
)

# Initialize training
state = iterative_process.initialize()

# Train the model
NUM_ROUNDS = 5500
for round_num in range(1, NUM_ROUNDS + 1):
    state, metrics = iterative_process.next(state, federated_datasets)
    print(f"Round {round_num}: {metrics}")

# Evaluate the model
# Evaluate the model
def evaluate_model(state):
    keras_model = create_keras_model()
    keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model_weights = iterative_process.get_model_weights(state)
    keras_model.set_weights(model_weights.trainable)

    # Preprocess test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.one_hot(tf.squeeze(y), depth=10))  # Remove extra dimension
    ).batch(10)

    test_loss, test_accuracy = keras_model.evaluate(test_dataset, verbose=0)
    return test_loss, test_accuracy


final_test_loss, final_test_accuracy = evaluate_model(state)
print(f"Final Test Loss: {final_test_loss:.4f}")
print(f"Final Test Accuracy: {final_test_accuracy:.4f}")
