import tensorflow as tf
import tensorflow_federated as tff

# Preprocess CIFAR-10 data
def preprocess_data(images, labels):
    images = tf.image.resize(images, [32, 32])
    images = tf.cast(images, tf.float32) / 255.0  # Normalize
    labels = tf.one_hot(labels.flatten(), depth=10)  # One-hot encode
    return images, labels

# Load and preprocess CIFAR-10
def load_and_prepare_cifar10():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images, train_labels = preprocess_data(train_images, train_labels)
    test_images, test_labels = preprocess_data(test_images, test_labels)
    return (train_images, train_labels), (test_images, test_labels)

(train_images, train_labels), (test_images, test_labels) = load_and_prepare_cifar10()

# Define a function to create TFF datasets
def create_tf_dataset(images, labels):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=100).batch(20)  # Shuffle and batch data
    return dataset

# Prepare federated data
NUM_CLIENTS = 5
client_datasets = [
    create_tf_dataset(
        train_images[i * 1000:(i + 1) * 1000],
        train_labels[i * 1000:(i + 1) * 1000]
    )
    for i in range(NUM_CLIENTS)
]

# Define the Keras model
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

# Define the model function
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

# Define optimizer factories
def client_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=0.01)

def server_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=1.0)

# Build federated averaging process
iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn=model_fn,
    client_optimizer_fn=client_optimizer_fn,
    server_optimizer_fn=server_optimizer_fn
)

# Initialize federated training
state = iterative_process.initialize()

# Train the federated model
NUM_ROUNDS = 20
for round_num in range(1, NUM_ROUNDS + 1):
    state, metrics = iterative_process.next(state, client_datasets)
    print(f"Round {round_num}: {metrics}")

# Evaluate the trained model
def evaluate_model(state):
    keras_model = create_keras_model()
    keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Extract trainable weights from the TFF state
    model_weights = iterative_process.get_model_weights(state)
    keras_model.set_weights(model_weights.trainable)

    # Evaluate the model on the test data
    test_loss, test_accuracy = keras_model.evaluate(test_images, test_labels, verbose=0)
    return test_loss, test_accuracy


final_test_loss, final_test_accuracy = evaluate_model(state)
print(f"Final Test Loss: {final_test_loss:.4f}")
print(f"Final Test Accuracy: {final_test_accuracy:.4f}")
