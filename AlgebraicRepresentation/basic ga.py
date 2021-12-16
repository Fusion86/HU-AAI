import keras
import tensorflow as tf

# Workaround for issue with tensorflow when using a GPU
# See https://stackoverflow.com/questions/43990046/tensorflow-blas-gemm-launch-failed
gpu_workaround = True

if __name__ == "__main__":
    if gpu_workaround:
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()

    # Setup neural network
    model = keras.models.Sequential(
        [
            keras.layers.Flatten(),
            keras.layers.Dense(600, activation="relu"),
            keras.layers.Dense(100, activation="relu"),
            keras.layers.Dropout(0.1),
            # 10 output nodes because we have 10 digits (0-9)
            keras.layers.Dense(10),
        ]
    )

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    # Train network
    model.fit(train_x, train_y, epochs=5)

    # Test network
    loss, accuracy = model.evaluate(test_x, test_y, verbose=2)

    # Write results to a file
    network_structure_lst = []
    for layer in model.layers:
        if type(layer) == keras.layers.Dense:
            network_structure_lst.append(str(layer.units))

    network_structure = "-".join(network_structure_lst[:-1])

    with open("results.txt", "a") as f:
        f.write(f"{network_structure},{loss:.3},{accuracy:.3}\n")
