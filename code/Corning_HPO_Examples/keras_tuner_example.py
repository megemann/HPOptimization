import keras_tuner as kt  # type: ignore
from tensorflow import keras  # type: ignore
from generate_dummy_data import generate_dummy_data  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore

def get_model(hp):
    """Create a model with hyperparameters to be tuned by Keras Tuner.
    
    Args:
        hp: Keras Tuner hyperparameter object
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential()
    
    # Input layer
    model.add(keras.layers.Input(shape=(256, 256, 1)))
    
    # Basic tuning parameters
    activation = hp.Choice("Activation", values=["gelu", "relu", "tanh"])
    nodes = hp.Int("Nodes", min_value=2, max_value=32, step=2)
    kernel = hp.Int("Kernel", min_value=2, max_value=4)
    
    # Conv2D layer
    model.add(keras.layers.Conv2D(12, (kernel, kernel), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    
    # Dynamic number of layers
    n_layers = hp.Int("Layers", min_value=1, max_value=5)
    for _ in range(n_layers):
        model.add(keras.layers.Dense(nodes, activation=activation))
    
    # Optional dropout
    if hp.Boolean("Add_Drop"):
        dropout_rate = hp.Float("Drop_Rate", min_value=0.2, max_value=0.5)
        model.add(keras.layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(keras.layers.Dense(4, activation='softmax'))
    
    # Compile with tunable learning rate
    learning_rate = hp.Float(
        "Learning_Rate",
        min_value=0.001,
        max_value=0.1,
        sampling="log"
    )
    
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    )
    
    return model

def main():
    # Get and preprocess data
    train_data, train_labels = generate_dummy_data(1000)
    test_data, test_labels = generate_dummy_data(200)
    
    # Preprocess data
    train_data = train_data.reshape(-1, 256, 256, 1)
    test_data = test_data.reshape(-1, 256, 256, 1)
    
    # Normalize data
    train_data = train_data.astype('float32') / 255.0
    test_data = test_data.astype('float32') / 255.0
    
    # Convert labels to categorical
    if len(train_labels.shape) == 1 or train_labels.shape[1] == 1:
        train_labels = to_categorical(train_labels, num_classes=4)
        test_labels = to_categorical(test_labels, num_classes=4)
    
    # Create tuner
    tuner = kt.Hyperband(
        get_model,
        objective='val_accuracy',
        max_epochs=20,
        factor=3
    )
    
    # Start the search
    tuner.search(
        train_data,
        train_labels,
        validation_data=(test_data, test_labels)
    )
    
    # Get best model and hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models()[0]
    
    # Print results
    print("\nBest hyperparameters:")
    for param, value in best_hps.values.items():
        print(f"    {param}: {value}")

if __name__ == "__main__":
    main() 