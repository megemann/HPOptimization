import optuna  # type: ignore
from tensorflow import keras  # type: ignore
from generate_dummy_data import generate_dummy_data  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore

def objective(trial):
    """Objective function for Optuna to optimize.
    
    This function builds and trains a model with hyperparameters suggested by Optuna,
    then returns the validation accuracy for optimization.
    """
    # Make a model and add input
    model = keras.models.Sequential()
    
    # Define hyperparameters using Optuna trial suggestions
    # Basic tuning parameters
    activation = trial.suggest_categorical("Activation", ["gelu", "relu", "tanh"])
    nodes = trial.suggest_int("Nodes", 2, 32)
    kernel = trial.suggest_int("Kernel", 2, 4)
    
    # Input layer
    model.add(keras.layers.Input(shape=(256, 256, 1)))

    # Conv2D layer
    model.add(keras.layers.Conv2D(12, (kernel, kernel), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten before dense layers
    model.add(keras.layers.Flatten())
    
    # Dynamic number of layers
    n_layers = trial.suggest_int("Layers", 1, 5)
    for _ in range(n_layers):
        model.add(keras.layers.Dense(nodes, activation=activation))
    
    # Optional dropout
    add_dropout = trial.suggest_categorical("Add_Drop", ['True', 'False'])
    if add_dropout == 'True':
        dropout_rate = trial.suggest_float("Drop_Rate", 0.2, 0.5)
        model.add(keras.layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(keras.layers.Dense(4, activation='softmax'))
    
    # Compile with suggested learning rate
    learning_rate = trial.suggest_float("Learning_Rate", 0.0001, 0.01, log=True)
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    )
    
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
    
    # Train the model
    history = model.fit(
        train_data,
        train_labels,
        epochs=10,
        validation_data=(test_data, test_labels)
    )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(test_data, test_labels)
    return accuracy

def main():
    # Create a study object with direction="maximize" for accuracy
    study = optuna.create_study(
        study_name="keras_hpo",
        direction="maximize"
    )
    
    # Start the optimization with 20 trials
    study.optimize(objective, n_trials=3)
    
    # Print results
    print("\nBest trial:")
    print("Value: ", study.best_trial.value)
    print("Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main() 