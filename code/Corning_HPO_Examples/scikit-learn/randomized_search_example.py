from sklearn.model_selection import RandomizedSearchCV  # type: ignore
from scikeras.wrappers import KerasClassifier  # type: ignore

def run_randomized_search(train_data, train_labels, test_data, test_labels):
    # Define parameter space
    param_distributions = {
        'Layers': [1,2,3,4,5],
        'Activation': ['gelu', 'relu', 'tanh'],
        'Nodes': [2, 10, 18, 26, 32],
        'Kernel': [2,3,4],
        'Add_Drop': ['True', 'False'],
        'Drop_Rate': [0.3, 0.35, 0.4, 0.45, 0.5],
        'Learning_Rate': [0.01, 0.005, 0.001, 0.0005, 0.0001]
    }
    
    # Create KerasClassifier
    model = KerasClassifier(
        model=get_model,
        epochs=10,
        batch_size=32,
        verbose=0,
        Layers=1,  # Default values
        Activation='relu',
        Nodes=10,
        Kernel=3,
        Add_Drop='False',
        Drop_Rate=0.3,
        Learning_Rate=0.01
    )
    
    # Initialize RandomizedSearchCV - NOTE must use older version of scikit-learn (1.5.2)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=3,
        cv=3,
        verbose=1,
        n_jobs=1  # Use n_jobs=1 to avoid potential multiprocessing issues
    )
    
    # Fit the random search
    print("Fitting RandomizedSearchCV...")
    random_search_result = random_search.fit(train_data, train_labels)
    
    # Get results
    results = random_search_result.cv_results_
    best_model = random_search_result.best_estimator_
    
    return results, best_model


from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

def get_model(Nodes=10, Activation='relu', Kernel=3, Layers=1, Add_Drop='False', Drop_Rate=0.3, Learning_Rate=0.01):
    """Create a Keras model with configurable hyperparameters.
    
    Parameters:
    - Nodes: Number of nodes in first dense layer
    - Activation: Activation function to use
    - Kernel: Kernel size for Conv2D layer
    - Layers: Number of additional dense layers
    - Add_Drop: Whether to add dropout layer
    - Drop_Rate: Dropout rate if Add_Drop is True
    - Learning_Rate: Learning rate for Adam optimizer
    """
    model = Sequential()

    model.add(Input(shape=(256, 256, 1)))
    
    # Conv2D layer with configurable kernel size
    model.add(Conv2D(12, (Kernel, Kernel), activation='relu'))
    
    # Flatten before dense layers
    model.add(Flatten())
    
    # Add configurable number of Dense layers
    for _ in range(0, Layers):
        model.add(Dense(Nodes, activation=Activation))
    
    # Optional dropout layer
    if Add_Drop == 'True':
        model.add(Dropout(Drop_Rate))
    
    # Output layer (assuming 4 classes based on dummy data)
    model.add(Dense(4, activation='softmax'))
    
    # Compile model with configurable learning rate
    model.compile(
        optimizer=Adam(learning_rate=Learning_Rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

