from sklearn.model_selection import ParameterSampler
import numpy as np

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

def run_parameter_sampler(train_data, train_labels, test_data, test_labels):
    # Initialize sampler
    sampler = ParameterSampler(param_distributions, n_iter=3)
    
    # Store results
    trial_list = []
    
    # Iterate through parameter combinations
    for param in sampler:
        # Get model with current parameters
        model = get_model(param)
        
        # Train model
        history = model.fit(
            train_data, 
            train_labels,
            epochs=10,
            validation_data=(test_data, test_labels)
        )
        
        # Evaluate model
        loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
        
        # Store results
        trial_list.append({
            'val_accuracy': accuracy,
            'val_loss': loss,
            'params': param
        })
    
    return trial_list 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

def get_model(param):
    """Create and return a Keras model with configurable hyperparameters."""
    model = Sequential()
    
    # First Conv2D layer
    model.add(Conv2D(param['Nodes'], (param['Kernel'], param['Kernel']), 
                     activation=param['Activation'],
                     input_shape=(256, 256, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Add dynamic number of Dense layers
    model.add(Flatten())  # Flatten the 2D arrays for the dense layers
    
    for i in range(0, param['Layers']):
        model.add(Dense(32, activation='relu'))
    
    # Add optional dropout
    if param['Add_Drop'] == 'True':
        model.add(Dropout(param['Drop_Rate']))
    
    # Final output layer with 4 classes (matching your dummy data)
    model.add(Dense(4, activation='softmax'))
    
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer='adam'
    )
    
    return model