from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

def get_model(param):
    """Create a Keras model with configurable hyperparameters.
    
    Parameters:
    param (dict): Dictionary containing model parameters:
        - Nodes: Number of nodes in first dense layer
        - Activation: Activation function to use
        - Kernel: Kernel size for Conv2D layer
        - Layers: Number of additional dense layers
        - Add_Drop: Whether to add dropout layer
        - Drop_Rate: Dropout rate if Add_Drop is True
        - Learning_Rate: Learning rate for Adam optimizer
    """
    model = Sequential()
    
    # First Dense layer with configurable nodes and activation
    model.add(Dense(param['Nodes'], activation=param['Activation']))
    
    # Conv2D layer with configurable kernel size
    model.add(Conv2D(12, (param['Kernel'], param['Kernel']), activation='relu'))
    
    # Flatten before dense layers
    model.add(Flatten())
    
    # Add configurable number of Dense layers
    for _ in range(0, param['Layers']):
        model.add(Dense(12, activation='relu'))
    
    # Optional dropout layer
    if param['Add_Drop'] == 'True':
        model.add(Dropout(param['Drop_Rate']))
    
    # Output layer (assuming 4 classes based on dummy data)
    model.add(Dense(4, activation='softmax'))
    
    # Compile model with configurable learning rate
    model.compile(
        optimizer=Adam(learning_rate=param['Learning_Rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model