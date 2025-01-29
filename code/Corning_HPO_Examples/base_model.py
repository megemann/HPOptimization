import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from generate_dummy_data import generate_dummy_data

def get_model():
    model = Sequential()
    # Input layer expecting (256, 256, 1) images
    model.add(InputLayer(input_shape=(256, 256, 1)))
    
    # Convolutional layer
    model.add(Conv2D(4, (2,2), activation='relu'))  # Output: (None, 255, 255, 4)
    
    # Max pooling layer
    model.add(MaxPooling2D((2,2)))  # Output: (None, 127, 127, 4)
    
    # Dropout layer
    model.add(Dropout(0.4))
    
    # Flatten layer
    model.add(Flatten())
    
    # Dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='softmax'))  # 4 output classes
    
    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    try:
        # Create the model
        print("Creating model...")
        model = get_model()
        
        # Generate dummy training data
        print("Generating dummy data...")
        X_train, y_train = generate_dummy_data(100)
        X_val, y_val = generate_dummy_data(20)
        
        # Create early stopping callback
        print("Creating early stopping callback...")
        from tensorflow.keras.callbacks import EarlyStopping
        callback = EarlyStopping(monitor='val_loss', patience=5)
        
        # Train the model
        print("Training model...")
        history = model.fit(
            X_train, 
            y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            callbacks=[callback]
        )
        print("Model training complete.")
        print("Training history:")
        print(history.history)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
