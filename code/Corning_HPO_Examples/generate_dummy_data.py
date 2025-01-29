import numpy as np

def generate_dummy_data(num_samples=100):
    # Create random black images (256x256x1)
    X = np.zeros((num_samples, 256, 256, 1))
    
    # Create random labels (4 classes)
    y = np.random.randint(0, 4, num_samples)
    # Convert to one-hot encoding
    y_one_hot = np.zeros((num_samples, 4))
    y_one_hot[np.arange(num_samples), y] = 1
    
    return X, y_one_hot

if __name__ == "__main__":
    # Test the data generation
    X, y = generate_dummy_data(10)
    print("Generated data shapes:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}") 