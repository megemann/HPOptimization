import sys
import os
from tensorflow.keras.utils import to_categorical  # type: ignore

# Add the parent directory to the Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_dummy_data import generate_dummy_data
from parameter_sampler_example import run_parameter_sampler
from randomized_search_example import run_randomized_search
def preprocess_data(data, labels):
    # Reshape data to have proper dimensions for CNN
    data = data.reshape(-1, 256, 256, 1)  # Changed to match 256x256 input shape
    # Normalize data
    data = data.astype('float32') / 255.0
    # Convert labels to categorical
    # Check if labels are already one-hot encoded
    if len(labels.shape) == 1 or labels.shape[1] == 1:
        # Convert labels to categorical only if they're not already one-hot encoded
        labels = to_categorical(labels, num_classes=4)
    
    return data, labels

def main():
    # Generate dummy data for example using existing function
    train_data, train_labels = generate_dummy_data(1000)
    test_data, test_labels = generate_dummy_data(200)
    
    # Preprocess the data
    train_data, train_labels = preprocess_data(train_data, train_labels)
    test_data, test_labels = preprocess_data(test_data, test_labels)

    print("\nRunning RandomizedSearchCV example...")
    random_results, best_model = run_randomized_search(
        train_data, train_labels, test_data, test_labels
    )
    
    print("Running ParameterSampler example...")
    parameter_results = run_parameter_sampler(
        train_data, train_labels, test_data, test_labels
    )
    
    # Print best results from each method
    print("\nBest ParameterSampler result:")
    best_param = max(parameter_results, key=lambda x: x['val_accuracy'])
    print(f"Accuracy: {best_param['val_accuracy']}")
    print(f"Parameters: {best_param['params']}")
    
    print("\nBest RandomizedSearchCV result:")
    print(f"Best score: {random_results['mean_test_score'].max()}")
    print(f"Best parameters: {best_model.get_params()}")

if __name__ == "__main__":
    main()