import numpy as np

def add_label_noise(y_train, y_test, noise_level=0.2):
    """
    Add noise to labels of training and testing datasets.
    
    Args:
    - y_train: Training labels (categorical format)
    - y_test: Testing labels (categorical format)
    - noise_level: Percentage of labels to randomize (0.0 to 1.0)
    
    Returns:
    - Noisy training and testing labels
    """
    # Create copies to avoid modifying original arrays
    y_train_noisy = y_train.copy()
    y_test_noisy = y_test.copy()
    
    # Number of classes
    num_classes = y_train.shape[1]
    
    # Add noise to training labels
    num_train_noise = int(len(y_train) * noise_level)
    train_noise_indices = np.random.choice(len(y_train), num_train_noise, replace=False)
    
    for idx in train_noise_indices:
        # Create a random label different from the original
        original_class = np.argmax(y_train[idx])
        noisy_class = np.random.choice([c for c in range(num_classes) if c != original_class])
        
        # Create new one-hot encoded label
        y_train_noisy[idx] = 0
        y_train_noisy[idx, noisy_class] = 1
    
    # Add noise to testing labels (optional, less common)
    num_test_noise = int(len(y_test) * (noise_level / 2))  # Less noise on test set
    test_noise_indices = np.random.choice(len(y_test), num_test_noise, replace=False)
    
    for idx in test_noise_indices:
        original_class = np.argmax(y_test[idx])
        noisy_class = np.random.choice([c for c in range(num_classes) if c != original_class])
        
        y_test_noisy[idx] = 0
        y_test_noisy[idx, noisy_class] = 1
    
    return y_train_noisy, y_test_noisy

# Example usage in main script:
# y_train, y_test = add_label_noise(y_train, y_test, noise_level=0.2)