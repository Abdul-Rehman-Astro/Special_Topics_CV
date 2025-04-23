import os
import numpy as np
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
# from noise_adder import add_label_noise


def add_label_noise(y_train, y_test, noise_level=1):

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

# 1. Data Loading
print("Loading dataset...")
dataset_path = "/home/abdul_rehman/scratch/Assignment2/UCMercedDatasetOriginal/Images"  # Replace with your path

# Initialize lists for images and labels
images = []
labels = []
label_to_idx = {}

# Get class folders
class_folders = sorted([d for d in os.listdir(dataset_path) 
                       if os.path.isdir(os.path.join(dataset_path, d))])
print(f"Found {len(class_folders)} classes")

# Create label mapping
for idx, class_name in enumerate(class_folders):
    label_to_idx[class_name] = idx
# Load images
for class_name in class_folders:
    class_path = os.path.join(dataset_path, class_name)
    image_files = glob.glob(os.path.join(class_path, '*.tif'))
    
    print(f"Loading {len(image_files)} images from class {class_name}")
    
    for img_path in image_files:
        # Load and resize image to a smaller size
        img = Image.open(img_path).convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        
        images.append(img_array)
        labels.append(label_to_idx[class_name])

# Convert to numpy arrays
X = np.array(images)
y = np.array(labels)



# Example usage in main script:
# y_train, y_test = add_label_noise(y_train, y_test, noise_level=0.2)

# 2. Train-Test Split
print("\nSplitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# 3. Convert labels to categorical
num_classes = len(class_folders)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)


y_train, y_test = add_label_noise(y_train, y_test, noise_level=1)


# # 4. Create a simple CNN model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     Flatten(),
#     Dense(64, activation='relu'),
#     Dropout(0.5),
#     Dense(num_classes, activation='softmax')
# ])

from tensorflow.keras.applications import ResNet50

# 4. Create ResNet50 model
model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3),
    pooling='avg'
)

# Add classification layers
x = model.output
x = Dense(64, activation='relu')(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs=model.input, outputs=outputs)

# Optional: Freeze base model layers
for layer in model.layers[:-2]:
    layer.trainable = False

# 5. Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train the model
print("\nTraining the model...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_test, y_test)
)

# 7. Evaluate the model
print("\nEvaluating the model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# 8. Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 9. Show some predictions
print("\nMaking predictions on test images...")
indices = np.random.randint(0, len(X_test), 5)
sample_images = X_test[indices]
predictions = model.predict(sample_images)

# Create reverse mapping for labels
idx_to_label = {v: k for k, v in label_to_idx.items()}

# Display results
plt.figure(figsize=(15, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(sample_images[i])
    pred_class = idx_to_label[np.argmax(predictions[i])]
    true_class = idx_to_label[np.argmax(y_test[indices[i]])]
    color = 'green' if pred_class == true_class else 'red'
    plt.title(f'Pred: {pred_class}\nTrue: {true_class}', color=color)
    plt.axis('off')
plt.tight_layout()
plt.show()



# Add these imports
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots

# Create output directory
output_dir = r'/home/abdul_rehman/scratch/Assignment2/Results/100_noise_Results'
os.makedirs(output_dir, exist_ok=True)

# Modify training to increase epochs based on dataset size
# epochs = min(50, max(20, len(X_train) // 100))  # Dynamically adjust epochs

# After training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_history.png'))
plt.close()

# Save the model
model.save(os.path.join(output_dir, 'ucmerced_model_100_noise.h5'))

# Predictions visualization
plt.figure(figsize=(15, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(sample_images[i])
    pred_class = idx_to_label[np.argmax(predictions[i])]
    true_class = idx_to_label[np.argmax(y_test[indices[i]])]
    color = 'green' if pred_class == true_class else 'red'
    plt.title(f'Pred: {pred_class}\nTrue: {true_class}', color=color)
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sample_predictions.png'))
plt.close()

# Print output directory for user reference
print(f"\nResults saved in: {output_dir}")
