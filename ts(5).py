import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from keras import layers, models, utils
from collections import deque

# Constants
data_dir = "/home/rutub/Documents/Tsbd1/"
train_path = os.path.join(data_dir, "archive (2)/train")
test_path = os.path.join(data_dir, "archive (2)/test")
IMG_HEIGHT = 30
IMG_WIDTH = 30

# Mapping classes to names
classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited', 17: 'No entry', 18: 'General caution',
    19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve',
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead',
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left',
    38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory',
    41: 'End of no passing', 42: 'End no passing vehicle with a weight greater than 3.5 tons'
}

# Load and preprocess images
def load_images(path):
    image_data = []
    image_labels = []

    for i in range(len(classes)):
        class_path = os.path.join(path, str(i))
        for img in os.listdir(class_path):
            try:
                image = cv2.imread(os.path.join(class_path, img), cv2.IMREAD_GRAYSCALE)  # Read as grayscale
                image_fromarray = Image.fromarray(image, 'L')
                resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
                image_data.append(np.array(resize_image))
                image_labels.append(i)
            except Exception as e:
                print(f"Error in {img}: {e}")
    
    image_data = np.array(image_data)
    image_labels = np.array(image_labels)
    
    return image_data, image_labels

# Load training data
X_train, y_train = load_images(train_path)
X_train = X_train / 255.0  # Normalize pixel values

# Reshape for the model input
X_train = X_train.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)

# Split training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Convert labels to categorical
y_train = utils.to_categorical(y_train, num_classes=len(classes))
y_val = utils.to_categorical(y_val, num_classes=len(classes))

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# Define CNN model
model = models.Sequential([
    layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),  # Change input shape to (30, 30, 1)
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.BatchNormalization(axis=-1),
    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.BatchNormalization(axis=-1),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.5),
    layers.Dense(len(classes), activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_val, y_val))

# Evaluate on validation set
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc*100:.2f}%")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the model
model.save('traffic_sign_classifier.h5')



# Load the trained model
model = models.load_model('traffic_sign_classifier.h5')

# Function to preprocess image
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from keras import models
from collections import deque

# Constants
IMG_HEIGHT = 30
IMG_WIDTH = 30

# Load the trained model
model = models.load_model('traffic_sign_classifier.h5')

# Function to preprocess image
def preprocess_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))  # Resize to model's expected input size
    img = img / 255.0  # Normalize
    img = img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)  # Reshape for model input
    return img

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open camera.")
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX
predictions_deque = deque(maxlen=10)  # Store last 10 predictions for smoothing

while True:
    success, imgOriginal = cap.read()

    if not success:
        print("Failed to read frame from the camera!")
        break

    # Preprocess the captured image
    img = preprocess_img(imgOriginal)

    # Make prediction
    predictions = model.predict(img)
    predictions_deque.append(predictions)
    avg_predictions = np.mean(predictions_deque, axis=0)  # Calculate moving average
    classIndex = np.argmax(avg_predictions)
    probabilityValue = np.max(avg_predictions)

    # Only display the prediction if the probability is above a certain threshold
    if probabilityValue > 0.5:
        class_name = classes[classIndex]  # Get class name from class index
        cv2.putText(imgOriginal, f"CLASS: {class_name}", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, f"PROBABILITY: {round(probabilityValue * 100, 2)}%", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Result", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
