import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.preprocessing import image
from mtcnn import MTCNN

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Add dropout layer here
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# Set up the ImageDataGenerator for training, validation, and testing
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize the images to a range of [0, 1]
    rotation_range=40,  # Random rotation between 0 and 40 degrees
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,  # Shear images randomly
    zoom_range=0.2,  # Zoom in and out
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Fill any missing pixels with nearest neighbor
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for validation and testing

# Set up directories for training, testing, and validation
train_dir = "/Users/rakeshpathlavath/Downloads/Face_Mask_Dataset/Train/"
validation_dir = "/Users/rakeshpathlavath/Downloads/Face_Mask_Dataset/Test/"
test_dir = "/Users/rakeshpathlavath/Downloads/Face_Mask_Dataset/Test/"

# Load the images in batches from the directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Resize images to 150x150
    batch_size=32,
    class_mode='binary'  # Binary classification (with/without mask)
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Compile the model
model.compile(
    loss='binary_crossentropy',  
    optimizer='adam', 
    metrics=['accuracy']
)

# Train the model
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=[early_stopping]  # Add early stopping
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=50)

# Print the test accuracy
print(f'Test accuracy: {test_accuracy * 100:.2f}%')



# Save the trained model
model.save('face_mask_detection_model.keras')

# Load the trained model
model = tf.keras.models.load_model('face_mask_detection_model.keras')

# Load the face detection model (Haar Cascade for face detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



# Initialize MTCNN detector
detector = MTCNN()

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the frame to RGB (MTCNN expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using MTCNN
    result = detector.detect_faces(rgb_frame)

    # Loop over the faces detected
    for person in result:
        x, y, w, h = person['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Crop the face from the frame
        face_img = frame[y:y + h, x:x + w]
        
        # Preprocess the image for model prediction
        img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (150, 150))  # Resize to 150x150 as expected by the model
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img / 255.0  # Normalize the image

        # Predict if the person is wearing a mask
        prediction = model.predict(img)
        
        # Reverse the label mapping if necessary:
        label = 'Without Mask' if prediction[0] > 0.5 else 'With Mask'

        # Put the label on the frame
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display the frame with face rectangles and labels
    cv2.imshow('Face Mask Detection', frame)
    
    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
