import tensorflow
import keras
from keras.models import load_model

from PIL import Image
import numpy as np
import cv2


# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Save the captured image
        cv2.imwrite('captured_image.jpg', frame)
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Model loading
model = load_model("keras_model.h5")

# Class import
class_names = open("labels.txt", "r").readlines()

# Test Image
image = Image.open('captured_image.jpg').convert("RGB")

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)