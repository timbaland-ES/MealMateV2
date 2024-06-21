
import torch
import cv2
import matplotlib.pyplot as plt

# Load the YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to load and prepare the image
def load_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image '{image_path}'")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Function to perform object detection
def detect_objects(model, image):
    results = model(image)  # Perform inference
    return results

# Function to visualize the detection results and filter for apples
def visualize_detections(image, results, target_class='apple'):
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    n = len(labels)
    x_shape, y_shape = image.shape[1], image.shape[0]

    for i in range(n):
        row = cord[i]
        if row[4] >= 0.5:  # Confidence threshold
            label = results.names[int(labels[i])]
            if label == target_class:  # Filter for target class
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                bgr = (0, 255, 0)  # Color for bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(image, f"{label} {row[4]:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Main function to run the object detection
def main(image_path):
    try:
        image = load_image(image_path)
        if image is None:
            return
        
        results = detect_objects(model, image)
        visualize_detections(image, results)
    except Exception as e:
        print(f"Error processing image: {e}")

# Path to the image file
image_path ='/home/timbo/repos/yolov5/data/images/fruits.jpeg'
# Run the main function
main(image_path)
