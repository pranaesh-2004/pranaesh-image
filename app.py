import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
import os

# Load the YOLOv5 model (for object detection)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
target_classes = ['cell phone', 'laptop', 'remote', 'refrigerator']  # Added 'refrigerator' for home appliances

# Define the class names for classification
class_names = ['Apple iPhone', 'Vivo IQ Z6 Lite', 'Dell', 'Onida PXL', 'Whirlpool 235']  # Added 'Whirlpool 235'

# Initialize video capture (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Store the latest captured frame and detection details
latest_frame = None
detection_details = {}

def count_objects(results):
    """Count the number of detected objects for each class."""
    counts = {class_name: 0 for class_name in target_classes}
    for det in results.xyxy[0]:  # Each detection contains [x1, y1, x2, y2, confidence, class_id]
        class_id = int(det[5])
        class_name = yolo_model.names[class_id]
        if class_name in target_classes:
            counts[class_name] += 1
    return counts

st.title("Object Detection with YOLOv5")

# Create a button to start video capture
if st.button("Start Video"):
    # Start video capture and display
    video_placeholder = st.empty()
    while True:
        success, frame = cap.read()
        if not success:
            st.error("Failed to capture video.")
            break

        # Perform object detection using YOLOv5
        results = yolo_model(frame)
        results.render()  # Draw bounding boxes on the frame
        counts = count_objects(results)

        # Update detection details for display
        detection_details.update(counts)

        # Display object counts on the frame
        cv2.putText(frame, f"Cell Phones: {counts['cell phone']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Laptops: {counts['laptop']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Remotes: {counts['remote']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Refrigerators: {counts['refrigerator']}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Classify detected objects based on their counts
        for det in results.xyxy[0]:
            class_id = int(det[5])
            class_name = yolo_model.names[class_id]
            x1, y1, x2, y2 = map(int, det[:4])

            if class_name == 'cell phone':
                predicted_class = 'Vivo IQ Z6 Lite'
                detection_details['classification'] = predicted_class
                detection_details['probabilities'] = {predicted_class: 1.0}  # Assign 100% probability
                cv2.putText(frame, predicted_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif class_name == 'laptop':
                predicted_class = 'Dell'
                detection_details['classification'] = predicted_class
                detection_details['probabilities'] = {predicted_class: 1.0}  # Assign 100% probability
                cv2.putText(frame, predicted_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif class_name == 'remote':
                predicted_class = 'Onida PXL'
                detection_details['classification'] = predicted_class
                detection_details['probabilities'] = {predicted_class: 1.0}  # Assign 100% probability
                cv2.putText(frame, predicted_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif class_name == 'refrigerator':
                predicted_class = 'Whirlpool 235'
                detection_details['classification'] = predicted_class
                detection_details['probabilities'] = {predicted_class: 1.0}  # Assign 100% probability
                cv2.putText(frame, predicted_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Store the latest frame for image capture
        latest_frame = frame.copy()

        # Convert the frame to RGB and display it
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        video_placeholder.image(image, use_column_width=True)

        # Display object counts
        st.write("Detected Objects:")
        st.write(counts)

    cap.release()

# Capture image button
if st.button("Capture Image"):
    if latest_frame is not None:
        filename = os.path.join('static', 'captured_image.jpg')
        cv2.imwrite(filename, latest_frame)
        st.success('Image captured successfully!')
        st.image(filename, caption='Captured Image', use_column_width=True)
        st.json(detection_details)
    else:
        st.error('No frame available to capture!')

if __name__ == '__main__':
    st.sidebar.title("Settings")
    st.sidebar.write("Adjust settings as needed.")
