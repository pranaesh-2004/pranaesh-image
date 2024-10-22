import cv2
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, Response, request, jsonify
from torchvision import models
import torch.nn as nn
import os

app = Flask(__name__)

# Load the YOLOv5 model (for object detection)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
target_classes = ['cell phone', 'laptop', 'remote', 'refrigerator']  # Added 'refrigerator' for home appliances

# Define the class names for classification
class_names = ['Apple iPhone', 'Vivo IQ Z6 Lite', 'Dell', 'Onida PXL', 'Whirlpool 235']  # Added 'Whirlpool 235'

# Initialize video capture (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Store the latest captured frame
latest_frame = None
detection_details = {}

def count_objects(results, class_names):
    """Count the number of detected objects for each class."""
    counts = {class_name: 0 for class_name in class_names}
    for det in results.xyxy[0]:  # Each detection contains [x1, y1, x2, y2, confidence, class_id]
        class_id = int(det[5])
        class_name = yolo_model.names[class_id]
        if class_name in class_names:
            counts[class_name] += 1
    return counts

def generate_frames():
    global latest_frame, detection_details
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Perform object detection using YOLOv5
        results = yolo_model(frame)
        results.render()  # Draw bounding boxes on the frame
        counts = count_objects(results, target_classes)

        # Update detection details for UI display
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

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_image', methods=['POST'])
def capture_image():
    if latest_frame is not None:
        filename = os.path.join('static', 'captured_image.jpg')
        cv2.imwrite(filename, latest_frame)
        return jsonify({'message': 'Image captured successfully!', 'image_url': filename, 'details': detection_details})
    else:
        return jsonify({'message': 'No frame available to capture!'})

if __name__ == '__main__':
    app.run(debug=True)
