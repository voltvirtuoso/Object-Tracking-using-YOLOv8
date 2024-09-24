import os
import urllib.request
import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
from tqdm import tqdm
import random

# Define the path to the YOLOv8 weights file and the URL to download it
weights_path = "yolov8s.pt"
weights_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt"  # YOLOv8s weights URL

def download_weights(url, destination):
    """Download weights if they do not exist with a progress bar."""
    if not os.path.exists(destination):
        print(f"Weights not found. Attempting auto download.")
        print(f"Downloading weights from {url}...")
        # Get the file size
        with urllib.request.urlopen(url) as response:
            total_size = int(response.info().get('Content-Length', 0))
        # Download the file with progress bar
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            def report_progress(block_num, block_size, total_size):
                downloaded_size = block_num * block_size
                pbar.update(downloaded_size - pbar.n)
            urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print("Download complete.")

# Download YOLOv8 weights if not present
download_weights(weights_url, weights_path)

# Load YOLOv8 model
model = YOLO(weights_path)

# Initialize SORT tracker
tracker = Sort()

# Set the confidence threshold for object detection
CONFIDENCE_THRESHOLD = 0.1  # Adjust this value based on your requirements

def detect_objects(frame):
    """Detect objects in the frame using YOLOv8."""
    # Run inference
    results = model(frame)
    
    # Extract detection results
    detections = []
    for result in results:
        # Extract boxes, scores, and class ids
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        
        # Combine results into a list of dictionaries, filtering by confidence threshold
        for box, score, class_id in zip(boxes, scores, class_ids):
            if score >= CONFIDENCE_THRESHOLD:
                detections.append({
                    'xmin': box[0],
                    'ymin': box[1],
                    'xmax': box[2],
                    'ymax': box[3],
                    'confidence': score,
                    'class_id': int(class_id)
                })
    
    return detections

def track_vehicles(frame, detected_objects):
    """Track vehicles using SORT tracker."""
    detections = []
    
    for obj in detected_objects:
        # Include all detected classes (adjust the logic if you want to filter specific classes)
        x1, y1, x2, y2 = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
        detections.append([x1, y1, x2, y2, obj['confidence']])
    
    # Convert detections to numpy array if there are any detections
    if detections:
        detections = np.array(detections)
    else:
        detections = np.empty((0, 5))
    
    # Update tracker
    tracked_objects = tracker.update(detections)
    
    return tracked_objects

def get_unique_color(track_id):
    """Generate a unique color for each track_id."""
    random.seed(track_id)  # Ensure reproducibility
    return tuple(random.randint(0, 255) for _ in range(3))

def resize_with_max_dimension(image, max_dimension):
    """Resize the image such that the largest dimension equals the max_dimension, keeping aspect ratio intact."""
    height, width = image.shape[:2]
    if max_dimension is None:
        return image

    # Determine the scaling factor based on the larger dimension
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return resized_image
    else:
        # No resizing needed if the largest dimension is already smaller than max_dimension
        return image

ip_url = 'http://192.168.43.1:8080/video'
cap_toll = cv2.VideoCapture(ip_url)  # Replace with your camera stream

# Dictionary to keep track of object paths
object_paths = {}

max_dimension = 720  # Set the max dimension for resizing

while True:
    ret, frame_toll = cap_toll.read()
    
    if not ret:
        break

    # Resize the frame to the max dimension while keeping the aspect ratio
    frame_toll = resize_with_max_dimension(frame_toll, max_dimension)

    detected_objects_toll = detect_objects(frame_toll)
    tracked_objects = track_vehicles(frame_toll, detected_objects_toll)

    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
        confidence = obj[4]  # Use confidence for drawing
        
        # Get or generate a unique color for the track_id
        color = get_unique_color(track_id)
        
        # Draw the bounding box
        cv2.rectangle(frame_toll, (x1, y1), (x2, y2), color, 2)
        
        # Draw the track ID and confidence score
        label = f'ID: {track_id}, Conf: {confidence:.2f}'
        cv2.putText(frame_toll, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Update the object paths
        if track_id not in object_paths:
            object_paths[track_id] = []
        object_paths[track_id].append((x1, y1))
        
        # Draw the path
        path = object_paths[track_id]
        if len(path) > 1:
            for i in range(1, len(path)):
                cv2.line(frame_toll, path[i - 1], path[i], color, 2)
    
    cv2.imshow("Camera", frame_toll)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_toll.release()
cv2.destroyAllWindows()
