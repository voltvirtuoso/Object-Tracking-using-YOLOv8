# Vehicle Detection and Tracking with YOLOv8 and SORT

### Key Points:
- **Description:** 
  This project uses YOLOv8 for vehicle detection and SORT (Simple Online and Realtime Tracking) for tracking vehicles in a video stream. It detects and tracks vehicles such as cars and trucks, assigning unique colors to each tracked vehicle and visualizing their movement paths.

- **Installation Instructions:**
  1. **Clone the repository:**
     ```bash
     git clone https://github.com/yourusername/vehicle-detection-tracking.git
     cd vehicle-detection-tracking
     ```
  2. **Create a virtual environment:**
     ```bash
     python -m venv .venv
     source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
     ```
  3. **Install required packages:**
     ```bash
     pip install -r requirements.txt
     ```

- **Usage Instructions:**
  1. **Download YOLOv8 weights:**
     The weights file (`yolov8s.pt`) will be automatically downloaded if not present.
  2. **Run the script:**
     ```bash
     python main.py
     ```
     Make sure to replace `"highway.mp4"` with the path to your video file if necessary.

- **Code Description:**
  - `download_weights(url, destination)`: Downloads YOLOv8 weights with a progress bar.
  - `detect_objects(frame)`: Runs YOLOv8 on the input frame to detect objects and returns a list of detections.
  - `track_vehicles(frame, detected_objects)`: Filters detections for vehicles and tracks them using SORT.
  - `get_unique_color(track_id)`: Generates a unique color for each tracked vehicle.
  - **Main Loop:**
    - Reads frames from the video stream.
    - Detects objects and tracks vehicles.
    - Draws bounding boxes, tracking IDs, and movement paths on the frames.
    - Displays the video with visual annotations.

- **Notes:**
  - The code assumes vehicle class IDs are `0`. Adjust the `class_id` filter in `track_vehicles` if your model uses different IDs.
  - The display window will be resized to fit the screen if the input video resolution is too large.

- **License and Contact:**
  - **License:** This project is licensed under the MIT License.


Feel free to modify any sections to better fit your project details or personal preferences!
