# Drive Safe - a driver fatigue detection app 

Detects fatigue driving using OpenCV, dlib (face recognition and facial landmark detection), imutils (image processing).   

---
## Supported fatigue symptoms:

- [x] Slow blinking or struggles to keep eyes open
- [x] Nodding off or struggles to keep head up
- [x] Yawning


---
## Installation
- Python 3.9.10
- CMake 3.22.1
- Install requirements: ```pip install -r requirements.txt```
---
## Usage
```
python monitor.py
    -p / --face_landmark_predictor [path to shape_predictor_68_face_landmarks.dat]
    -v / --verbose [turns on verbose mode]
    -i / --input [optional path to input video file]
```
Note: By providing an optional input video, the app will run detection on the provided video instead of a live camera feed. 
### GUI usage
Press 'p' to pause or continue. Press 'q' to exit the app.

---
### Credit
- [Real-Time Eye Blink Detection using Facial Landmarks](https://www.semanticscholar.org/paper/Real-Time-Eye-Blink-Detection-using-Facial-Soukupov%C3%A1-Cech/4fa1ba3531219ca8c39d8749160faf1a877f2ced) by Tereza Soukupová and Jan Cech
- [68-point facial landmarks model](https://github.com/davisking/dlib-models#shape_predictor_68_face_landmarksdatbz2) by Davis E. King.