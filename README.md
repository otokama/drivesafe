# Drive Safe - a driver fatigue detection app 

Detects fatigue driving using OpenCV, dlib (face recognition and facial landmark detection), imutils (image processing).   
Supported symptoms:
- Slow blinking or struggles to keep eyes open
- Yawning
- Nodding off or struggles to keep head up
---
## Installation
- Python 3.9.10
- Install requirements: ```pip install -r requirements.txt```
---
## Usage
```
python monitor.py
    -p / --face_landmark_predictor [path to shape_predictor_68_face_landmarks.dat]
    -v / --verbose [turns on verbose mode]
    -i / --input [optional input video file relative path]
```
---
### Credit
[68-point facial landmarks model](https://github.com/davisking/dlib-models#shape_predictor_68_face_landmarksdatbz2) by Davis E. King.
