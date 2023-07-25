# Face-Recognition-system
This is a computer vision project that utilizes face recognition technology to identify and log individuals in a live video stream. The system is capable of detecting and recognizing faces in real-time, comparing them with a pre-trained dataset of known faces, and logging the recognized individuals along with the date and time of the detection.

To use the provided code for real-time face recognition and logging, follow these steps:

1. Install Required Libraries using these commands:

pip install opencv-python
pip install face-recognition

2. Prepare the Pre-Trained Data:
Place the "haarcascade_frontalface_default.xml" file in the same directory as the Python script. This file is used for facial features detection.

3. Prepare Known Faces:
Before running the code you need to prepare images of known faces that you want the system to recognize. Make sure to name the images meaningfully and update the "known_face_names" list accordingly.

4. Run the Code:
Run the Python script. It will open a live video stream using your computer's camera . The script will start detecting faces in real-time and data will get saved in a csv file.
