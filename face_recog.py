# import packages
import cv2
import face_recognition
import numpy as np
import csv
import os
from datetime import datetime

# loaded pretrained dataset
trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# take video input
video = cv2.VideoCapture(0)

# load image file
one_img = face_recognition.load_image_file(''' put the image path''')
# encoding [raw data]
nss_encoding = face_recognition.face_encodings(one_img)[0]

two_img = face_recognition.load_image_file(''' put other image's path''')
nss2_encoding = face_recognition.face_encodings(two_img)[0]

# list for encoding we have
known_face_encoding = [nss_encoding, nss2_encoding]

# list for names we know
known_face_names = ["one", "two"]

person = known_face_names.copy()

# declare some variables
face_location = []
face_encoding = []
face_names = []
s = True

# current date time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# create a csv file with today's date
f = open(current_date + '.csv', 'w+', newline='')
log_writer = csv.writer(f)

# while loop
while True:
    _, frame = video.read()
    # convert greyscale
    grey_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect face coordinates
    face_cord = trained_data.detectMultiScale(grey_img)

    # making a rectangle or square on face format--(image, coordinate top, cord bottom, color, thickness)
    for (x, y, w, h) in face_cord:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 153), 2)
    # resize the frame we got
    resized_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_resize = resized_frame[:, :, ::-1]
    if s == True:
        # location of face
        face_locations = face_recognition.face_locations(rgb_resize)
        # face encodings
        face_encodings = face_recognition.face_encodings(rgb_resize, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # comparing faces and saving the data
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            match_got = np.argmin(face_distance)
            if matches[match_got]:
                name = known_face_names[match_got]
            face_names.append(name)
            if name in known_face_names:
                person.remove(name)
                current_time = now.strftime("%H-%M-%S")
                log_writer.writerow([name, current_date, current_time])
    # display the frame
    cv2.imshow("face_recognition", frame)
    key = cv2.waitKey(1)
    # we can quit by pressing Q
    if key == 113 or key == 81:
        break
# close and release the resources
video.release()
f.close()
