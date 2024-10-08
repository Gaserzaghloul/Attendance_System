import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
# from PIL import ImageGrab

path = 'ImagesAttendance'  # Folder where images are stored
images = []  #  store the image data
classNames = []  # store the name of the person from the imagee
myList = os.listdir(path)  # List of all files in the directory
print(myList)
for cl in myList:
    fullPath = f'{path}/{cl}'
    print(f"Loading image from: {fullPath}")
    curImg = cv2.imread(fullPath)  # Read the image from the path
    if curImg is None:
        print(f"Error loading image {fullPath}.")
        continue  # if there's an issue Skip this image and move to the next one
    images.append(curImg)  # Add the image to the list
    classNames.append(os.path.splitext(cl)[0])  # Extract name from the filename
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image from BGR to RGB
        encode = face_recognition.face_encodings(img)[0]  # Encode the face
        encodeList.append(encode)  # Add encoding to the list
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()  # Read the existing attendance
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])  # Extract the names from the attendance file

        if name not in nameList:  # Check if the name is already in the list
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')  # Get the current time
            f.writelines(f'\n{name},{dtString}')  # Write the name and time into the file


#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()  # Capture frame from webcam
    # img = captureScreen()  # Uncomment this line if you are capturing screen instead
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize the image for faster processing
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB

    facesCurFrame = face_recognition.face_locations(imgS)  # locations
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)  # encodings

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  # Compare
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  #  distances
        matchIndex = np.argmin(faceDis)  # best match

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()  # matched name
            # printinnng
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Sscale locations
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # rectangle face
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)  # filled rectangle
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)  # Put name text
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
