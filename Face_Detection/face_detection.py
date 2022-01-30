from random import randrange
import cv2

# We call cv2 and makes a classifier(it can classify object into distinct categories)
# haar cascade is the algorithm
train_face_data = cv2.CascadeClassifier('/Users/rohitnarwal/Desktop/AI_Projects/Training_files/haarcascade_frontalface_default.xml')
train_smile_data = cv2.CascadeClassifier('/Users/rohitnarwal/Desktop/AI_Projects/Training_files/haarcascade_smile.xml')
# train_eye_data = cv2.CascadeClassifier('/Users/rohitnarwal/Desktop/AI_Projects/Training_files/haarcascade_eye.xml')
train_eye_data = cv2.CascadeClassifier('/Users/rohitnarwal/Desktop/AI_Projects/Training_files/haarcascade_eye_tree_eyeglasses.xml')
# Importing image for detetion from webcam
webcam = cv2.VideoCapture(0)
# This will make the image capturing through webcam untill we close it
while True:
    # This will return boolean and frame itself
    # frame_success stor boolean about frame captured or not
    # frame stores the actual frame
    frame_success, frame = webcam.read()
    if not frame_success:
        break
    # We need to convert image into single color to save memory and we do it with cvtcolor
    # command
    grey_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = train_face_data.detectMultiScale(grey_img)
    #Drawing rectangle around face
    for x, y, w, h in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)
        cv2.putText(frame, 'Face', (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        in_face_coordinates = frame[y:y+h, x:x+w]
        face_grey_img = cv2.cvtColor(in_face_coordinates, cv2.COLOR_BGR2GRAY)
        # To make algorithm more optimize we can make frames blur to improve it using scaleFactor
        # minNeighbors tell neighboring rectangles
        eye_coordinates = train_eye_data.detectMultiScale(face_grey_img, scaleFactor=1.2, minNeighbors=10)
        smile_coordinates = train_smile_data.detectMultiScale(face_grey_img, scaleFactor=1.8, minNeighbors=30)
        for x_s, y_s, w_s, h_s in smile_coordinates:
            cv2.rectangle(in_face_coordinates, (x_s, y_s), (x_s+w_s, y_s+h_s), (0, 255, 0), 5)
            cv2.putText(in_face_coordinates, 'Smiling', (x_s, y_s+h_s+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        for x_s, y_s, w_s, h_s in eye_coordinates:
            cv2.rectangle(in_face_coordinates, (x_s, y_s), (x_s+w_s, y_s+h_s), (255, 0, 0), 5)
            cv2.putText(in_face_coordinates, 'Eyes', (x_s, y_s+h_s+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    cv2.imshow("Face, Smile, and Eye detection", frame)
    key = cv2.waitKey(1)
    if key==27 or key==113:
        break
# Releases the webcam object
webcam.release()
cv2.destroyAllWindows()
"""

# Face detection for image
face_coordinates = train_data.detectMultiScale(grey_img)
#Drawing rectangle around face
for x, y, w, h in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 5)

# This will show the picture
cv2.imshow("Face detection", img)
# This command will make the picture wait untill the program is closed
cv2.waitKey()

"""