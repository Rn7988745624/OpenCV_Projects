import cv2
from cv2 import cvtColor
# Video data
video_file = cv2.VideoCapture('Car_and_Pedestrain.mp4')
# pre-trained data
car_classifier_file = 'haarcascade_car.xml'
pedestrain_classifier_file = 'haarcascade_fullbody.xml'

# making a classifier
car_classifier = cv2.CascadeClassifier(car_classifier_file)
pedestrain_classifier = cv2.CascadeClassifier(pedestrain_classifier_file)
while True:
    frame_success, frame = video_file.read()
    if frame_success:
        # This will chack whether the frame is provided or not
        grey_frame = cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    # Making a detector
    car_coordinates = car_classifier.detectMultiScale(grey_frame, scaleFactor = 1.7, minNeighbors = 10)
    pedestrain_coordinates = pedestrain_classifier.detectMultiScale(grey_frame)
    for (x, y, w, h) in car_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    for (x, y, w, h) in pedestrain_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
    cv2.imshow("Car and Pedestrain Detector", frame)
    key = cv2.waitKey(1)
    if key==27 or key==113:
        break
video_file.release()














"""
FOR IMAGE

# our data
img_file = 'img.jpeg'
# pre-trained data
classifier_file = 'haarcascade_car.xml'

# Image in opencv format
img = cv2.imread(img_file)


# Converting color image into grey image to save time
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# making a classifier
car_classifier = cv2.CascadeClassifier(classifier_file)


# Making a detector
car_coordinates = car_classifier.detectMultiScale(grey_img)

for (x, y, w, h) in car_coordinates:
    cv2.rectangle(img, (x, y), (x+h, y+w), (0, 255, 0), 3)
cv2.imshow('car detector', img)
cv2.waitKey()

"""