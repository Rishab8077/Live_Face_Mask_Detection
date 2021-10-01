import cv2
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model=load_model('Model/model.h5')

img_width,img_height= 200,200

#cascade classifier is opencv pre trained classifier to detect the face
face_cascade= cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")

#to start webcam
cap = cv2.VideoCapture(0) #for webcam (0) and for external camera (1)
#cap = cv2.VideoCapture('file location') this is for detection from video

img_count_full=0

#parameters for text
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (1, 1)
class_lable=' '
# fontScale
fontScale = 1 #0.5
# green color in BGR
color = (0, 255, 0)
# Line thickness of 2 px
thickness = 2 #1

# starting reading images and prediction
while True:
    img_count_full += 1

    # read image from webcam
    responce, image = cap.read()

    if responce == False:
        break
    # resize image with 50%
    # scale = 50
    # width = int(image.shape[1] * scale/100)
    # height = int(image.shape[0] * scale/100)
    # dim=(width,height)

    # resize image
    # image = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
    # conver to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect the faces
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 6)  # 1.1 ,3) for mp4

    # take face then predict wether it identifies wears mask or not
    img_count = 0
    for (x, y, w, h) in faces:
        org = (x - 10, y - 10)
        img_count += 1
        # now extracting the face from original image
        color_face = image[y:y + h, x:x + w]  # color face
        cv2.imwrite('faces/input/%d%dface.jpg' % (img_count_full, img_count), color_face)
        img = load_img('faces/input/%d%dface.jpg' % (img_count_full, img_count), target_size=(img_width, img_height))
        img = img_to_array(img) / 255
        img = np.expand_dims(img, axis=0)  # we changed the dim 3 channel into 4 channel
        pred_prob = model.predict(img)
        # we will get prediction as 2d probablity / print(pred_prob[0][0].round(2))
        pred = np.argmax(pred_prob)  # the probablity value will be between 0 and 1

        # while training when checked the class_indicies we got '0' for with mask and '1' without mask

        if pred == 0:
            print("User with mask - predic = ", pred_prob[0][0])
            class_lable = "Mask"
            color = (0, 255, 0)
            cv2.imwrite('faces/with_mask/%d%dface.jpg' % (img_count_full, img_count), color_face)

        else:
            print('user not wearing mask - prob = ', pred_prob[0][1])
            class_lable = "No Mask"
            color = (0, 0, 255)
            cv2.imwrite('faces/without_mask/%d%dface.jpg' % (img_count_full, img_count), color_face)

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.putText(image, class_lable, org, font, fontScale, color, thickness, cv2.LINE_AA)

        # display window
        cv2.imshow('LIVE Face Mask Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#release the videocapture object
cap.release()
cv2.destroyAllWindows()