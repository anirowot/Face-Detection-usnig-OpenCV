import cv2
import os
import numpy as np
import faceRecognition as fr

test_img = cv2.imread(r'C:\Users\user\Desktop\Python_Project\Test Images\test2.jpg')
faces_detected,gray_img = fr.faceDetection(test_img)
print("faces_detected:", faces_detected)

#for (x,y,w,h) in faces_detected:
#cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)
# resized_img = cv2.resize(test_img,(600,500))
# cv2.imshow("Face Detection", resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows

# faces,faceID = fr.labels_for_training_data(r'C:\Users\user\Desktop\Python_Project\Training Images')
# face_recognizer = fr.train_classifier(faces,faceID)
# face_recognizer.save('trainingData.yml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'C:\Users\user\Desktop\Python_Project\trainingData.yml')

name = {0:"SRK", 1:"Himanshu"}

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+h,x:x+h]
    label,confidence = face_recognizer.predict(roi_gray)
    print("Confidence:", confidence)
    print("Label:", label)
    fr.draw_rect(test_img,face)
    predicted_name = name[label]
    if(confidence>45):
        continue
    fr.put_text(test_img,predicted_name,x,y)

resized_img = cv2.resize(test_img,(600,500))
cv2.imshow("Face Detection", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows