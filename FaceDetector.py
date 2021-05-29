import cv2
import numpy as np

def DetectFace(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original = image.copy()
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    detected_faces = []
    for (x,y,w,h) in faces:
        face = original[y:y+h, x:x+w]
        detected_faces.append(face)
        image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
    return image, detected_faces

if __name__ == '__main__':
    scale = 1
    # image = cv2.imread(r'D:/Rishabh_Pant.jpg')
    image = cv2.imread(r'C:\Users\shashank\Pictures\Shashank.jpeg')
    image = cv2.resize(image,(int(scale*image.shape[1]),int(scale*image.shape[0])),interpolation = cv2.INTER_AREA)
    image, faces = DetectFace(image)
    cv2.imshow('Image', image)
    print(len(faces))
    for face in faces:
        cv2.imshow('Face', face)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()