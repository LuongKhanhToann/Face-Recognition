import cv2 as cv
import numpy as np
import os

def rescale(img, scale):
    w = img.shape[1]
    h = img.shape[0]
    dim = ((int)(w*scale),(int)(h*scale))
    return cv.resize(img, dim, cv.INTER_AREA)

#call detect face function
haar_face = cv.CascadeClassifier("C:/Users/admin/Downloads/Toan/haar_face.xml")
people = []
dir=r"C:/Users/admin/Downloads/Toan/train"
features=[]
labels=[]
for i in os.listdir(dir):
    people.append(i)
    
for i in people:
    path = os.path.join(dir,i)
    label = people.index(i)

    for j in os.listdir(path):
        img_path = os.path.join(path,j)
        img = cv.imread(img_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        face_rect =  haar_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

        for(x,y,w,h) in face_rect:
            cv.rectangle(img, (x+20,y-20), (x+w-20,y+h+20),(250,0,0),thickness=2)
            face_roi = gray[y:y+h, x:x+w]
            features.append(face_roi)
            labels.append(label)
            cv.putText(img,'Traing: '+str(len(features)),(20,40),cv.FONT_HERSHEY_COMPLEX,1,(0,0,1),2)
            cv.putText(img,str(people[label]),(x+20,y-30),cv.FONT_HERSHEY_COMPLEX,1,(250,0,0),2)
        cv.imshow('Picture', img)
        cv.waitKey(2)

print('Number of Features: '+str(len(features))+'---- Number of Labels: '+ str(len(labels)))
features = np.array(features, dtype='object')
labels = np.array(labels)
face_regconizer = cv.face.LBPHFaceRecognizer_create()
face_regconizer.train(features, labels) 

face_regconizer.save('C:\\Users/Admin/Downloads/toan/face_train.yml')
np.save("C:\\Users/Admin/Downloads/toan/features.npy",features)
np.save("C:\\Users/Admin/Downloads/toan/labels.npy",labels)