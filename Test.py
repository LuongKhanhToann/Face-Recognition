import numpy as np 
import cv2 as cv 
import os   

people = ["Chien","Cong","Duc","Hung","Long","PCong","Phuc","Quan","Toan","Trung"]

#Detect face in image
haar_cascade =  cv.CascadeClassifier(r'C:\\Users/Admin/Downloads/Toan/haar_face.xml')

#Reading data training
features = np.load(r"C:\\Users/Admin/Downloads/Toan/features.npy",allow_pickle=True)
labels = np.load(r"C:\\Users/Admin/Downloads/Toan/labels.npy")

face_recognizer = cv.face.LBPHFaceRecognizer_create()#khoi tao model
face_recognizer.read('C:\\Users/Admin/Downloads/Toan/face_train.yml')# goi model

test_folder = r"C:\Users\admin\Downloads\Toan\test"
count = 0
for filename in os.listdir(test_folder):
    count += 1
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        img_path = os.path.join(test_folder, filename)
        img = cv.imread(img_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1)
        
        for(x,y,w,h) in face_rect:
            cv.rectangle(img,(x,y),(x+w,y+h),(0,200,0),thickness=4)
            face_roi = gray[y:y+h,x:x+h]
            label, confidence=face_recognizer.predict(face_roi)
            text = f"{people[label]} - {confidence:.2f}"
            cv.putText(img, text, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if count == 100:
            break
        cv.imshow(f"Image {filename}", img)
        cv.waitKey(0)
        cv.destroyAllWindows()