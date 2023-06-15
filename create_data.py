import numpy as np 
import cv2 as cv 
import os 
from random import randint  

os.mkdir("C:/Users/admin/Downloads/Toan/train") #Tạo folder
os.mkdir("C:/Users/admin/Downloads/Toan/test")

directory = r'C:\Users\admin\Downloads\Toan\data_train'
label=[]

pathvideo=[]
pathfolder=[]
indexfolder=0

for i in os.listdir(directory):
    label.append(i)
    path1 = os.path.join(directory,i)
    pathfolder.append(path1)
    for j in os.listdir(path1):
        path2 = os.path.join(path1,j)
        pathvideo.append(path2)
for i in pathvideo:
    os.mkdir("C:/Users/admin/Downloads/Toan/train/"+str(label[indexfolder]))
    capture = cv.VideoCapture(i)
    haar_face = cv.CascadeClassifier()
    cnt=0
    while(1):
        isTrue,img = capture.read()
        cv.imshow('Video',img)
        if 50<=cnt<=250:
            cv.imwrite("C:/Users/admin/Downloads/Toan/train/"+str(label[indexfolder])+"/"+str(label[indexfolder])+str(cnt)+'.jpg', img)
        elif cnt<50:
            cv.imwrite("C:/Users/admin/Downloads/Toan/test/"+str(randint(0,9))+str(randint(0,9))+str(randint(0,9))+str(randint(0,9))+'.jpg', img)
        cnt=cnt+1
        if cnt>=250:
            indexfolder = indexfolder +1
            break
        elif cv.waitKey(1)==27:
            break 