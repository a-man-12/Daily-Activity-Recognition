import cv2
import numpy as np
import glob
import math
import csv
    
def sift(frame):
    sift=cv2.xfeatures2d.SIFT_create()
    kp=sift.detect(frame,None)
    corners=[]
    for i in range(10):
        x,y=kp[i].pt
        corners.append([x,y])
    return corners

def detect(file):
    cap = cv2.VideoCapture(file)
    ret, frame1 = cap.read()
    no_frames= cap.get(cv2.CAP_PROP_FRAME_COUNT)
    count=0
    avg=[0 for x in range(10)]
    while(count<no_frames-3):
        ret, frame2 = cap.read()
        #converting to grayscale
        first_gray=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        last_gray=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        #applying sift
        h1=sift(first_gray)
        h2=sift(last_gray)
        frame1=frame2
        count=count+1
        f=[0 for x in range(10)]
        #calculating flow
        for i in range(0,10):
            a=pow((h1[i][0]-h2[i][0]),2)+pow((h1[i][1]-h2[i][1]),2)
            f[i]=math.sqrt(a)
            avg[i]=avg[i]+f[i]
        
    for i in range(0,10):
        avg[i]=avg[i]/count
    with open('test.csv','a',newline='') as csvFile:
        writer=csv.writer(csvFile)
        writer.writerow(avg)
    csvFile.close()
    #cv2.imwrite("frame.jpg",last_gray)
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    path_in = "D:/Major Project/tv_human_interactions_videos/Others/*.avi"
    for fname in glob.glob(path_in):
        detect(fname)
        print(fname)
    #detect(path_in)
