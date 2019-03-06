import cv2
import numpy as np
import glob

def harris(frame):
    gray = np.float32(frame)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.7*dst.max(),255,0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    return(corners)
    
def detect(file):
    cap = cv2.VideoCapture(file)
    ret, frame1 = cap.read()
    no_frames= cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fgbg=cv2.createBackgroundSubtractorMOG2()
    count=0
    while(count<no_frames-3):
        ret, frame2 = cap.read()
        #converting to grayscale
        first_gray=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        last_gray=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        #applying background subtraction
        first_gray=fgbg.apply(first_gray)
        last_gray=fgbg.apply(last_gray)
        #applying harris corner
        h1=harris(first_gray)
        h2=harris(last_gray)
        count=count+1
        h1=list(h1)
        h2=list(h2)
        while(len(h1)<7):
            h1.append([0, 0])
        while(len(h2)<7):
            h2.append([0, 0])
        
    
        
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    path_in = "D:/Major Project/tv_human_interactions_videos/Handshake/*.avi"
    detect(path_in)
    for fname in glob.glob(path_in):
        detect(fname)
        print(fname)
