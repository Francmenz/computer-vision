import cv2 as cv
video = cv.VideoCapture(0)
while True:
    boolean,frame = video.read()
    if boolean == True:
        haar = cv.CascadeClassifier('haarfacecascde.xml')
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray,\
                scaleFactor=1.1,minNeighbors=12)
        for (x,y,w,h)in faces:
            cv.rectangle(frame,(x,y),(x+w,y+h),\
                (0,255,0), thickness=3)
            cv.putText(cv.rectangle(frame,(x,y),(x+w,y+h),\
                (0,255,0), thickness=3),"Face Detected",(20,20),cv.FONT_HERSHEY_TRIPLEX,2.1,(0,0,255))
            cv.imshow("myface",frame)
        if cv.waitKey(4000) & 0xFF == ord("g"):
            break
video.release
cv.destroyAllWindows()