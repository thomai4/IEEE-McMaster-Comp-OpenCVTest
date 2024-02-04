import cv2  

#based on code from: https://www.geeksforgeeks.org/detect-cat-faces-in-real-time-using-python-opencv/
#classifiers found at https://github.com/opencv/opencv/tree/master/data/haarcascades
#catFace_cascade = cv2.CascadeClassifier('data/haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
humanFace_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

# capture frames from a camera  
cap = cv2.VideoCapture(0)

#uncomment to capture still frame
#img = cv2.imread('data/5.jpg')  
  
# loop runs if capturing has been initialized.  
while 1:  
  
    # reads frames from a camera  
    ret, img = cap.read()  
  
    # convert to gray scale of each frames  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

    # Detects faces of different sizes in the input image  
    faces = humanFace_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:  
        # To draw a rectangle in a face  
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
        roi_gray = gray[y:y+h, x:x+w]  
        roi_color = img[y:y+h, x:x+w]
    for (x,y,w,h) in eyes:  
        # To draw a rectangle in a face  
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)  
        roi_gray = gray[y:y+h, x:x+w]  
        roi_color = img[y:y+h, x:x+w]
    
    cv2.imshow('img',img)  

    k = cv2.waitKey(30) & 0xff #if escape key detected
    if k == 27:  
        break
# Close the window  
cap.release()    
# De-allocate any associated memory usage  
cv2.destroyAllWindows()  