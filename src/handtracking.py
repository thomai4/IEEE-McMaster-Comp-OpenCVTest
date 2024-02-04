#based on code from: https://github.com/akpythonyt/Computer-vision/blob/69dce7b96a0ffc457a17cb31926077e0d9ef67a2/Handtracker2.py#L9
#library documentation: https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer

#Required things
#1.Mediapipe
#2.OpenCV
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
hands=mp_hands.Hands()
while True:
    success, image = cap.read()
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    results = hands.process(image)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:                                                                         
        mp_drawing.draw_landmarks(                                                 
            image,
            hand_landmarks,mp_hands.HAND_CONNECTIONS)
    cv2.imshow('MediaPipe Hands', image)
    k = cv2.waitKey(30) & 0xff #if escape key detected
    if k == 27:
        break
# Close the window  
cap.release()    
# De-allocate any associated memory usage  
cv2.destroyAllWindows()   