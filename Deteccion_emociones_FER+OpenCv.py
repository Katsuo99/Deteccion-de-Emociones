from fer import FER
import cv2
 
detector = FER() 
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
frame = cv2.imread("happy.jpg")
emotion, score = detector.top_emotion(frame)
print(emotion)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
auxFrame = gray.copy()
faces = faceClassif.detectMultiScale(gray,1.3,5)

for (x,y,w,h) in faces:
    rostro = auxFrame[y:y+h,x:x+w]
    rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
    result = emotion
    cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
    cv2.imwrite("foto.png", frame)
