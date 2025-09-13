import cv2 as cv
import numpy as np

model=cv.dnn.readNetFromCaffe('ssd_deploy.prototxt','ssd_weights.caffemodel')

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

cap=cv.VideoCapture(0)

while True:
    success,img=cap.read()

    if not success:
        break
    
    (h,w)=img.shape[:2]

    blob=cv.dnn.blobFromImage(cv.resize(img,(300,300)),0.007843,(300,300),127.5)

    model.setInput(blob)

    detect=model.forward()

    for i in range(detect.shape[2]):
        confidence=detect[0,0,i,2]

        if confidence>.2:
            idx = int(detect[0, 0, i, 1])
            box = detect[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASSES[idx]}"
            cv.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv.putText(img, label, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv.imshow('webcam',img)

    if cv.waitKey(1) & 0XFF==ord('q'):
        break

cap.release()
cv.destroyAllWindows()