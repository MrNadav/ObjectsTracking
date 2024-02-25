import cv2
from cvzone.FaceDetectionModule import FaceDetector
import serial
import numpy as np
import time
stream_url = 'http://192.168.189.182/stream'
cap = cv2.VideoCapture(stream_url)

ws, hs = 1280, 720
cap.set(3, ws)
cap.set(4, hs)

if not cap.isOpened():
    print("Camera couldn't Access!!!")
    exit()

ser = serial.Serial('COM3', 115200)  # Adjust 'COM7' to your ESP32's COM port
time.sleep(2)  # Wait for the connection to establish

# port = "COM7"
# board = pyfirmata.Arduino(port)
# servo_pinX = board.get_pin('d:9:s') #pin 9 Arduino
# servo_pinY = board.get_pin('d:10:s') #pin 10 Arduino

detector = FaceDetector()
servoPos = [90, 90] # initial servo position

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img, draw=False)

    if bboxs:
        #get the coordinate
        fx, fy = bboxs[0]["center"][0], bboxs[0]["center"][1]
        pos = [fx, fy]
        #convert coordinat to servo degree
        servoX = np.interp(fx, [0, ws], [180, 0])  # Notice the swapped positions of 180 and 0
        servoY = np.interp(fy, [0, hs], [0, 180])
        # Clamp values to servo limits
        servoX = max(0, min(180, servoX))
        servoY = max(0, min(180, servoY))
        
        command = f"ServoX{int(servoX)}ServoY{int(servoY)}\n"
        ser.write(command.encode())

        cv2.circle(img, (fx, fy), 80, (0, 0, 255), 2)
        cv2.putText(img, str(pos), (fx+15, fy-15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2 )
        cv2.line(img, (0, fy), (ws, fy), (0, 0, 0), 2)  # x line
        cv2.line(img, (fx, hs), (fx, 0), (0, 0, 0), 2)  # y line
        cv2.circle(img, (fx, fy), 15, (0, 0, 255), cv2.FILLED)
        cv2.putText(img, "TARGET LOCKED", (850, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3 )

    else:
        cv2.putText(img, "NO TARGET", (880, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        cv2.circle(img, (640, 360), 80, (0, 0, 255), 2)
        cv2.circle(img, (640, 360), 15, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (0, 360), (ws, 360), (0, 0, 0), 2)  # x line
        cv2.line(img, (640, hs), (640, 0), (0, 0, 0), 2)  # y line


    cv2.putText(img, f'Servo X: {int(servoPos[0])} deg', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(img, f'Servo Y: {int(servoPos[1])} deg', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
ser.close()