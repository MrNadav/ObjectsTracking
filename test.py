import cv2
import numpy as np
import threading
import queue
import serial
import time

# Load MobileNet SSD
net = cv2.dnn.readNetFromCaffe('MobileNet-SSD-master/deploy.prototxt', 'MobileNet-SSD-master/mobilenet_iter_73000.caffemodel')

# Initialize video capture
stream_url = 'http://192.168.189.182/stream'
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera couldn't Access!!!")
    exit()

ser = serial.Serial('COM3', 115200)  # Adjust to your ESP32's COM port
time.sleep(2)  # Wait for the connection to establish

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=False)

frameQueue = queue.Queue(maxsize=2)

def capture_thread(cap, frameQueue):
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if not frameQueue.full():
            frameQueue.put(frame)

thread = threading.Thread(target=capture_thread, args=(cap, frameQueue))
thread.start()

servoPos = [90, 90]  # Initial servo position
command_delay = 0.1  # Delay in seconds between servo commands
last_command_time = time.time()

while True:
    if not frameQueue.empty():
        img = frameQueue.get()

        # Background subtraction
        fgmask = fgbg.apply(img)
        kernel = np.ones((3,3), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_area = 0
        target = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > largest_area:
                x, y, w, h = cv2.boundingRect(contour)
                roi = img[y:y+h, x:x+w]
                blob = cv2.dnn.blobFromImage(cv2.resize(roi, (300, 300)), 0.007843, (300, 300), 127.5)
                net.setInput(blob)
                detections = net.forward()

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.4:
                        largest_area = area
                        target = (x + w//2, y + h//2, confidence, int(detections[0, 0, i, 1]))

        if target:
            x, y, _, class_id = target
            if time.time() - last_command_time > command_delay:
                # Calculate servo positions
                servoX = np.interp(x, [0, 800], [180, 0])
                servoY = np.interp(y, [0, 800], [0, 180])
                command = f"ServoX{int(servoX)}ServoY{int(servoY)}\n"
                ser.write(command.encode())
                last_command_time = time.time()

                # Visualization
                cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(img, f"ID: {class_id}", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
ser.close()
