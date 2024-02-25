# import cv2
# import numpy as np
# import threading
# import queue

# # Load YOLOv4-tiny model
# net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use CPU

# # Initialize video capture
# cap = cv2.VideoCapture(0)
# ws, hs = 800, 600  # Width and height for the camera resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, ws)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hs)

# if not cap.isOpened():
#     print("Camera couldn't be accessed!!!")
#     exit()

# # Frame queue
# frameQueue = queue.Queue(maxsize=2)

# def capture_thread(cap, frameQueue):
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break
#         if not frameQueue.full():
#             frameQueue.put(frame)

# thread = threading.Thread(target=capture_thread, args=(cap, frameQueue))
# thread.start()

# # Get the names of the output layers for YOLO
# layer_names = net.getLayerNames()
# output_layers_indices = net.getUnconnectedOutLayers().flatten()  # Ensure it's a flat array
# output_layers = [layer_names[i - 1] for i in output_layers_indices]

# # Initialize background subtractor (if you still need it for your specific logic)
# fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# # Initialize servo positions
# servoPos = [90, 90]  # initial servo position

# while True:
#     if not frameQueue.empty():
#         img = frameQueue.get()

#         # Apply background subtraction
#         fgmask = fgbg.apply(img)
#         # Optional: apply some morphological operations to clean up the noise
#         kernel = np.ones((5,5),np.uint8)
#         fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

#         # Convert frame to blob for YOLOv4-tiny
#         blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
#         net.setInput(blob)
#         outs = net.forward(output_layers)

#         found = False
#         for out in outs:
#             for detection in out:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]
#                 if confidence > 0.4:  # Confidence threshold
#                     center_x = int(detection[0] * ws)
#                     center_y = int(detection[1] * hs)
#                     w = int(detection[2] * ws)
#                     h = int(detection[3] * hs)

#                     # Calculate coordinates for the bounding box
#                     x = int(center_x - w / 2)
#                     y = int(center_y - h / 2)
#                     fx, fy = x + w // 2, y + h // 2

#                     servoX = np.interp(fx, [0, ws], [0, 180])
#                     servoY = np.interp(fy, [0, hs], [0, 180])
#                     servoPos = [servoX, servoY]

#                     # Draw detection as per the original logic
#                     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                     cv2.putText(img, f"Moving Object ID: {class_id}", (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

#                     # Assuming you want to draw crosshair and "TARGET LOCKED" text as before
#                     cv2.line(img, (0, fy), (ws, fy), (0, 0, 0), 2)
#                     cv2.line(img, (fx, hs), (fx, 0), (0, 0, 0), 2)
#                     cv2.circle(img, (fx, fy), 15, (0, 0, 255), cv2.FILLED)
#                     cv2.putText(img, "TARGET LOCKED", (500, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

#                     found = True
#                     break  # Exit after first detection for simplicity

#         if not found:
#             # If no object detected
#             cv2.putText(img, "NO TARGET", (880, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

#         # Update servo position texts
#         cv2.putText(img, f'Servo X: {int(servoPos[0])} deg', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
#         cv2.putText(img, f'Servo Y: {int(servoPos[1])} deg', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

#         cv2.imshow("Image", img)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



import cv2
import numpy as np
import threading
import queue
import serial
import time
# Load MobileNet SSD
net = cv2.dnn.readNetFromCaffe('MobileNet-SSD-master/deploy.prototxt', 'MobileNet-SSD-master/mobilenet_iter_73000.caffemodel')

# Initialize video capture
cap = cv2.VideoCapture(0)
ws, hs = 800, 800  # Width and height of the camera resolution
cap.set(3, ws)
cap.set(4, hs)

if not cap.isOpened():
    print("Camera couldn't Access!!!")
    exit()

ser = serial.Serial('COM3', 115200)  # Adjust 'COM7' to your ESP32's COM port
time.sleep(2)  # Wait for the connection to establish

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=False)

# Frame queue
frameQueue = queue.Queue(maxsize=2)

def capture_thread(cap, frameQueue):
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Resize frame for faster processing

        if not frameQueue.full():
            frameQueue.put(frame)

# Start capture thread
thread = threading.Thread(target=capture_thread, args=(cap, frameQueue))
thread.start()

# Initialize servo positions
servoPos = [90, 90]  # initial servo position

while True:
    if not frameQueue.empty():
        img = frameQueue.get()

        # Apply background subtraction
        fgmask = fgbg.apply(img)
        # Optional: apply some morphological operations to clean up the noise
        kernel = np.ones((5,5),np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # Find contours in the foreground mask
        contours, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found = False
        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Filter out small movements
                continue

            # Get bounding box of the contour, detect object within this region
            x, y, w, h = cv2.boundingRect(contour)
            roi = img[y:y+h, x:x+w]

            blob = cv2.dnn.blobFromImage(cv2.resize(roi, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.4:  # Threshold for detection
                    found = True
                    class_id = int(detections[0, 0, i, 1])
                    fx, fy = x + w//2, y + h//2

                    servoX = np.interp(fx, [0, ws], [180, 0])  # Flips direction for X servo
                    servoY = np.interp(fy, [0, hs], [0, 180])  # Keeps original direction for Y servo

                    servoX = max(0, min(180, servoX))
                    servoY = max(0, min(180, servoY))
                    
                    command = f"ServoX{int(servoX)}ServoY{int(servoY)}\n"
                    ser.write(command.encode())

                    # Draw detection and crosshair as before
                    cv2.circle(img, (fx, fy), 80, (0, 0, 255), 2)
                    cv2.putText(img, f"Moving Object ID: {class_id}", (fx + 15, fy - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    cv2.line(img, (0, fy), (ws, fy), (0, 0, 0), 2)
                    cv2.line(img, (fx, hs), (fx, 0), (0, 0, 0), 2)
                    cv2.circle(img, (fx, fy), 15, (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, "TARGET LOCKED", (500, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                    break  # Track the first detected moving object for simplicity

            if found:
                break  # Stop searching once we've found a moving object

        if not found:
            # If no moving object detected
            cv2.putText(img, "NO TARGET", (880, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

        cv2.putText(img, f'Servo X: {int(servoPos[0])} deg', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(img, f'Servo Y: {int(servoPos[1])} deg', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
ser.close()