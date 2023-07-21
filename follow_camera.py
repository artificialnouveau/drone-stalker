import cv2
import time
from djitellopy import Tello

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Connect and initialize the Tello drone
tello = Tello()
tello.connect()
tello.streamon()

# Give the drone a moment to initialize
time.sleep(2)

# Main loop
while True:
    # Get the current frame from the Tello camera
    frame = tello.get_frame_read().frame

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Parse the output of YOLO
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # If a camera is detected
            if class_id == 2 and confidence > 0.5:  # Assume that the class_id of a camera is 2
                # Get the position of the camera
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])

                # Move the drone based on the location of the camera
                if center_x < frame.shape[1] // 2:
                    tello.move_left(20)
                else:
                    tello.move_right(20)

                if center_y < frame.shape[0] // 2:
                    tello.move_up(20)
                else:
                    tello.move_down(20)

                break
    else:
        # If no cameras were detected, just hover
        tello.send_rc_control(0, 0, 0, 0)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and land the drone
tello.land()
