import cv2
import dlib
import numpy as np
import time
from djitellopy import Tello

# Load Dlib's face detector
detector = dlib.get_frontal_face_detector()

# Load Dlib's face recognition model
shape_predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Load image of the person to be followed
img = cv2.imread('person.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)

if len(rects) > 0:
    shape = shape_predictor(gray, rects[0])
    face_descriptor1 = face_recognition_model.compute_face_descriptor(img, shape)

# Connect and initialize the Tello drone
tello = Tello()
tello.connect()
tello.streamon()

# Give the drone a moment to initialize
time.sleep(2)

# Variables to manage photo capture
last_capture = time.time()
capture_interval = 60  # Capture a photo every 60 seconds

# Main loop
while True:
    # Get the current frame from the Tello camera
    frame = tello.get_frame_read().frame

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    rects = detector(gray, 1)

    # If a face is detected
    for rect in rects:
        shape = shape_predictor(gray, rect)
        face_descriptor2 = face_recognition_model.compute_face_descriptor(frame, shape)
        distance = np.linalg.norm(np.array(face_descriptor1) - np.array(face_descriptor2))

        # If this face matches the person to be followed
        if distance < 0.6:  # threshold, adjust based on your requirements
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Calculate the center of the face
            center_x = x + w // 2
            center_y = y + h // 2

            # Move the drone based on the location of the face
            if center_x < frame.shape[1] // 2:
                tello.move_left(20)
            else:
                tello.move_right(20)

            if center_y < frame.shape[0] // 2:
                tello.move_up(20)
            else:
                tello.move_down(20)

            # If it's time to capture a photo
            if time.time() - last_capture > capture_interval:
                # Take a photo and save it with a timestamp
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(f"{timestamp}.jpg", frame)
                last_capture = time.time()

            break
    else:
        # If no faces were detected, just hover
        tello.send_rc_control(0, 0, 0, 0)

    # Show the image with OpenCV
    cv2.imshow('Tello Tracking', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and land the drone
tello.land()
cv2.destroyAllWindows()
