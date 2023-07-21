import cv2
import time
from djitellopy import Tello

# Load Haarcascade face classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Connect and initialize the Tello drone
tello = Tello()
tello.connect()
tello.streamon()

# Give the drone a moment to initialize
time.sleep(2)

# Initialize face tracking variables
face_start_time = None
face_end_time = None
face_detected = False

# Main loop
while True:
    # Get the current frame from the Tello camera
    frame = tello.get_frame_read().frame

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If a face is detected
    if len(faces) > 0 and not face_detected:
        (x, y, w, h) = faces[0]  # We take the first face detected
        face_detected = True
        face_start_time = time.time()

        # Take a photo of the face and save it with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(f"{timestamp}.jpg", frame)

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
    else:
        # If no faces were detected, just hover
        tello.send_rc_control(0, 0, 0, 0)

    # Check if 30 seconds have passed since the drone started following a face
    if face_detected:
        face_end_time = time.time()
        if face_end_time - face_start_time > 30:  # 30 seconds
            face_detected = False

    # Show the image with OpenCV
    cv2.imshow('Tello Tracking', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and land the drone
tello.land()
cv2.destroyAllWindows()
