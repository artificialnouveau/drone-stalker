import cv2
from djitellopy import Tello

# Function to estimate the height of the tallest person using Haar Cascade classifier
def estimate_tallest_person_height(frame):
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_cascade.detectMultiScale(gray, 1.1, 4)

    tallest_height = 0

    for (x, y, w, h) in bodies:
        person_height = h
        if person_height > tallest_height:
            tallest_height = person_height

    return tallest_height


# Function to control the Tello drone to follow the tallest person
def follow_tallest_person(tello, estimated_height):
    desired_distance = 100  # Desired distance from the tallest person in centimeters
    max_speed = 50  # Maximum speed for the drone's movement in centimeters per second

    current_distance = tello.get_distance_tof()  # Get the current distance from the Tello's ToF sensor

    # Calculate the difference between the current distance and the desired distance
    distance_difference = current_distance - (estimated_height + desired_distance)

    # Adjust the drone's position to maintain the desired distance from the tallest person
    if abs(distance_difference) > 5:  # A small threshold to prevent excessive adjustments
        if distance_difference > 0:
            # If the drone is too close, move backward
            tello.move_backward(max_speed)
        else:
            # If the drone is too far, move forward
            tello.move_forward(max_speed)
    else:
        # If the drone is at the desired distance, hover in place
        tello.send_rc_control(0, 0, 0, 0)


def main():
    tello = Tello()
    tello.connect()
    tello.streamon()

    while True:
        frame = tello.get_frame_read().frame

        # Estimate the height of the tallest person in the frame
        estimated_height = estimate_tallest_person_height(frame)

        # Follow the tallest person with the drone
        follow_tallest_person(tello, estimated_height)

        cv2.imshow("Tello Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
