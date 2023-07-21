# Drone Stalker

This repository includes scripts that use a Tello drone to track specific targets, namely human faces and cameras. Each script is written in Python, and leverages Computer Vision libraries such as OpenCV, dlib, and YOLO for object detection and tracking.

---

Certainly, I'll add a section to the README about running the scripts.

---

## Running the Scripts

To run any of these scripts, you first need to connect your Tello drone to your computer. Once connected, you can run the scripts from your terminal or command line.

Follow these steps to run a script:

1. **Connect the Tello Drone**: Power on your Tello drone. Then, connect your computer to the drone's WiFi network. It should appear as a network named something like "TELLO-XXXXXX".

2. **Navigate to the Script's Directory**: Use the `cd` command to navigate to the directory containing the script you want to run. For example, if your script is located in a folder named "tello" on your Desktop, you would use the following command:

    ```bash
    cd ~/Desktop/tello
    ```

    This path may vary depending on where you've stored the scripts.

3. **Run the Script**: Now you can run the script with the Python command followed by the script's filename. For example, to run the `follow_face_30s.py` script, you would use the following command:

    ```bash
    python follow_face_30s.py
    ```

    If you're using a specific version of Python, you may need to specify that version in the command. For example, if you're using Python 3.7, you would use the following command:

    ```bash
    python3.7 follow_face_30s.py
    ```

Remember to replace `'person.jpg'` in the `follow_specific_person.py` script with the path to your reference image before running the script.

Similarly, ensure that the path to the YOLO model files in `follow_camera.py` is correct for your local setup.

---

That's it! The script should now run, and your Tello drone should start tracking faces or cameras, depending on the script you're running.



## Dependencies

To run these scripts, you need Python 3.6 or later and the following libraries:

- `cv2` (OpenCV)
- `dlib`
- `numpy`
- `djitellopy`

For the object detection script, you also need the YOLOv3 model files (`yolov3.weights` and `yolov3.cfg`).

You can install the necessary Python libraries with pip:

```bash
pip install opencv-python-headless dlib numpy djitellopy2
```

## Scripts

### 1. Follow a Face for 30 seconds

The first script (`follow_face_30s.py`) enables the Tello drone to detect and follow the first face it sees for up to 30 seconds, before moving on to track the next face it identifies. The drone also takes a photo of each face it detects.

### 2. Follow a Specific Person

The second script (`follow_specific_person.py`) uses a reference image to recognize a specific person's face. The drone will then follow this person around, taking a picture of the individual every minute.

Before running this script, you need to replace `'person.jpg'` in the script with the path to your reference image.

### 3. Detect and Follow Cameras

The third script (`follow_camera.py`) uses the YOLO object detection model to identify cameras in the drone's field of view and follow them.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
