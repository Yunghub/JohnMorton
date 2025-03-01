import cv2
import numpy as np
import os
import time
from flask import Flask, render_template, Response, request, jsonify
import serial
import time

app = Flask(__name__)

# Global VideoCapture object (adjust the index if needed)
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: Unable to access the camera.")

# Dataset paths
DATASET_PATH = "dataset"
RECYCLE_PATH = os.path.join(DATASET_PATH, "recyclable")
NON_RECYCLE_PATH = os.path.join(DATASET_PATH, "non_recyclable")

# Ensure dataset directories exist
for path in [RECYCLE_PATH, NON_RECYCLE_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

def capture_frame():
    """Captures a single frame from the global camera."""
    global cap
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def compute_matches(img1, img2):
    """Uses ORB feature detection and BFMatcher to compare two images."""
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good_matches = [m for m in matches if m.distance < 50]
    return len(good_matches)

def classify_image(captured_img):
    """
    Compares the captured image with each image in the training folders.
    Returns the predicted label ("Recyclable" or "Non-Recyclable").
    """
    scores = {"Recyclable": 0, "Non-Recyclable": 0}
    
    def update_score(folder, label):
        nonlocal scores
        count = 0
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            train_img = cv2.imread(filepath)
            if train_img is None:
                continue
            match_count = compute_matches(captured_img, train_img)
            scores[label] += match_count
            count += 1
        if count == 0:
            print(f"No training data for {label} items yet.")
    
    update_score(RECYCLE_PATH, "Recyclable")
    update_score(NON_RECYCLE_PATH, "Non-Recyclable")
    
    print("Matching scores:", scores)
    if scores["Recyclable"] > scores["Non-Recyclable"]:
        return "Recyclable"
    elif scores["Non-Recyclable"] > scores["Recyclable"]:
        return "Non-Recyclable"
    else:
        return "Unknown"

def gen_frames():
    """Generator function that reads frames from the camera and yields them as MJPEG."""
    global cap
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route to stream the video feed."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train', methods=['POST'])
def train():
    """
    Captures a frame and saves it in the appropriate folder
    based on the label ('recyclable' or 'non_recyclable').
    """
    label = request.form.get("label")
    if label not in ["recyclable", "non_recyclable"]:
        return jsonify({"error": "Invalid label"}), 400

    frame = capture_frame()
    if frame is None:
        return jsonify({"error": "Failed to capture frame"}), 500

    if label == "recyclable":
        folder = RECYCLE_PATH
        label_str = "Recyclable"
    else:
        folder = NON_RECYCLE_PATH
        label_str = "Non-Recyclable"

    filename = f"{label_str}_{int(time.time())}.jpg"
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, frame)
    return jsonify({"message": f"Item saved as {filepath}"})

@app.route('/classify', methods=['POST'])
def classify_route():
    """
    Captures a frame, runs the classifier, sends a serial command,
    and returns the classification result.
    """
    frame = capture_frame()
    if frame is None:
        return jsonify({"error": "Failed to capture frame"}), 500

    result = classify_image(frame)
    
    # Open serial connection on COM9 with baud rate 115200 and send command
    try:
        ser = serial.Serial('COM9', 115200, timeout=1)
        if result == "Recyclable":
            ser.write(b'recyclable\n')
            print("Sent: recyclable")
            time.sleep(3)
        elif result == "Non-Recyclable":
            ser.write(b'non_recyclable\n')
            print("Sent: non_recyclable")
            time.sleep(3)
        else:
            print("Unknown classification result, no command sent.")
        ser.close()
    except Exception as e:
        print("Serial communication error:", e)
        return jsonify({"error": "Serial communication error"}), 500

    return jsonify({"result": result})

if __name__ == '__main__':
    try:
        # Run the Flask app on all interfaces (0.0.0.0) and port 5000.
        app.run(host="0.0.0.0", port=5000, debug=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
