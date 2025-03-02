import cv2
import numpy as np
import os
import time
from flask import Flask, render_template, Response, request, jsonify
import serial
import threading
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)

# Global VideoCapture object (adjust the index if needed)
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: Unable to access the camera.")

# Dataset paths
DATASET_PATH = "dataset"
RECYCLE_PATH = os.path.join(DATASET_PATH, "recyclable")
NON_RECYCLE_PATH = os.path.join(DATASET_PATH, "non_recyclable")
MODEL_PATH = os.path.join(DATASET_PATH, "sgd_model.pkl")
SCALER_PATH = os.path.join(DATASET_PATH, "scaler.pkl")

# Ensure dataset directories exist
for path in [DATASET_PATH, RECYCLE_PATH, NON_RECYCLE_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

# Global flag to control automatic mode
automatic_mode = False

# Motion detection parameters
MIN_CONTOUR_AREA = 1000  # Minimum area of motion contour to trigger detection
MOTION_DELAY = 5  # Delay in seconds after motion is detected
BACKGROUND_SUBTRACTOR = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Initialize the classifier and feature extractor
sgd_classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, 
                           max_iter=1000, tol=1e-3, class_weight='balanced', 
                           random_state=42, n_jobs=-1)
scaler = StandardScaler()

# Training data storage
X_train = []  # Feature vectors
y_train = []  # Labels (0 for non-recyclable, 1 for recyclable)

# HOG feature extractor parameters
HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

def load_model():
    """Load the trained classifier and scaler if they exist."""
    global sgd_classifier, scaler, X_train, y_train
    
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            sgd_classifier = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("Loaded pre-trained SGD classifier model")
            
            # Load any existing training data from image files
            load_training_data()
            return True
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def save_model():
    """Save the trained classifier and scaler."""
    if len(X_train) > 0:
        joblib.dump(sgd_classifier, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        print("SGD classifier model saved")

def load_training_data():
    """Load training data from image files in the dataset folders."""
    global X_train, y_train
    
    X_train = []
    y_train = []
    
    # Load recyclable items (class 1)
    for filename in os.listdir(RECYCLE_PATH):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            filepath = os.path.join(RECYCLE_PATH, filename)
            features = extract_features(cv2.imread(filepath))
            if features is not None:
                X_train.append(features)
                y_train.append(1)
    
    # Load non-recyclable items (class 0)
    for filename in os.listdir(NON_RECYCLE_PATH):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            filepath = os.path.join(NON_RECYCLE_PATH, filename)
            features = extract_features(cv2.imread(filepath))
            if features is not None:
                X_train.append(features)
                y_train.append(0)
    
    print(f"Loaded {len(X_train)} training examples: {y_train.count(1)} recyclable, {y_train.count(0)} non-recyclable")

def train_classifier():
    """Train the SGD classifier with current data."""
    global sgd_classifier, scaler, X_train, y_train
    
    if len(X_train) < 2 or len(set(y_train)) < 2:
        print("Not enough training data to fit classifier")
        return False
    
    try:
        # Scale features
        X_scaled = scaler.fit_transform(X_train)
        
        # Train the classifier
        sgd_classifier.fit(X_scaled, y_train)
        
        # Save the model
        save_model()
        
        # Evaluate (if we have enough data)
        if len(X_train) >= 5:
            # Try to do cross-validation if we have enough data per class
            try:
                scores = cross_val_score(sgd_classifier, X_scaled, y_train, cv=min(5, min(y_train.count(0), y_train.count(1))))
                print(f"Cross-validation accuracy: {np.mean(scores):.4f}")
            except Exception as e:
                print(f"Not enough data for cross-validation: {e}")
                # Just check training accuracy instead
                y_pred = sgd_classifier.predict(X_scaled)
                accuracy = accuracy_score(y_train, y_pred)
                print(f"Training accuracy: {accuracy:.4f}")
        
        return True
    except Exception as e:
        print(f"Error training classifier: {e}")
        return False

def extract_features(img):
    """
    Extract features from an image for machine learning classification.
    Uses a combination of HOG, color histograms, and basic image statistics.
    """
    if img is None:
        return None
    
    # Resize to a fixed size for consistent feature extraction
    img_resized = cv2.resize(img, (128, 128))
    
    # Convert to different color spaces for feature extraction
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    
    features = []
    
    # 1. HOG features (shape and texture)
    try:
        hog_features = HOG.compute(img_gray).flatten()
        features.extend(hog_features)
    except Exception as e:
        print(f"Error computing HOG features: {e}")
        # Use a smaller size if HOG fails
        hog_features = cv2.HOGDescriptor((64, 64), (8, 8), (4, 4), (4, 4), 9).compute(
            cv2.resize(img_gray, (64, 64))).flatten()
        features.extend(hog_features)
    
    # 2. Color histograms (color distribution)
    for channel in range(3):  # BGR channels
        hist = cv2.calcHist([img_resized], [channel], None, [32], [0, 256])
        features.extend(hist.flatten())
    
    # 3. HSV color space histograms (for better color representation)
    for channel in range(3):  # HSV channels
        hist = cv2.calcHist([img_hsv], [channel], None, [32], [0, 256])
        features.extend(hist.flatten())
    
    # 4. Basic statistics for each channel
    for color_space in [img_resized, img_hsv]:
        for channel in range(3):
            channel_data = color_space[:, :, channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75)
            ])
    
    # 5. Edge detection features
    edges = cv2.Canny(img_gray, 100, 200)
    edge_features = [
        np.mean(edges),
        np.std(edges),
        np.sum(edges) / (128 * 128)
    ]
    features.extend(edge_features)
    
    return np.array(features)

def capture_frame():
    """Captures a single frame from the global camera and applies zoom by cropping."""
    global cap
    ret, frame = cap.read()
    if not ret:
        return None

    # Apply zoom by cropping the center of the frame.
    height, width = frame.shape[:2]
    zoom_factor = 2  # Adjust this value to change the zoom level (higher = more zoomed in)
    new_width = width // zoom_factor
    new_height = height // zoom_factor
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2
    cropped = frame[start_y:start_y+new_height, start_x:start_x+new_width]
    # Resize the cropped image back to original dimensions (or desired streaming size)
    zoomed_frame = cv2.resize(cropped, (width, height))
    return zoomed_frame

def classify_image(frame):
    """
    Classifies an image using the SGD classifier.
    Returns the predicted label ("Recyclable" or "Non-Recyclable" or "Unknown").
    """
    global sgd_classifier, scaler, X_train, y_train
    
    # Check if we have a trained model
    if len(X_train) < 2 or len(set(y_train)) < 2:
        return "Need more training data"
    
    # Extract features
    features = extract_features(frame)
    if features is None:
        return "Feature extraction failed"
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Get the decision value (distance from the hyperplane)
    # This gives us a confidence measure
    decision_value = sgd_classifier.decision_function(features_scaled)[0]
    
    # Predict based on decision value with confidence threshold
    confidence = abs(decision_value)
    confidence_threshold = 0.5  # Adjust as needed
    
    print(f"Classification decision value: {decision_value:.4f}, confidence: {confidence:.4f}")
    
    if confidence < confidence_threshold:
        return "Unknown"
    
    # Make prediction (0 = Non-Recyclable, 1 = Recyclable)
    prediction = sgd_classifier.predict(features_scaled)[0]
    
    if prediction == 1:
        return "Recyclable"
    else:
        return "Non-Recyclable"

def gen_frames():
    """Generator function that reads frames from the camera and yields them as MJPEG."""
    global cap, X_train, y_train
    while True:
        frame = capture_frame()
        if frame is None:
            break
        
        # Overlay classification status and training info
        text_lines = []
        if automatic_mode:
            text_lines.append("AUTO MODE ON")
        
        text_lines.append(f"Training data: {len(X_train)} samples")
        if y_train:
            text_lines.append(f"({y_train.count(1)} recyclable, {y_train.count(0)} non-recyclable)")
        
        # Display text lines
        for i, line in enumerate(text_lines):
            cv2.putText(frame, line, (10, 30 + i*25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
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
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train', methods=['POST'])
def train():
    """
    Captures a frame, extracts features, adds to the training data,
    trains the classifier, and saves the image in the appropriate folder.
    """
    global X_train, y_train
    
    label = request.form.get("label")
    if label not in ["recyclable", "non_recyclable"]:
        return jsonify({"error": "Invalid label"}), 400

    frame = capture_frame()
    if frame is None:
        return jsonify({"error": "Failed to capture frame"}), 500

    if label == "recyclable":
        folder = RECYCLE_PATH
        class_label = 1
        category = "Recyclable"
    else:
        folder = NON_RECYCLE_PATH
        class_label = 0
        category = "Non-Recyclable"

    # Extract features and add to training data
    features = extract_features(frame)
    if features is None:
        return jsonify({"error": "Feature extraction failed"}), 500
    
    X_train.append(features)
    y_train.append(class_label)
    
    # Save the training image
    filename = f"{category}_{int(time.time())}.jpg"
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, frame)
    
    # Train the classifier with the updated dataset
    training_success = train_classifier()
    
    return jsonify({
        "message": f"Item saved and trained as {category}",
        "details": f"Model has {y_train.count(1)} recyclable and {y_train.count(0)} non-recyclable samples",
        "model_updated": training_success
    })

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
            print(f"Classification result '{result}', no command sent.")
        ser.close()
    except Exception as e:
        print(f"Serial communication error: {e}")
        return jsonify({"error": f"Serial communication error: {e}"}), 500

    return jsonify({"result": result})

@app.route('/toggle_automatic', methods=['POST'])
def toggle_automatic():
    global automatic_mode
    mode = request.form.get("mode")
    if mode == "on":
        automatic_mode = True
        message = "Automatic mode enabled."
    elif mode == "off":
        automatic_mode = False
        message = "Automatic mode disabled."
    else:
        return jsonify({"error": "Invalid mode."}), 400
    return jsonify({"message": message})

def detect_motion():
    """Detects motion using background subtraction and morphological operations."""
    global cap
    ret, frame = cap.read()
    if not ret:
        return False

    # Apply background subtraction
    fg_mask = BACKGROUND_SUBTRACTOR.apply(frame)

    # Remove shadows (shadows are marked as gray in the mask)
    _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of the detected motion
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contour is large enough to indicate motion
    for contour in contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            return True
    return False

def automatic_monitor():
    """Background thread that monitors for motion and triggers classification."""
    global automatic_mode
    last_detection_time = 0
    while True:
        if automatic_mode:
            current_time = time.time()
            if detect_motion() and (current_time - last_detection_time) > MOTION_DELAY:
                print("Motion detected, triggering classification in a few seconds.")
                last_detection_time = current_time
                #time.sleep(MOTION_DELAY)  # Wait for a few seconds after motion detected
                
                # Capture multiple frames for more reliable classification
                frames = []
                for _ in range(3):  # Capture 3 frames
                    frame = capture_frame()
                    if frame is not None:
                        frames.append(frame)
                    time.sleep(0.5)
                
                if frames:
                    # Classify each frame and use majority voting
                    results = [classify_image(frame) for frame in frames]
                    # Filter out Unknown results
                    valid_results = [r for r in results if r not in ["Unknown", "Need more training data", "Feature extraction failed"]]
                    
                    if valid_results:
                        final_result = max(set(valid_results), key=valid_results.count)
                    else:
                        final_result = "Unknown"
                    
                    print(f"Automatic classification results: {results}, final: {final_result}")
                    
                    # Send command via COM9
                    try:
                        ser = serial.Serial('COM9', 115200, timeout=1)
                        if final_result == "Recyclable":
                            ser.write(b'recyclable\n')
                            print("Sent: recyclable")
                            time.sleep(3)
                        elif final_result == "Non-Recyclable":
                            ser.write(b'non_recyclable\n')
                            print("Sent: non_recyclable")
                            time.sleep(3)
                        else:
                            print(f"Result '{final_result}', no command sent.")
                        ser.close()
                    except Exception as e:
                        print(f"Serial communication error: {e}")
                else:
                    print("Failed to capture frames during automatic classification.")
        time.sleep(0.5)

@app.route('/reset_model', methods=['POST'])
def reset_model():
    """Reset the SGD classifier model and training data."""
    global X_train, y_train, sgd_classifier, scaler
    
    X_train = []
    y_train = []
    
    # Reset the classifier and scaler
    sgd_classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, 
                               max_iter=1000, tol=1e-3, class_weight='balanced', 
                               random_state=42, n_jobs=-1)
    scaler = StandardScaler()
    
    # Delete model files if they exist
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        os.remove(SCALER_PATH)
    
    return jsonify({"message": "Model and training data reset successfully"})

@app.route('/model_info', methods=['GET'])
def model_info():
    """Return information about the current model."""
    global X_train, y_train
    
    recyclable_count = y_train.count(1) if y_train else 0
    non_recyclable_count = y_train.count(0) if y_train else 0
    
    model_status = {
        "total_samples": len(X_train),
        "recyclable_samples": recyclable_count,
        "non_recyclable_samples": non_recyclable_count,
        "model_trained": len(X_train) > 0 and len(set(y_train)) > 1,
        "model_file_exists": os.path.exists(MODEL_PATH)
    }
    
    return jsonify(model_status)

if __name__ == '__main__':
    # Try to load existing model
    load_model()
    
    # Start the monitoring thread
    monitor_thread = threading.Thread(target=automatic_monitor, daemon=True)
    monitor_thread.start()
    
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    finally:
        # Save model before exiting
        if len(X_train) > 0:
            save_model()
        cap.release()
        cv2.destroyAllWindows()
