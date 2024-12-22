import cv2
import numpy as np
import time
import os

# Paths to the YOLO model configuration and weights
YOLO_CONFIG_PATH = "./cfg/yolov3.cfg"  # Update the path if necessary
YOLO_WEIGHTS_PATH = "./yolov3.weights"  # Update the path if necessary
YOLO_CLASSES_PATH = "./data/coco.names"  # Update if necessary

# Load the COCO class labels
def load_classes(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]

# Load YOLO model
def load_yolo_model(config_path, weights_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net

# Process detections
def process_detections(outputs, frame, classes, confidence_threshold=0.5, nms_threshold=0.4):
    height, width = frame.shape[:2]
    boxes, confidences, class_ids = [], [], []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    return [(boxes[i[0]], confidences[i[0]], class_ids[i[0]]) for i in indices]

# Draw bounding boxes and labels
def draw_predictions(frame, detections, classes):
    for box, confidence, class_id in detections:
        x, y, w, h = box
        label = f"{classes[class_id]}: {confidence:.2f}"
        color = (0, 255, 0)  # Green for bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# Main function
def main():
    # Load classes and YOLO model
    classes = load_classes(YOLO_CLASSES_PATH)
    net = load_yolo_model(YOLO_CONFIG_PATH, YOLO_WEIGHTS_PATH)
    
    # Open the camera
    cap = cv2.VideoCapture(0)  # 0 for default camera; modify for external camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("[INFO] Starting video stream...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Preprocess frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        
        # Run inference
        start_time = time.time()
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        detections = process_detections(outputs, frame, classes)
        end_time = time.time()
        
        # Draw detections
        frame = draw_predictions(frame, detections, classes)
        
        # Show FPS
        fps = f"FPS: {1 / (end_time - start_time):.2f}"
        cv2.putText(frame, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Display the frame
        cv2.imshow("Object Detection", frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
