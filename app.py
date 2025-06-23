# === SmartEye: AI Security System with Image Processing & Human-Only Pose Detection ===

import cv2
import time
import torch
import numpy as np
import streamlit as st
from datetime import datetime
from pathlib import Path
from PIL import Image
import json
import os
import threading
from playsound import playsound
import requests
from urllib.parse import quote

# === YOLOv8 Setup ===
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
pose_model = YOLO("yolov8n-pose.pt")

# === Setup ===
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
DETECTION_CLASSES = ['person', 'car', 'knife', 'backpack', 'cell phone', 'suitcase']
SAVE_IMAGE = True
CONFIDENCE_THRESHOLD = 0.4
TWILIO_API_URL = "https://api.callmebot.com/whatsapp.php"
TWILIO_PHONE = "YOUR_PHONE_NUMBER"
TWILIO_API_KEY = "YOUR_API_KEY"

# === Utility: Play Alert Sound ===
def play_alert():
    threading.Thread(target=playsound, args=("alert.wav",)).start()

# === Utility: Send WhatsApp Alert with Optional Image ===
def send_whatsapp_alert(message, image_path=None):
    try:
        full_msg = quote(message)
        url = f"{TWILIO_API_URL}?phone={TWILIO_PHONE}&text={full_msg}&apikey={TWILIO_API_KEY}"
        requests.get(url)
        if image_path:
            img_msg = quote("Snapshot: " + image_path)
            url_img = f"{TWILIO_API_URL}?phone={TWILIO_PHONE}&text={img_msg}&apikey={TWILIO_API_KEY}"
            requests.get(url_img)
    except Exception as e:
        print("WhatsApp notification error:", e)

# === Log Detection ===
def log_detection(detections, frame, save_image=True):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = LOG_DIR / f"log_{timestamp}.json"
    output_path = LOG_DIR / f"frame_{timestamp}.jpg"

    data = {
        "timestamp": timestamp,
        "detections": detections
    }
    if save_image:
        cv2.imwrite(str(output_path), frame)
        data['image_path'] = str(output_path)

    with open(log_path, "w") as f:
        json.dump(data, f, indent=4)

    return str(output_path) if save_image else None

# === Detection Pipeline ===
def detect_objects(frame):
    results = model(frame)[0]
    detections = []
    humans = []
    annotated = frame.copy()

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        cls_name = model.names[cls_id]
        conf = float(box.conf[0].item())
        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detection = {
            "class": cls_name,
            "confidence": round(conf, 2),
            "bbox": [x1, y1, x2, y2]
        }
        detections.append(detection)

        if cls_name == "person":
            humans.append((x1, y1, x2, y2))

        if cls_name in DETECTION_CLASSES:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return annotated, detections, humans

# === Pose Estimation: Human-Only ===
def detect_pose(frame, human_bboxes):
    poses = []
    annotated = frame.copy()

    for (x1, y1, x2, y2) in human_bboxes:
        person_roi = annotated[y1:y2, x1:x2]
        if person_roi.size == 0:
            continue
        results = pose_model(person_roi)[0]
        for kp in results.keypoints.xy:
            keypoints = kp.cpu().numpy().astype(int)
            for i, (x, y) in enumerate(keypoints):
                cv2.circle(annotated, (x + x1, y + y1), 3, (0, 0, 255), -1)

            if len(keypoints) >= 5:
                y_diffs = abs(keypoints[0][1] - keypoints[1][1])
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                if y_diffs < 10:
                    pose_state = "Fall Detected"
                    play_alert()
                    img_path = log_detection({"pose": pose_state}, annotated, SAVE_IMAGE)
                    send_whatsapp_alert(f"⚠️ SmartEye: Fall detected at {timestamp}", img_path)
                elif abs(keypoints[11][1] - keypoints[13][1]) < 15:
                    pose_state = "Sitting"
                    send_whatsapp_alert(f"🪑 SmartEye: Sitting posture detected at {timestamp}")
                elif keypoints[5][1] < keypoints[11][1]:
                    pose_state = "Standing"
                else:
                    pose_state = "Jumping"
                    send_whatsapp_alert(f"🤸 SmartEye: Jumping detected at {timestamp}")

                poses.append(pose_state)
                cv2.putText(annotated, pose_state, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return annotated, poses

# === Streamlit GUI ===
st.set_page_config(page_title="SmartEye Security AI", layout="wide")
st.title("🔒 SmartEye: Real-Time IR+RGB Detection + Human Pose Alerts")
st.markdown("""SmartEye is a dual-mode AI security system using YOLOv8 and Pose Detection.
Detects persons, weapons, behaviors (sit, fall, jump), and logs events from IR and RGB inputs.
Only analyzes pose for human objects and supports image input processing too.""")

option = st.radio("Select Input Source:", ("Webcam", "Upload Video", "Image"))
run_detection = st.toggle("Activate Detection", value=False)

frame_placeholder = st.empty()
log_placeholder = st.empty()

cap = None
uploaded_img = None

if option == "Webcam":
    cap = cv2.VideoCapture(0)
elif option == "Upload Video":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        temp_path = f"temp_{int(time.time())}.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp_path)
elif option == "Image":
    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_img is not None:
        img = Image.open(uploaded_img)
        frame = np.array(img.convert("RGB"))
        run_detection = True

# === Main Loop ===
if run_detection:
    st.info("Detection Running. Press Stop to end.")
    if cap:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Video has ended or camera feed lost.")
                break

            annotated_obj, detections, humans = detect_objects(frame)
            annotated_pose, poses = detect_pose(annotated_obj, humans)
            frame_rgb = cv2.cvtColor(annotated_pose, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            if detections or poses:
                log_detection({"objects": detections, "poses": poses}, annotated_pose, SAVE_IMAGE)
                with log_placeholder.container():
                    st.write("🚨 **Event Alert:**")
                    for det in detections:
                        st.json(det)
                    for p in poses:
                        st.write(f"🧍 Pose Detected: {p}")

            if not run_detection:
                st.warning("Detection paused.")
                break

        cap.release()
    elif uploaded_img is not None:
        annotated_obj, detections, humans = detect_objects(frame)
        annotated_pose, poses = detect_pose(annotated_obj, humans)
        frame_rgb = cv2.cvtColor(annotated_pose, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        if detections or poses:
            log_detection({"objects": detections, "poses": poses}, annotated_pose, SAVE_IMAGE)
            with log_placeholder.container():
                st.write("🚨 **Event Alert:**")
                for det in detections:
                    st.json(det)
                for p in poses:
                    st.write(f"🧍 Pose Detected: {p}")

st.markdown("---")
st.caption("SmartEye © 2025 | Dual Vision AI + WhatsApp Alerts by Lokhi D.")
