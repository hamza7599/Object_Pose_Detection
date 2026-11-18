# 🎯 VisionAI Studio | YOLOv8 Detection & Pose Estimation Portal

A modern, high-performance web application built with **Streamlit** and **Ultralytics YOLOv8** that provides real-time multi-modal computer vision capabilities. It features a custom dark glassmorphism UI, dual operational sections, and full support for browser-based webcam streaming, image annotation, and video processing.

---

## ✨ Key Features

1. **🚗 Car & Person Object Detection**
   * Powered by `yolov8n.pt`.
   * Specifically filtered to track and bound **Persons (Class 0)** and **Cars (Class 2)** in real time.
2. **🏃 Human Skeletal Pose Estimation**
   * Powered by `yolov8n-pose.pt`.
   * Maps 17 full-body joint keypoints and skeletal limbs.
3. **📁 3 Interactive Input Modes (Per Section)**
   * **Real-Time Camera:** Live browser webcam inference via `streamlit-webrtc` (fully compatible with local and cloud deployments).
   * **Picture Upload:** Instant bounding box & keypoint overlay on uploaded images (`.jpg`, `.jpeg`, `.png`).
   * **Video Upload:** Frame-by-frame annotation and downloadable processing preview for video files (`.mp4`, `.avi`, `.mov`).
4. **☁️ Streamlit Community Cloud Ready**
   * Configured with headless Linux graphics and multimedia dependencies (`packages.txt` & `requirements.txt`).

---

## 📂 Project Structure

```text
yolo-vision-portal/
├── app.py               # Main Streamlit application with custom CSS & inference logic
├── requirements.txt     # Python packages for local and cloud environments
└── packages.txt         # Linux system dependencies required for OpenCV & video codecs
