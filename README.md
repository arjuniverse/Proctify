# 🧠 PROCTIFY – AI Smart Exam Monitoring System  
### *Ensuring Integrity Through AI*

PROCTIFY is a full-stack AI-powered online exam proctoring system that monitors candidates in real time using computer vision and behavioral analysis. It detects suspicious activities like looking away, phone usage, and multiple faces to ensure exam integrity.

---

## 🚀 Features

### 🎥 Real-Time Monitoring
- Live webcam streaming using OpenCV  
- Continuous frame-by-frame analysis  

### 👤 Face Detection (MediaPipe)
- Detects face presence  
- Flags:
  - ❌ No face  
  - 🚫 Multiple faces (High violation)  

### 🧭 Head Pose Estimation
- Tracks yaw (left/right) and pitch (up/down)  
- Detects:
  - Looking away  
  - Looking down  

### 👀 Eye Gaze Tracking
- Detects gaze direction:
  - Center  
  - Left / Right  
  - Down  

### 📱 Object Detection (YOLOv8)
Detects:
- Phone 📵 (HIGH violation)  
- Book 📚  
- Person 👥  

---

## 🧮 Trust Score System

- Initial Score: **100%**

| Action              | Penalty   |
|--------------------|----------|
| Looking away       | -2/sec   |
| Looking down       | -2/sec   |
| Phone detected     | -30      |
| Multiple faces     | -50      |

📊 Displayed live in the UI.

---

## 📊 Behavior Analysis

```python
tracking_data = {
    "total_time": 0,
    "looking_forward_time": 0,
    "looking_away_time": 0,
    "looking_down_time": 0,
    "violations": 0,
    "phone_detected": 0,
    "multiple_faces": 0,
    "alerts": []
}
```

---

## ⚠️ Warning System

- ⚠️ Warning 1/3  
- ⚠️ Warning 2/3  
- 🚫 Final Warning  

---

## ❌ Auto Termination

Exam is terminated if:
- Phone detected  
- Multiple faces detected  
- Violations ≥ 3  

### On Termination:
- Camera stops  
- UI freezes  
- Message displayed:  
  ```
  🚫 Exam Terminated – Cheating Detected
  ```
- Logs saved  
- Report generated  
- Alert triggered  

---

## 📸 Evidence Capture

- Screenshots captured during violations  
- Stored in `/evidence/`  

---

## 🔔 Admin Alert System

Logs events in JSON format:

```json
{
  "student_id": "123",
  "event": "phone_detected",
  "time": "timestamp",
  "severity": "HIGH"
}
```

Saved in:
```
logs/alerts.jsonl
```

---

## 📄 Report Generation

Generated automatically after exam.

### Includes:
- Candidate details  
- Exam duration  
- Behavior percentages  
- Violations  
- Trust Score  
- Final result (PASS / FAIL)  
- Evidence images  

### Formats:
- JSON  
- PDF (ReportLab)  

### Result Logic:

```python
if phone_detected > 0 or violations >= 3:
    status = "FAILED"
else:
    status = "PASSED"
```

---

## 🖥️ UI Pages

- **Login Page** – Username & Password  
- **Details Page** – Name, Student ID, Exam Name  
- **Camera Check** – Face verification & validation  
- **Exam Page** – Live monitoring dashboard  
- **Report Page** – Final analysis & results  

---

## 🗂️ Project Structure

```
/project
 ├── app.py
 ├── detection/
 │    ├── face_detection.py
 │    ├── gaze_tracking.py
 │    ├── head_pose.py
 │    ├── object_detection.py
 │    ├── behavior_analysis.py
 │
 ├── utils/
 │    ├── report_generator.py
 │    ├── alert_system.py
 │    ├── csv_logger.py
 │
 ├── static/
 │    ├── css/
 │    ├── js/
 │
 ├── templates/
 │    ├── login.html
 │    ├── details.html
 │    ├── camera_check.html
 │    ├── exam.html
 │    ├── report.html
 │
 ├── logs/
 ├── reports/
 ├── evidence/
 ├── requirements.txt
```

---

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/proctify.git
cd proctify

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open in browser:
```
http://127.0.0.1:5000
```

---

## 📦 Requirements

```
flask
opencv-python
mediapipe
ultralytics
numpy
reportlab
```

---

## ⚡ Notes

- First run will download YOLOv8 model (`yolov8n.pt`)  
- Webcam access is required  
- Designed for local execution  
- Login is demo-based (no authentication)  

---

## 🧩 System Highlights

- Modular architecture  
- Real-time AI inference  
- Evidence-based reporting  
- Automated decision system  
- Clean PROCTIFY UI (dark + blue theme)  

---

## 🎯 Final Goal

✔ Live cheating detection  
✔ Smart AI monitoring  
✔ Automatic termination  
✔ Detailed reporting  
✔ Fully functional local system  

---

## 📌 Future Improvements

- Face recognition (identity verification)  
- Multi-camera support  
- Cloud deployment  
- LMS integration  
- Advanced anomaly detection  

---

## 📜 License

This project is for educational purposes. You can modify and use it as needed.

---
