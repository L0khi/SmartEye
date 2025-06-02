# 🔒 SmartEye – AI-Powered Security Detection System

SmartEye is a real-time security monitoring system powered by YOLOv8 and pose estimation.
It detects:
- Persons, weapons, and objects
- Human behaviors like sitting, jumping, and **falling**

🧠 Features:
- Real-time detection with webcam or video
- Custom WhatsApp alerts with image snapshots
- Audio alarm on fall detection
- Streamlit GUI dashboard
- Event logging in JSON + JPG

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📸 Screenshot
_Add a screenshot in the `assets/` folder to show UI_

## 📬 WhatsApp Alerts
Configure your [CallMeBot WhatsApp key](https://www.callmebot.com/blog/free-api-whatsapp-messages/) and update these in `app.py`:

```python
TWILIO_PHONE = "YOUR_PHONE"
TWILIO_API_KEY = "YOUR_API_KEY"
```

## 📁 Logs
All events are stored in `/logs/` with frame snapshots and timestamps.

---

### 📬 Contact
[GitHub](https://github.com/L0khi) | [Fiverr](https://fiverr.com/kulwantdhillon) | [Upwork](https://www.upwork.com/freelancers/~yourprofile)
