# 👓 GestraSpeak: Real-Time Sign Language to Speech Translation

GestraSpeak is a real-time computer vision application that translates sign language gestures into audible speech. Designed with future integration into wearable technology (e.g., smart glasses) in mind, the system aims to empower the hearing-impaired community by enabling more seamless communication with the world around them.

## 🔍 Overview

The project combines advanced gesture recognition with audio synthesis to build a lightweight and responsive sign-to-speech engine. Using real-time webcam input, GestraSpeak identifies hand and body landmarks, interprets the gesture, and outputs corresponding audio.

## ⚙️ Features

- 🎥 **Real-time webcam-based hand and pose detection**
- 🧠 **Gesture interpretation using MediaPipe landmarks**
- 🔊 **Speech generation via text-to-speech synthesis**
- 📦 **Lightweight implementation with potential for embedded deployment (e.g., smart glasses)**

## 🛠 Technologies Used

- **Python 3.8+** – Core development language  
- **OpenCV** – Video capture and image processing  
- **MediaPipe** – Hand and pose landmark detection  
- **pyttsx3** – Text-to-speech conversion  
- **NumPy** – Mathematical operations and calculations  

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or later
- Webcam
- Package manager: `pip`

### Installation

```bash
git clone https://github.com/iHakawaTi/GestraSpeak.git
cd GestraSpeak
pip install -r requirements.txt
