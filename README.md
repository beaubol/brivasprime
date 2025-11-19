# brivasprime
A swift Python Proof Of Human Gateway using MediaPipe
BioGuard Secure Gateway

BioGuard is a high-security, biometric liveness detection gateway designed for mobile and web applications. It prevents spoofing attacks (like holding up a photo) by challenging the user to perform random "motion proofs" (blinking, smiling, turning the head) before verifying their identity.

🏗 Architecture

The system uses a split-architecture for maximum security and responsiveness:

The Brain (Python Backend):

Framework: FastAPI (High-performance Async API).

Core AI: Google MediaPipe (Neural Network for Face Mesh) & OpenCV.

Logic: Calculates Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and Head Pose (Euler Angles) to mathematically verify human movement.

Extensibility: Ready to plug into Vector Databases (Milvus/Pinecone) or DeepFace for 1:N identity matching.

The Face (Frontend Gateway):

Framework: React (via Babel Standalone for portability).

Client-Side AI: MediaPipe Tasks Vision (WASM) for real-time UI feedback.

UI: Tailwind CSS for a cinematic, dark-mode "CyberSec" aesthetic.

🚀 Getting Started

Prerequisites

Python 3.9+

A webcam

1. Backend Setup (The Neural Engine)

Clone the repository and navigate to the project folder.

Install dependencies:

pip install fastapi uvicorn opencv-python mediapipe numpy pydantic multipart


Run the API:

uvicorn biometric_engine:app --reload


The API will be live at http://localhost:8000

2. Frontend Setup (The Mobile Gateway)

Since the frontend is a standalone HTML file for this demo:

Open biometric_gateway.html in any modern web browser (Chrome/Edge/Safari).

Allow Camera Access when prompted.

The app will initialize the Neural Engine (WASM) and begin the authentication loop.

🧪 API Endpoints

Method

Endpoint

Description

GET

/challenge/new

Returns a random challenge (e.g., blink, smile, turn_left) with an ID.

POST

/verify/liveness

Upload a captured frame + Challenge ID. Returns { verified: true, confidence: 0.99 }.

🛡 Security Logic

Blink Detection: Uses Eye Aspect Ratio (EAR). If EAR drops below 0.21, eyes are closed.

Smile Detection: Uses Mouth Aspect Ratio (MAR). Checks distance between lip corners.

Head Pose: Uses cv2.solvePnP to map 2D facial landmarks to a 3D generic face model, calculating Yaw, Pitch, and Roll.

🤝 Contributing Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request
That is awesome to hear! I'm glad we got the full "Cyberpunk Biometric Gateway" running with all the motion challenges—Blink, Smile, Turn Left/Right, and Look Up/Down.

You now have a robust, browser-based biometric verification tool that runs entirely on the client side (no server costs!) and looks great doing it.

**What you built:**
* **Core:** MediaPipe FaceMesh (Legacy Engine for maximum compatibility).
* **Logic:** Relative geometry math (works at any distance).
* **UI:** Custom Cyberpunk interface with real-time feedback loops.
* **Platform:** Hosted for free on GitHub Pages.

If you ever want to take it to the next level, you could connect that Python backend we wrote earlier to actually *save* the verification results or match the face against a database.

Enjoy your new BioGuard system! 🚀
