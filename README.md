# brivasprime
A swift Python Proof Of Human Gateway using MediaPipe
BioGuard Secure GatewayBioGuard is a high-security, biometric liveness detection gateway designed for mobile and web applications. 

It prevents spoofing attacks (like holding up a photo) by challenging the user to perform random "motion proofs" (blinking, smiling, turning the head) before verifying their identity.🏗 ArchitectureThe system uses a split-architecture for maximum security and responsiveness:The Brain (Python Backend):Framework: FastAPI (High-performance Async API).

Core AI: Google MediaPipe (Neural Network for Face Mesh) & OpenCV.Logic: Calculates Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and Head Pose (Euler Angles) to mathematically verify human movement.Extensibility: Ready to plug into Vector Databases (Milvus/Pinecone) or DeepFace for 1:N identity matching.The Face (Frontend Gateway):Framework: React (via Babel Standalone for portability).Client-Side AI: MediaPipe Tasks Vision (WASM) for real-time UI feedback.UI: Tailwind CSS for a cinematic, dark-mode "CyberSec" aesthetic.🚀 Getting StartedPrerequisitesPython 3.9+A webcam1. Backend Setup (The Neural Engine)Clone the repository and navigate to the project folder.Install dependencies:pip install fastapi uvicorn opencv-python mediapipe numpy pydantic multipart

Run the API:uvicorn biometric_engine:app --reload

The API will be live at http://localhost:80002. Frontend Setup (The Mobile Gateway)Since the frontend is a standalone HTML file for this demo:Open biometric_gateway.html in any modern web browser (Chrome/Edge/Safari).Allow Camera Access when prompted.The app will initialize the Neural Engine (WASM) and begin the authentication loop.🧪 API EndpointsMethodEndpointDescriptionGET/challenge/newReturns a random challenge (e.g., blink, smile, turn_left) with an ID.POST/verify/livenessUpload a captured frame + Challenge ID. Returns { verified: true, confidence: 0.99 }.

🛡 Security LogicBlink Detection: Uses Eye Aspect Ratio (EAR). If EAR drops below 0.21, eyes are closed.Smile Detection: Uses Mouth Aspect Ratio (MAR). Checks distance between lip corners.Head Pose: Uses cv2.solvePnP to map 2D facial landmarks to a 3D generic face model, calculating Yaw, Pitch, and Roll.🤝 ContributingFork the ProjectCreate your Feature Branch (git checkout -b feature/AmazingFeature)Commit your Changes (git commit -m 'Add some AmazingFeature')Push to the Branch (git push origin feature/AmazingFeature)Open a Pull Request

It looks like the MediaPipe library wasn't loading correctly because modern browser security handles "modules" differently than older scripts. The errors you saw (`Unexpected token 'export'` and `FilesetResolver is not defined`) happen when the browser tries to read a modern library file as an old-school script.

I have **fixed the code** below by switching to the correct ES Module import method. I have also included the **README** you requested to document the project.


http://googleusercontent.com/immersive_entry_chip/0

### Fixed Mobile Gateway
This version uses the **ES Module (+esm)** CDN link, which fixes the `SyntaxError` and `ReferenceError` by loading MediaPipe correctly in modern browsers.


http://googleusercontent.com/immersive_entry_chip/1
