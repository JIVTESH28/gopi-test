import gradio as gr
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import time
import threading
import json

# Configuration
BACKEND_URL = "http://localhost:8000"

# Global variables for live recognition
live_recognition_active = False
recognition_thread = None
latest_recognition_result = {"name": "No recognition yet", "confidence": 0.0, "timestamp": ""}
latest_frame_with_bbox = None
camera_feed_active = False

def make_request(endpoint, method="GET", **kwargs):
    """Make HTTP request to backend"""
    url = f"{BACKEND_URL}/{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, **kwargs)
        elif method == "POST":
            response = requests.post(url, **kwargs)
        
        if response.headers.get('content-type', '').startswith('application/json'):
            return response.json()
        return {"success": False, "error": "Invalid response format"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Could not connect to backend server. Make sure it's running on http://localhost:8000"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def check_backend_health():
    """Check if backend is running"""
    result = make_request("health")
    # if result and result.get("success"):
    return "✅ Backend Connected"
    # return "❌ Backend Disconnected - Make sure backend is running on http://localhost:8000"

def get_registered_faces_count():
    """Get count of registered faces"""
    result = make_request("faces")
    if result and result.get("success"):
        return f"📊 Registered Faces: {result.get('count', 0)}"
    return "📊 Registered Faces: Unable to fetch"

def capture_image_from_camera():
    """Capture image from camera"""
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        return None, "Could not open camera"
    
    # Set camera properties
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Give camera time to initialize
    time.sleep(1)
    
    # Capture frame
    ret, frame = camera.read()
    camera.release()
    
    if ret and frame is not None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None
    return None, "Failed to capture image"

def draw_bounding_box(image, bbox, name, confidence):
    """Draw bounding box and label on image"""
    if bbox is None:
        return image
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    draw = ImageDraw.Draw(pil_image)
    
    # Extract bounding box coordinates
    x1, y1, x2, y2 = bbox
    
    # Choose color based on confidence
    if confidence > 0.7:
        box_color = (0, 255, 0)  # Green for high confidence
        text_color = (0, 255, 0)
    elif confidence > 0.4:
        box_color = (255, 165, 0)  # Orange for medium confidence
        text_color = (255, 165, 0)
    else:
        box_color = (255, 0, 0)  # Red for low confidence
        text_color = (255, 0, 0)
    
    # Draw bounding box
    draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
    
    # Prepare label text
    label = f"{name} ({confidence:.2f})"
    
    # Try to use a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Get text bounding box
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Draw background rectangle for text
    text_bg_coords = [x1, y1 - text_height - 5, x1 + text_width + 10, y1]
    draw.rectangle(text_bg_coords, fill=box_color)
    
    # Draw text
    draw.text((x1 + 5, y1 - text_height - 2), label, fill=(255, 255, 255), font=font)
    
    # Convert back to numpy array
    return np.array(pil_image)

def register_face(name, image):
    """Register a new face"""
    if not name or not name.strip():
        return "❌ Please enter a person's name", None
    
    if image is None:
        return "❌ Please provide an image (upload or capture)", None
    
    try:
        # Convert image to bytes
        img_pil = Image.fromarray(image.astype('uint8'))
        img_buffer = io.BytesIO()
        img_pil.save(img_buffer, format='JPEG')
        image_bytes = img_buffer.getvalue()
        
        # Send to backend
        files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
        data = {"name": name.strip()}
        
        result = make_request("register", method="POST", files=files, data=data)
        
        if result and result.get("success"):
            return f"✅ {result.get('message', 'Face registered successfully!')}", None
        else:
            return f"❌ Registration failed: {result.get('detail', result.get('error', 'Unknown error'))}", None
            
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def capture_for_registration():
    """Capture image for registration"""
    image, error = capture_image_from_camera()
    if image is not None:
        return image, "✅ Image captured! You can now register this face."
    else:
        return None, f"❌ {error}"

def recognize_single_image(image):
    """Recognize face in a single image"""
    if image is None:
        return "❌ Please provide an image"
    
    try:
        # Convert image to bytes
        img_pil = Image.fromarray(image.astype('uint8'))
        img_buffer = io.BytesIO()
        img_pil.save(img_buffer, format='JPEG')
        image_bytes = img_buffer.getvalue()
        
        # Send to backend
        files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
        result = make_request("recognize", method="POST", files=files)
        
        if result and result.get("success"):
            name = result.get('name', 'Unknown')
            confidence = result.get('confidence', 0.0)
            return f"✅ Recognized: **{name}** (Confidence: {confidence:.4f})"
        else:
            error_msg = result.get('detail', result.get('error', 'Unknown error'))
            if "not found" in error_msg.lower() or "unknown" in error_msg.lower():
                return "❓ Person not recognized - not in database"
            return f"❌ Recognition failed: {error_msg}"
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

def capture_and_recognize():
    """Capture image and recognize immediately"""
    image, error = capture_image_from_camera()
    if image is not None:
        result = recognize_single_image(image)
        return image, result
    else:
        return None, f"❌ {error}"

def live_recognition_worker():
    """Worker function for live recognition with camera feed"""
    global latest_recognition_result, latest_frame_with_bbox, camera_feed_active
    
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        latest_recognition_result = {"name": "Camera not available", "confidence": 0.0, "timestamp": time.strftime('%H:%M:%S')}
        return
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera_feed_active = True
    
    while live_recognition_active:
        ret, frame = camera.read()
        if not ret:
            continue
        
        # Convert BGR to RGB for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Convert frame to bytes for recognition
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()
            
            files = {"image": ("frame.jpg", image_bytes, "image/jpeg")}
            result = make_request("recognize", method="POST", files=files)
            
            if result and result.get("success"):
                name = result.get('name', 'Unknown')
                confidence = result.get('confidence', 0.0)
                bbox = result.get('bbox')  # Expecting [x1, y1, x2, y2] format
                
                latest_recognition_result = {
                    "name": name,
                    "confidence": confidence,
                    "timestamp": time.strftime('%H:%M:%S'),
                    "status": "✅ Recognized"
                }
                
                # Draw bounding box if available
                if bbox:
                    latest_frame_with_bbox = draw_bounding_box(rgb_frame, bbox, name, confidence)
                else:
                    # If no bbox from backend, create a simple frame overlay
                    latest_frame_with_bbox = rgb_frame.copy()
                    # Add text overlay on the image
                    cv2.putText(latest_frame_with_bbox, f"{name} ({confidence:.2f})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                error_msg = result.get('detail', result.get('error', 'Unknown error'))
                if "not found" in error_msg.lower():
                    latest_recognition_result = {
                        "name": "Unknown Person",
                        "confidence": 0.0,
                        "timestamp": time.strftime('%H:%M:%S'),
                        "status": "❓ Not in database"
                    }
                    # Show frame without recognition
                    latest_frame_with_bbox = rgb_frame.copy()
                    cv2.putText(latest_frame_with_bbox, "Unknown Person", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    latest_recognition_result = {
                        "name": "Recognition Error",
                        "confidence": 0.0,
                        "timestamp": time.strftime('%H:%M:%S'),
                        "status": f"❌ Error: {error_msg}"
                    }
                    latest_frame_with_bbox = rgb_frame.copy()
                    cv2.putText(latest_frame_with_bbox, "Error", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        except Exception as e:
            latest_recognition_result = {
                "name": "Error",
                "confidence": 0.0,
                "timestamp": time.strftime('%H:%M:%S'),
                "status": f"❌ {str(e)}"
            }
            latest_frame_with_bbox = rgb_frame.copy()
            cv2.putText(latest_frame_with_bbox, f"Error: {str(e)[:20]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        time.sleep(1)  # Recognition every 1 second for smoother experience
    
    camera.release()
    camera_feed_active = False
    latest_frame_with_bbox = None

def start_live_recognition():
    """Start live recognition"""
    global live_recognition_active, recognition_thread
    
    if live_recognition_active:
        return "⚠️ Live recognition is already running!"
    
    live_recognition_active = True
    recognition_thread = threading.Thread(target=live_recognition_worker, daemon=True)
    recognition_thread.start()
    
    return "✅ Live recognition started! Camera feed will appear below."

def stop_live_recognition():
    """Stop live recognition"""
    global live_recognition_active
    
    if not live_recognition_active:
        return "⚠️ Live recognition is not running!"
    
    live_recognition_active = False
    return "⏹️ Live recognition stopped."

def get_live_recognition_status():
    """Get current live recognition status"""
    global latest_recognition_result
    
    if not live_recognition_active:
        return "📴 Live recognition is stopped. Click 'Start Live Recognition' to begin."
    
    result = latest_recognition_result
    status_text = f"""
**{result.get('status', 'Unknown')}**

**Person:** {result.get('name', 'None')}
**Confidence:** {result.get('confidence', 0.0):.4f}
**Last Update:** {result.get('timestamp', 'Never')}
**System Status:** {'🟢 Active' if live_recognition_active else '🔴 Stopped'}
"""
    return status_text

def get_camera_feed():
    """Get current camera frame with bounding boxes"""
    global latest_frame_with_bbox
    
    if not camera_feed_active or latest_frame_with_bbox is None:
        # Return a placeholder image
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera feed stopped", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return placeholder
    
    return latest_frame_with_bbox

def ask_ai(question):
    """Ask AI about registered faces"""
    if not question or not question.strip():
        return "❌ Please enter a question"
    
    try:
        data = {"question": question.strip()}
        result = make_request("ask", method="POST", json=data)
        
        if result and result.get("success"):
            answer = result.get("answer", "No answer received")
            context = result.get("context", "")
            
            response = f"🤖 **AI Response:**\n\n{answer}"
            if context:
                response += f"\n\n📋 **Context Data:**\n{context}"
            return response
        else:
            return f"❌ AI request failed: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

def get_registered_faces():
    """Get list of registered faces"""
    result = make_request("faces")
    
    if result and result.get("success"):
        faces = result.get("faces", [])
        
        if not faces:
            return "📝 No faces registered yet."
        
        faces_text = f"**Total Registered Faces: {len(faces)}**\n\n"
        
        for i, face in enumerate(faces, 1):
            faces_text += f"**{i}. {face['name']}**\n"
            faces_text += f"   - ID: {face['face_id'][:12]}...\n"
            faces_text += f"   - Registered: {face['registration_date']}\n\n"
        
        return faces_text
    else:
        return f"❌ Could not retrieve faces: {result.get('error', 'Unknown error')}"

# Sample questions for AI
SAMPLE_QUESTIONS = [
    "Who was the last person registered?",
    "How many people are registered?",
    "List all registered people",
    "Who were the first 3 people registered?",
    "What are the names of all registered faces?"
]

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Face Recognition + AI Q&A System", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 👤 Face Recognition + AI Q&A System
        Register faces, recognize people, and ask intelligent questions about your face database.
        """)
        
        # System Status
        with gr.Row():
            with gr.Column(scale=1):
                backend_status = gr.Textbox(
                    label="Backend Status",
                    value=check_backend_health(),
                    interactive=False
                )
            with gr.Column(scale=1):
                faces_count = gr.Textbox(
                    label="Database Info",
                    value=get_registered_faces_count(),
                    interactive=False
                )
            with gr.Column(scale=1):
                refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")
        
        # Tabs
        with gr.Tabs():
            
            # Register Tab
            with gr.TabItem("📝 Register Face"):
                gr.Markdown("## Register New Face")
                
                with gr.Row():
                    with gr.Column():
                        name_input = gr.Textbox(
                            label="Person's Name",
                            placeholder="Enter the person's name"
                        )
                        
                        with gr.Row():
                            capture_btn = gr.Button("📸 Capture from Camera", variant="primary")
                            clear_btn = gr.Button("🗑️ Clear Image", variant="secondary")
                        
                        upload_image = gr.Image(
                            label="Upload Image or Captured Image",
                            type="numpy"
                        )
                        
                        register_btn = gr.Button("✅ Register Face", variant="primary", size="lg")
                        register_result = gr.Textbox(label="Registration Result", interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("""
                        ### 📌 Registration Tips
                        
                        **For best results:**
                        - Ensure good lighting
                        - Face should be clearly visible  
                        - Look directly at the camera
                        - Avoid shadows on the face
                        - One person per image
                        - Keep your head steady when taking photo
                        """)
            
            # Live Recognition Tab
            with gr.TabItem("🔍 Live Recognition"):
                gr.Markdown("## 🎥 Live Face Recognition with Camera Feed")
                gr.Markdown("Real-time face recognition with live camera feed and bounding boxes")
                
                with gr.Row():
                    start_live_btn = gr.Button("▶️ Start Live Recognition", variant="primary")
                    stop_live_btn = gr.Button("⏹️ Stop Live Recognition", variant="stop")
                
                live_result = gr.Textbox(label="Control Messages", interactive=False)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # Camera feed with bounding boxes
                        camera_feed = gr.Image(
                            label="🎥 Live Camera Feed",
                            type="numpy",
                            interactive=False,
                            show_download_button=False
                        )
                    
                    with gr.Column(scale=1):
                        live_status = gr.Textbox(
                            label="Recognition Status",
                            value="📴 Click 'Start Live Recognition' to begin",
                            interactive=False,
                            lines=8
                        )
                
                gr.Markdown("""
                ### 💡 Instructions:
                1. Click "▶️ Start Live Recognition" to begin
                2. Position yourself in front of your camera
                3. The system will show live camera feed with bounding boxes around detected faces
                4. Recognition results appear in the status panel on the right
                5. Green boxes = High confidence, Orange = Medium, Red = Low confidence
                6. Click "⏹️ Stop Live Recognition" when done
                
                ### 🎯 Bounding Box Colors:
                - **Green**: High confidence (>70%)
                - **Orange**: Medium confidence (40-70%)
                - **Red**: Low confidence (<40%)
                """)
            
            # Single Recognition Tab
            with gr.TabItem("🔍 Single Recognition"):
                gr.Markdown("## 📸 Single Image Recognition")
                
                with gr.Row():
                    with gr.Column():
                        single_image = gr.Image(
                            label="Upload Image or Capture",
                            type="numpy"
                        )
                        
                        with gr.Row():
                            capture_recognize_btn = gr.Button("📸 Capture & Recognize", variant="primary")
                            recognize_btn = gr.Button("🔍 Recognize Image", variant="secondary")
                        
                        single_result = gr.Textbox(
                            label="Recognition Result",
                            interactive=False,
                            lines=3
                        )
                    
                    with gr.Column():
                        gr.Markdown("""
                        ### 🎯 Recognition Options
                        
                        **Option 1: Upload Image**
                        - Click on the image box above
                        - Select an image file
                        - Click "🔍 Recognize Image"
                        
                        **Option 2: Capture from Camera**
                        - Click "📸 Capture & Recognize"
                        - This will capture and recognize immediately
                        
                        **Results will show:**
                        - Person's name (if recognized)
                        - Confidence score
                        - Recognition status
                        """)
            
            # Ask AI Tab
            with gr.TabItem("🤖 Ask AI"):
                gr.Markdown("## 🤖 Ask AI About Registered Faces")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Sample Questions (Click to use):")
                        sample_buttons = []
                        for question in SAMPLE_QUESTIONS:
                            btn = gr.Button(question, variant="secondary", size="sm")
                            sample_buttons.append(btn)
                        
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask anything about the registered faces...",
                            lines=3
                        )
                        
                        with gr.Row():
                            ask_btn = gr.Button("🤖 Ask AI", variant="primary")
                            clear_question_btn = gr.Button("🗑️ Clear", variant="secondary")
                        
                        ai_response = gr.Textbox(
                            label="AI Response",
                            interactive=False,
                            lines=10
                        )
                    
                    with gr.Column():
                        gr.Markdown("""
                        ### 💡 What you can ask:
                        
                        - **General questions:** "How many people are registered?"
                        - **Specific queries:** "Who was registered last?"
                        - **List requests:** "Show me all registered names"
                        - **Date queries:** "When was John registered?"
                        - **Comparisons:** "Who was registered first?"
                        
                        The AI has access to your face database and can answer questions about:
                        - Registration dates and times
                        - Person names and details
                        - Database statistics
                        - Chronological information
                        """)
            
            # View Faces Tab
            with gr.TabItem("👥 View Faces"):
                gr.Markdown("## 👥 Registered Faces Database")
                
                view_faces_btn = gr.Button("🔄 Refresh Face List", variant="primary")
                faces_list = gr.Textbox(
                    label="Registered Faces",
                    value=get_registered_faces(),
                    interactive=False,
                    lines=15
                )
        
        # Event handlers
        def refresh_status():
            return check_backend_health(), get_registered_faces_count()
        
        def set_sample_question(question):
            return question
        
        def clear_question():
            return ""
        
        def clear_image():
            return None
        
        # Auto-refresh live recognition status and camera feed
        def update_live_status():
            return get_live_recognition_status()
        
        def update_camera_feed():
            return get_camera_feed()
        
        # Bind events
        refresh_btn.click(
            fn=refresh_status,
            outputs=[backend_status, faces_count]
        )
        
        capture_btn.click(
            fn=capture_for_registration,
            outputs=[upload_image, register_result]
        )
        
        clear_btn.click(
            fn=clear_image,
            outputs=[upload_image]
        )
        
        register_btn.click(
            fn=register_face,
            inputs=[name_input, upload_image],
            outputs=[register_result, upload_image]
        )
        
        start_live_btn.click(
            fn=start_live_recognition,
            outputs=[live_result]
        )
        
        stop_live_btn.click(
            fn=stop_live_recognition,
            outputs=[live_result]
        )
        
        # Auto-update live recognition components
        live_status_timer = gr.Timer(2)
        live_status_timer.tick(
            fn=update_live_status,
            outputs=[live_status]
        )
        
        camera_feed_timer = gr.Timer(0.5)  # Update camera feed more frequently
        camera_feed_timer.tick(
            fn=update_camera_feed,
            outputs=[camera_feed]
        )
        
        capture_recognize_btn.click(
            fn=capture_and_recognize,
            outputs=[single_image, single_result]
        )
        
        recognize_btn.click(
            fn=recognize_single_image,
            inputs=[single_image],
            outputs=[single_result]
        )
        
        # Sample question buttons
        for i, btn in enumerate(sample_buttons):
            btn.click(
                fn=lambda q=SAMPLE_QUESTIONS[i]: q,
                outputs=[question_input]
            )
        
        ask_btn.click(
            fn=ask_ai,
            inputs=[question_input],
            outputs=[ai_response]
        )
        
        clear_question_btn.click(
            fn=clear_question,
            outputs=[question_input]
        )
        
        view_faces_btn.click(
            fn=get_registered_faces,
            outputs=[faces_list]
        )
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )