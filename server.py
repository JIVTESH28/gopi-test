from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from backend.engineV2 import FaceRecognitionEngine
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Face Recognition + RAG Q&A System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the face recognition engine
face_engine = FaceRecognitionEngine()

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class FaceData(BaseModel):
    name: str
    registration_date: str
    face_id: str

# API Endpoints
@app.on_event("startup")
async def startup_event():
    print("Starting up application...")
    success = await face_engine.initialize()
    if success:
        print("Face recognition engine initialized successfully")
        # Run migration from pickle to SafeTensors
        print("Checking for pickle to SafeTensors migration...")
        migrated_count = face_engine.migrate_pickle_to_safetensors()
        if migrated_count > 0:
            print(f"Migrated {migrated_count} face encodings to SafeTensors format")
    else:
        print("Warning: Face recognition engine initialization failed")
    print("Application startup completed")

@app.post("/register")
async def register_face(
    name: str = Form(...),
    image: UploadFile = File(...)
):
    """Register a new face"""
    try:
        # Read image data
        image_data = await image.read()
        
        # Use engine to register face
        result = await face_engine.register_face(name, image_data)
        
        if result["success"]:
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=400, detail=result["error"])
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/recognize")
async def recognize_face(image: UploadFile = File(...)):
    """Recognize a face"""
    try:
        # Read image data
        image_data = await image.read()
        
        # Use engine to recognize face
        result = await face_engine.recognize_face(image_data)
        
        return JSONResponse(content=result)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Answer questions about registered faces using RAG"""
    try:
        result = await face_engine.ask_question(request.question)
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": f"Error processing question: {str(e)}",
            "answer": "Sorry, I encountered an error while processing your question."
        })

@app.get("/faces")
async def list_faces():
    """List all registered faces"""
    try:
        result = face_engine.get_all_faces()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving faces: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        health_data = face_engine.health_check()
        return health_data
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/migrate")
async def migrate_encodings():
    """Manually trigger migration from pickle to SafeTensors"""
    try:
        migrated_count = face_engine.migrate_pickle_to_safetensors()
        return JSONResponse(content={
            "success": True,
            "message": f"Migration completed. Migrated {migrated_count} faces from pickle to SafeTensors format."
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")

@app.post("/reset")
async def reset_database():
    """Reset all data (use with caution)"""
    try:
        result = face_engine.reset_database()
        if result["success"]:
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting database: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)