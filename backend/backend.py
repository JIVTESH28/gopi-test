from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import face_recognition
import numpy as np
import pickle
import cv2
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
from pydantic import BaseModel
import os
from datetime import datetime
import uuid
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import json
from typing import List, Dict, Any
import base64
from io import BytesIO
from PIL import Image
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

# Configuration - Load from environment variables
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "face_embeddings")

# Validate required environment variables
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file.")

# Initialize connections
mongo_client = MongoClient(MONGODB_URL)
db = mongo_client.face_recognition_db
faces_collection = db.faces

qdrant_client = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)

# Initialize OpenAI
llm = ChatOpenAI(
    model="gpt-4",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.5
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class FaceData(BaseModel):
    name: str
    registration_date: str
    face_id: str

# Initialize Qdrant collection
def initialize_qdrant():
    try:
        # Try to get collection info first
        try:
            collection_info = qdrant_client.get_collection(COLLECTION_NAME)
            print(f"Qdrant collection {COLLECTION_NAME} already exists")
            return True
        except Exception:
            # Collection doesn't exist, create it
            print(f"Creating Qdrant collection: {COLLECTION_NAME}")
            
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=128, distance=Distance.COSINE),
        )
        print(f"Successfully created Qdrant collection: {COLLECTION_NAME}")
        return True
        
    except Exception as e:
        print(f"Error initializing Qdrant: {e}")
        return False

def ensure_collection_exists():
    """Ensure collection exists before operations"""
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        return True
    except Exception:
        print(f"Collection {COLLECTION_NAME} doesn't exist, creating...")
        return initialize_qdrant()

# Initialize MongoDB collections
def initialize_mongodb():
    """Initialize MongoDB collections and indexes"""
    try:
        # Create index on name for faster queries
        faces_collection.create_index("name")
        faces_collection.create_index("registration_date")
        print("MongoDB initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing MongoDB: {e}")
        return False

# Utility functions
def process_image(image_data):
    """Process image data and return face encodings"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return None, "Invalid image format"
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            return None, "No face detected in the image"
        
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        if not face_encodings:
            return None, "Could not encode face"
        
        return face_encodings[0], None
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

def save_face_encoding_to_mongo(face_id: str, name: str, encoding: np.ndarray):
    """Save face encoding as pickle to MongoDB"""
    try:
        # Serialize the encoding
        encoding_pickle = pickle.dumps(encoding)
        
        face_doc = {
            "_id": face_id,
            "name": name,
            "encoding_pickle": encoding_pickle,
            "registration_date": datetime.now(),
            "created_at": datetime.now()
        }
        
        faces_collection.insert_one(face_doc)
        return True
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")
        return False

def search_similar_faces(query_encoding: np.ndarray, threshold: float = 0.6):
    """Search for similar faces in Qdrant"""
    try:
        # Ensure collection exists before searching
        if not ensure_collection_exists():
            print("Failed to ensure collection exists")
            return None
            
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_encoding.tolist(),
            limit=1,
            score_threshold=threshold
        )
        
        if search_result and len(search_result) > 0:
            return search_result[0]
        return None
    except Exception as e:
        print(f"Error searching in Qdrant: {e}")
        return None

def get_face_metadata_from_mongo(face_ids: List[str]) -> List[Dict]:
    """Retrieve face metadata from MongoDB"""
    try:
        cursor = faces_collection.find({"_id": {"$in": face_ids}})
        metadata = []
        for doc in cursor:
            metadata.append({
                "name": doc["name"],
                "registration_date": doc["registration_date"].strftime("%Y-%m-%d %H:%M:%S"),
                "face_id": doc["_id"]
            })
        return metadata
    except Exception as e:
        print(f"Error retrieving metadata: {e}")
        return []

def get_all_faces_metadata() -> List[Dict]:
    """Get all registered faces metadata"""
    try:
        cursor = faces_collection.find({})
        metadata = []
        for doc in cursor:
            metadata.append({
                "name": doc["name"],
                "registration_date": doc["registration_date"].strftime("%Y-%m-%d %H:%M:%S"),
                "face_id": doc["_id"]
            })
        return sorted(metadata, key=lambda x: x["registration_date"], reverse=True)
    except Exception as e:
        print(f"Error retrieving all metadata: {e}")
        return []

# API Endpoints
@app.on_event("startup")
async def startup_event():
    print("Starting up application...")
    qdrant_success = initialize_qdrant()
    mongo_success = initialize_mongodb()
    
    if not qdrant_success:
        print("Warning: Qdrant initialization failed")
    if not mongo_success:
        print("Warning: MongoDB initialization failed")
    
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
        
        # Process image and get face encoding
        encoding, error = process_image(image_data)
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        # Generate unique ID
        face_id = str(uuid.uuid4())
        
        # Ensure Qdrant collection exists
        if not ensure_collection_exists():
            raise HTTPException(status_code=500, detail="Failed to initialize vector database")
        
        # Save to MongoDB first
        if not save_face_encoding_to_mongo(face_id, name, encoding):
            raise HTTPException(status_code=500, detail="Failed to save face data to database")
        
        # Save to Qdrant
        try:
            point = PointStruct(
                id=face_id,
                vector=encoding.tolist(),
                payload={"name": name, "mongo_id": face_id}
            )
            
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[point]
            )
        except Exception as e:
            # If Qdrant fails, remove from MongoDB to maintain consistency
            faces_collection.delete_one({"_id": face_id})
            raise HTTPException(status_code=500, detail=f"Failed to save to vector database: {str(e)}")
        
        return JSONResponse(content={
            "success": True,
            "message": f"Face registered successfully for {name}",
            "face_id": face_id
        })
        
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
        
        # Process image and get face encoding
        encoding, error = process_image(image_data)
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        # Search for similar face
        result = search_similar_faces(encoding)
        
        if result:
            return JSONResponse(content={
                "success": True,
                "name": result.payload["name"],
                "confidence": float(result.score),
                "face_id": result.payload["mongo_id"]
            })
        else:
            return JSONResponse(content={
                "success": False,
                "message": "No matching face found",
                "name": "Unknown"
            })
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Answer questions about registered faces using RAG"""
    try:
        # Get all faces metadata
        faces_metadata = get_all_faces_metadata()
        
        if not faces_metadata:
            return JSONResponse(content={
                "success": True,
                "answer": "No faces have been registered yet."
            })
        
        # Build context document
        context_parts = []
        for face in faces_metadata:
            context_parts.append(
                f"Person: {face['name']}, Registered: {face['registration_date']}, ID: {face['face_id']}"
            )
        
        context = "Registered faces information:\n" + "\n".join(context_parts)
        
        # Create prompt for GPT
        prompt = f"""Based on the following registered faces information, please answer the user's question.

Context:
{context}

User Question: {request.question}

Please provide a helpful and accurate answer based only on the information provided above."""

        # Get response from OpenAI
        response = llm([HumanMessage(content=prompt)])
        
        return JSONResponse(content={
            "success": True,
            "answer": response.content,
            "context": context
        })
        
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
        faces = get_all_faces_metadata()
        return JSONResponse(content={
            "success": True,
            "faces": faces,
            "count": len(faces)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving faces: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check MongoDB connection
        mongo_client.admin.command('ping')
        mongo_status = "connected"
    except Exception:
        mongo_status = "disconnected"
    
    try:
        # Check Qdrant connection
        qdrant_client.get_collections()
        qdrant_status = "connected"
    except Exception:
        qdrant_status = "disconnected"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mongodb": mongo_status,
        "qdrant": qdrant_status
    }

@app.post("/reset")
async def reset_database():
    """Reset all data (use with caution)"""
    try:
        # Clear MongoDB
        faces_collection.delete_many({})
        
        # Clear Qdrant
        try:
            qdrant_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # Collection might not exist
        
        # Recreate Qdrant collection
        initialize_qdrant()
        
        return JSONResponse(content={
            "success": True,
            "message": "Database reset successfully"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting database: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)