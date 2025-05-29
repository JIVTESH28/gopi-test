import face_recognition
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file
import cv2
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
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
import tempfile

class FaceRecognitionEngine:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Configuration - Load from environment variables
        self.MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
        self.QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
        self.QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.COLLECTION_NAME = os.getenv("COLLECTION_NAME", "face_embeddings")
        
        # Validate required environment variables
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file.")
        
        # Initialize connections (will be done in initialize method)
        self.mongo_client = None
        self.db = None
        self.faces_collection = None
        self.qdrant_client = None
        self.llm = None

    async def initialize(self):
        """Initialize all connections and databases"""
        try:
            # Initialize MongoDB
            self.mongo_client = MongoClient(self.MONGODB_URL)
            self.db = self.mongo_client.face_recognition_db
            self.faces_collection = self.db.faces
            
            # Initialize Qdrant
            self.qdrant_client = QdrantClient(host=self.QDRANT_URL, port=self.QDRANT_PORT)
            
            # Initialize OpenAI
            self.llm = ChatOpenAI(
                model="gpt-4",
                openai_api_key=self.OPENAI_API_KEY,
                temperature=0.5
            )
            
            # Initialize databases
            qdrant_success = self.initialize_qdrant()
            mongo_success = self.initialize_mongodb()
            
            return qdrant_success and mongo_success
            
        except Exception as e:
            print(f"Error initializing engine: {e}")
            return False

    def initialize_qdrant(self):
        """Initialize Qdrant collection"""
        try:
            # Try to get collection info first
            try:
                collection_info = self.qdrant_client.get_collection(self.COLLECTION_NAME)
                print(f"Qdrant collection {self.COLLECTION_NAME} already exists")
                return True
            except Exception:
                # Collection doesn't exist, create it
                print(f"Creating Qdrant collection: {self.COLLECTION_NAME}")
                
            self.qdrant_client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=128, distance=Distance.COSINE),
            )
            print(f"Successfully created Qdrant collection: {self.COLLECTION_NAME}")
            return True
            
        except Exception as e:
            print(f"Error initializing Qdrant: {e}")
            return False

    def ensure_collection_exists(self):
        """Ensure collection exists before operations"""
        try:
            self.qdrant_client.get_collection(self.COLLECTION_NAME)
            return True
        except Exception:
            print(f"Collection {self.COLLECTION_NAME} doesn't exist, creating...")
            return self.initialize_qdrant()

    def initialize_mongodb(self):
        """Initialize MongoDB collections and indexes"""
        try:
            # Create index on name for faster queries
            self.faces_collection.create_index("name")
            self.faces_collection.create_index("registration_date")
            print("MongoDB initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing MongoDB: {e}")
            return False

    # Utility functions for SafeTensors
    def numpy_to_safetensor_bytes(self, arr: np.ndarray) -> bytes:
        """Convert numpy array to SafeTensors bytes"""
        # Convert numpy array to torch tensor
        tensor = torch.from_numpy(arr.copy())
        
        # Create a temporary file to save SafeTensors data
        with tempfile.NamedTemporaryFile() as tmp_file:
            # Save tensor as SafeTensors
            save_file({"face_encoding": tensor}, tmp_file.name)
            
            # Read the SafeTensors file content
            with open(tmp_file.name, 'rb') as f:
                return f.read()

    def safetensor_bytes_to_numpy(self, data: bytes) -> np.ndarray:
        """Convert SafeTensors bytes back to numpy array"""
        # Create a temporary file to load SafeTensors data
        with tempfile.NamedTemporaryFile() as tmp_file:
            # Write bytes to temporary file
            with open(tmp_file.name, 'wb') as f:
                f.write(data)
            
            # Load tensor from SafeTensors
            tensors = load_file(tmp_file.name)
            tensor = tensors["face_encoding"]
            
            # Convert torch tensor back to numpy array
            return tensor.numpy()

    def process_image(self, image_data):
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

    def save_face_encoding_to_mongo(self, face_id: str, name: str, encoding: np.ndarray):
        """Save face encoding as SafeTensors to MongoDB"""
        try:
            # Convert numpy array to SafeTensors bytes
            encoding_safetensors = self.numpy_to_safetensor_bytes(encoding)
            
            face_doc = {
                "_id": face_id,
                "name": name,
                "encoding_safetensors": encoding_safetensors,
                "registration_date": datetime.now(),
                "created_at": datetime.now(),
                "encoding_format": "safetensors"  # Add format indicator
            }
            
            self.faces_collection.insert_one(face_doc)
            return True
        except Exception as e:
            print(f"Error saving to MongoDB: {e}")
            return False

    def load_face_encoding_from_mongo(self, face_id: str) -> np.ndarray:
        """Load face encoding from MongoDB SafeTensors data"""
        try:
            doc = self.faces_collection.find_one({"_id": face_id})
            if not doc:
                return None
            
            # Check if it's the new SafeTensors format
            if "encoding_safetensors" in doc:
                encoding_bytes = doc["encoding_safetensors"]
                return self.safetensor_bytes_to_numpy(encoding_bytes)
            
            # Fallback for old pickle format (for backward compatibility)
            elif "encoding_pickle" in doc:
                import pickle
                print(f"Warning: Loading legacy pickle format for face_id {face_id}")
                return pickle.loads(doc["encoding_pickle"])
            
            else:
                print(f"No encoding found for face_id {face_id}")
                return None
                
        except Exception as e:
            print(f"Error loading face encoding: {e}")
            return None

    def search_similar_faces(self, query_encoding: np.ndarray, threshold: float = 0.6):
        """Search for similar faces in Qdrant"""
        try:
            # Ensure collection exists before searching
            if not self.ensure_collection_exists():
                print("Failed to ensure collection exists")
                return None
                
            search_result = self.qdrant_client.search(
                collection_name=self.COLLECTION_NAME,
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

    def get_face_metadata_from_mongo(self, face_ids: List[str]) -> List[Dict]:
        """Retrieve face metadata from MongoDB"""
        try:
            cursor = self.faces_collection.find({"_id": {"$in": face_ids}})
            metadata = []
            for doc in cursor:
                metadata.append({
                    "name": doc["name"],
                    "registration_date": doc["registration_date"].strftime("%Y-%m-%d %H:%M:%S"),
                    "face_id": doc["_id"],
                    "encoding_format": doc.get("encoding_format", "legacy_pickle")
                })
            return metadata
        except Exception as e:
            print(f"Error retrieving metadata: {e}")
            return []

    def get_all_faces_metadata(self) -> List[Dict]:
        """Get all registered faces metadata"""
        try:
            cursor = self.faces_collection.find({})
            metadata = []
            for doc in cursor:
                metadata.append({
                    "name": doc["name"],
                    "registration_date": doc["registration_date"].strftime("%Y-%m-%d %H:%M:%S"),
                    "face_id": doc["_id"],
                    "encoding_format": doc.get("encoding_format", "legacy_pickle")
                })
            return sorted(metadata, key=lambda x: x["registration_date"], reverse=True)
        except Exception as e:
            print(f"Error retrieving all metadata: {e}")
            return []

    def migrate_pickle_to_safetensors(self):
        """Migrate existing pickle encodings to SafeTensors format"""
        try:
            # Find all documents with pickle encodings
            pickle_docs = self.faces_collection.find({"encoding_pickle": {"$exists": True}, "encoding_safetensors": {"$exists": False}})
            
            migrated_count = 0
            for doc in pickle_docs:
                face_id = doc["_id"]
                try:
                    # Load pickle encoding
                    import pickle
                    encoding = pickle.loads(doc["encoding_pickle"])
                    
                    # Convert to SafeTensors
                    encoding_safetensors = self.numpy_to_safetensor_bytes(encoding)
                    
                    # Update document
                    self.faces_collection.update_one(
                        {"_id": face_id},
                        {
                            "$set": {
                                "encoding_safetensors": encoding_safetensors,
                                "encoding_format": "safetensors"
                            },
                            "$unset": {"encoding_pickle": ""}  # Remove old pickle data
                        }
                    )
                    
                    migrated_count += 1
                    print(f"Migrated face_id {face_id} from pickle to SafeTensors")
                    
                except Exception as e:
                    print(f"Failed to migrate face_id {face_id}: {e}")
            
            print(f"Migration completed. Migrated {migrated_count} faces from pickle to SafeTensors")
            return migrated_count
            
        except Exception as e:
            print(f"Error during migration: {e}")
            return 0

    # Main API Functions
    async def register_face(self, name: str, image_data: bytes):
        """Register a new face"""
        try:
            # Process image and get face encoding
            encoding, error = self.process_image(image_data)
            if error:
                return {"success": False, "error": error}
            
            # Generate unique ID
            face_id = str(uuid.uuid4())
            
            # Ensure Qdrant collection exists
            if not self.ensure_collection_exists():
                return {"success": False, "error": "Failed to initialize vector database"}
            
            # Save to MongoDB first (now using SafeTensors)
            if not self.save_face_encoding_to_mongo(face_id, name, encoding):
                return {"success": False, "error": "Failed to save face data to database"}
            
            # Save to Qdrant
            try:
                point = PointStruct(
                    id=face_id,
                    vector=encoding.tolist(),
                    payload={"name": name, "mongo_id": face_id}
                )
                
                self.qdrant_client.upsert(
                    collection_name=self.COLLECTION_NAME,
                    points=[point]
                )
            except Exception as e:
                # If Qdrant fails, remove from MongoDB to maintain consistency
                self.faces_collection.delete_one({"_id": face_id})
                return {"success": False, "error": f"Failed to save to vector database: {str(e)}"}
            
            return {
                "success": True,
                "message": f"Face registered successfully for {name}",
                "face_id": face_id,
                "encoding_format": "safetensors"
            }
            
        except Exception as e:
            return {"success": False, "error": f"Internal server error: {str(e)}"}

    async def recognize_face(self, image_data: bytes):
        """Recognize a face"""
        try:
            # Process image and get face encoding
            encoding, error = self.process_image(image_data)
            if error:
                return {"success": False, "error": error}
            
            # Search for similar face
            result = self.search_similar_faces(encoding)
            
            if result:
                return {
                    "success": True,
                    "name": result.payload["name"],
                    "confidence": float(result.score),
                    "face_id": result.payload["mongo_id"]
                }
            else:
                return {
                    "success": False,
                    "message": "No matching face found",
                    "name": "Unknown"
                }
                
        except Exception as e:
            return {"success": False, "error": f"Internal server error: {str(e)}"}

    async def ask_question(self, question: str):
        """Answer questions about registered faces using RAG"""
        try:
            # Get all faces metadata
            faces_metadata = self.get_all_faces_metadata()
            
            if not faces_metadata:
                return {
                    "success": True,
                    "answer": "No faces have been registered yet."
                }
            
            # Build context document
            context_parts = []
            for face in faces_metadata:
                context_parts.append(
                    f"Person: {face['name']}, Registered: {face['registration_date']}, ID: {face['face_id']}, Format: {face['encoding_format']}"
                )
            
            context = "Registered faces information:\n" + "\n".join(context_parts)
            
            # Create prompt for GPT
            prompt = f"""Based on the following registered faces information, please answer the user's question.

Context:
{context}

User Question: {question}

Please provide a helpful and accurate answer based only on the information provided above."""

            # Get response from OpenAI
            response = self.llm([HumanMessage(content=prompt)])
            
            return {
                "success": True,
                "answer": response.content,
                "context": context
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing question: {str(e)}",
                "answer": "Sorry, I encountered an error while processing your question."
            }

    def get_all_faces(self):
        """List all registered faces"""
        try:
            faces = self.get_all_faces_metadata()
            return {
                "success": True,
                "faces": faces,
                "count": len(faces)
            }
        except Exception as e:
            return {"success": False, "error": f"Error retrieving faces: {str(e)}"}

    def health_check(self):
        """Health check endpoint"""
        try:
            # Check MongoDB connection
            self.mongo_client.admin.command('ping')
            mongo_status = "connected"
        except Exception:
            mongo_status = "disconnected"
        
        try:
            # Check Qdrant connection
            self.qdrant_client.get_collections()
            qdrant_status = "connected"
        except Exception:
            qdrant_status = "disconnected"
        
        # Check encoding formats in database
        try:
            safetensors_count = self.faces_collection.count_documents({"encoding_format": "safetensors"})
            pickle_count = self.faces_collection.count_documents({"encoding_pickle": {"$exists": True}})
            total_faces = self.faces_collection.count_documents({})
        except Exception:
            safetensors_count = pickle_count = total_faces = 0
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "mongodb": mongo_status,
            "qdrant": qdrant_status,
            "database_stats": {
                "total_faces": total_faces,
                "safetensors_format": safetensors_count,
                "legacy_pickle_format": pickle_count
            }
        }

    def reset_database(self):
        """Reset all data (use with caution)"""
        try:
            # Clear MongoDB
            self.faces_collection.delete_many({})
            
            # Clear Qdrant
            try:
                self.qdrant_client.delete_collection(self.COLLECTION_NAME)
            except Exception:
                pass  # Collection might not exist
            
            # Recreate Qdrant collection
            self.initialize_qdrant()
            
            return {
                "success": True,
                "message": "Database reset successfully"
            }
        except Exception as e:
            return {"success": False, "error": f"Error resetting database: {str(e)}"}