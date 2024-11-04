import os
import PyPDF2
import faiss
import numpy as np
import pickle
from openai import OpenAI
from typing import Dict, List
from dotenv import load_dotenv
from pymongo import MongoClient
from moviepy.editor import VideoFileClip
from fastapi import HTTPException, status
from sentence_transformers import SentenceTransformer
import whisper

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
mongo_client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
mongo_db = mongo_client['alphagpt']

VECTOR_DIMENSION = 384
FAISS_INDEX_PATH = "faiss_index.bin"
DOCUMENTS_PATH = "stored_documents.pkl"

# Initialize or load FAISS index
if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_binary(FAISS_INDEX_PATH)
    with open(DOCUMENTS_PATH, 'rb') as f:
        stored_documents = pickle.load(f)
else:
    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    stored_documents = []

def save_faiss_state():
    """Save FAISS index and documents to disk."""
    faiss.write_binary(index, FAISS_INDEX_PATH)
    with open(DOCUMENTS_PATH, 'wb') as f:
        pickle.dump(stored_documents, f)
    print("FAISS state saved successfully!")

def extract_text_from_video(video_path: str) -> str:
    """Extracts and transcribes audio from video using Whisper base model."""
    try:
        whisper_model = whisper.load_model("base")
        video = VideoFileClip(video_path)
        audio = video.audio
        audio_path = "temp_audio.wav"
        audio.write_audiofile(audio_path)
        
        result = whisper_model.transcribe(audio_path)
        
        os.remove(audio_path)
        return result["text"]
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return ""

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        return ""

def process_folder_content(folder_path: str, content_type: str) -> List[Dict]:
    """Processes folder contents based on content type (video, pdf, text)."""
    content_data = []
    
    if not os.path.exists(folder_path):
        return content_data
        
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if content_type == 'video' and filename.endswith(('.mp4', '.avi', '.mov')):
            text_content = extract_text_from_video(file_path)
        elif content_type == 'pdf' and filename.endswith('.pdf'):
            text_content = extract_text_from_pdf(file_path)
        elif content_type == 'text' and filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                text_content = file.read()
        else:
            continue
            
        if text_content:
            embedding = model.encode(text_content)
            content_data.append({
                'filename': filename,
                'content': text_content,
                'embedding': embedding.tolist(),
                'type': content_type
            })
    
    return content_data

def initialize_local_vectordb(content_data: List[Dict]):
    """Initializes local FAISS vector database with embeddings."""
    global index, stored_documents
    
    if not content_data:
        return
        
    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    stored_documents = []
    
    vectors = [np.array(item['embedding'], dtype=np.float32) for item in content_data]
    index.add(np.vstack(vectors))
    
    stored_documents.extend(content_data)
    
    # Save state to disk
    save_faiss_state()
    print("Local vector database initialized successfully!")

def create_vector_database():
    """Process content and initialize local vector store."""
    try:
        # Process content from different sources
        video_data = process_folder_content('videos', 'video')
        pdf_data = process_folder_content('documents/pdfs', 'pdf')
        text_data = process_folder_content('documents/texts', 'text')
        
        all_content = video_data + pdf_data + text_data
        
        if all_content:
            initialize_local_vectordb(all_content)
            print("Vector database created successfully!")
        else:
            print("No content found to process!")
            
    except Exception as e:
        print(f"Error creating vector database: {str(e)}")

def get_platform_type(query: str) -> str:
    db_keywords = ['find', 'search', 'lookup', 'get', 'query', 'database', 
                  'record', 'entry', 'status', 'order', 'account']
    
    if any(keyword in query.lower() for keyword in db_keywords):
        return "mongodb"
    return "vector"

def search_mongodb_platform(query: str, collection_name: str) -> str:
    """
    Search MongoDB in a specified collection.
    
    Args:
        query: Search query string
        collection_name: Name of the MongoDB collection to search in
    """
    try:
        if not collection_name:
            return "No collection name provided for MongoDB search."
            
        collection = mongo_db[collection_name]
        keywords = query.lower().split()
        
        content_results = list(collection.find({
            "$or": [
                {"$text": {"$search": " ".join(keywords)}},  # If text index exists
                {"$or": [
                    {field: {"$regex": "|".join(keywords), "$options": "i"}} 
                    for field in collection.find_one({}).keys()  # Search all fields
                    if field != "_id"
                ]}
            ]
        }).limit(1))
        
        if content_results:
            result = content_results[0]
            if '_id' in result:
                del result['_id']
            return str(result)
            
        return f"No relevant information found in collection '{collection_name}'."
        
    except Exception as e:
        return f"Error searching MongoDB collection '{collection_name}': {str(e)}"

def search_vector_database(query: str) -> str:
    """Search local vector database using FAISS."""
    try:
        if not stored_documents:
            return "No content available in vector database."
        
        query_vector = model.encode(query)
        query_vector = np.array([query_vector]).astype('float32')
        
        _, I = index.search(query_vector, 1)
        
        if I[0][0] >= len(stored_documents):
            return "No relevant content found."
            
        best_match = stored_documents[I[0][0]]
        return best_match['content']
        
    except Exception as e:
        return f"Error searching vector database: {str(e)}"

def generate_response(context: str, query: str) -> str:
    """Generate a response using OpenAI GPT model."""
    try:
        messages = [
            {"role": "system", "content": "You are a helpful customer support assistant chatbot. Provide clear, concise answers based only on the context provided. Also provide a clear and helpful response to the user's question."},
            {"role": "user", "content": f"""
                Based on the following context, provide a clear and helpful response to the user's question.
                Keep the response natural and conversational. Only use information from the context to answer.
                If the context doesn't contain relevant information, say so politely.

                Context: {context}

                User Question: {query}
            """}
        ]

        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            n=1,
            stop=None,
            temperature=0.5
        )
        
        return response.choices[0].message.content
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(err)}"
        )