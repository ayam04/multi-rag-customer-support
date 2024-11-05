import os
import PyPDF2
import faiss
import numpy as np
import pickle
from openai import OpenAI
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from moviepy.editor import VideoFileClip
from fastapi import HTTPException, status
from sentence_transformers import SentenceTransformer
import whisper
import json
from bson.objectid import ObjectId
from datetime import datetime
import re

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
mongo_client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
mongo_db = mongo_client['alphagpt']

VECTOR_DIMENSION = 384
FAISS_FOLDER = "FAISS"
FAISS_INDEX_PATH = os.path.join(FAISS_FOLDER, "faiss_index.index")
DOCUMENTS_PATH = os.path.join(FAISS_FOLDER, "stored_documents.pkl")

os.makedirs(FAISS_FOLDER, exist_ok=True)

if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCUMENTS_PATH):
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(DOCUMENTS_PATH, 'rb') as f:
            stored_documents = pickle.load(f)
    except Exception as e:
        print(f"Error loading FAISS state: {e}")
        index = faiss.IndexFlatL2(VECTOR_DIMENSION)
        stored_documents = []
else:
    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    stored_documents = []

def save_faiss_state():
    try:
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(DOCUMENTS_PATH, 'wb') as f:
            pickle.dump(stored_documents, f)
        print("FAISS state saved successfully!")
    except Exception as e:
        print(f"Error saving FAISS state: {e}")

def extract_text_from_video(video_path: str) -> str:
    try:
        whisper_model = whisper.load_model("base")
        video = VideoFileClip(video_path)
        audio = video.audio
        temp_dir = os.path.join(FAISS_FOLDER, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        audio_path = os.path.normpath(os.path.join(temp_dir, "temp_audio.wav")).replace("\\", "/")
        
        try:
            audio.write_audiofile(audio_path, logger=None)
            result = whisper_model.transcribe(audio_path)
            return result["text"]
        finally:
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    print(f"Warning: Could not delete temporary audio file: {e}")
            video.close()
            
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return ""

def extract_text_from_pdf(pdf_path: str) -> str:
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
    content_data = []
    folder_path = os.path.normpath(folder_path)
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist")
        return content_data
        
    for filename in os.listdir(folder_path):
        file_path = os.path.normpath(os.path.join(folder_path, filename))
        
        try:
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
                print(f"Successfully processed: {filename}")
            else:
                print(f"Warning: No content extracted from {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    return content_data


def initialize_local_vectordb(content_data: List[Dict]):
    global index, stored_documents
    
    if not content_data:
        print("No content data provided for initialization")
        return
        
    try:
        index = faiss.IndexFlatL2(VECTOR_DIMENSION)
        stored_documents = []
        
        vectors = [np.array(item['embedding'], dtype=np.float32) for item in content_data]
        index.add(np.vstack(vectors))
        
        stored_documents.extend(content_data)
        
        save_faiss_state()
        print("Local vector database initialized successfully!")
    except Exception as e:
        print(f"Error initializing local vector database: {str(e)}")

def create_vector_database():
    try:
        os.makedirs(FAISS_FOLDER, exist_ok=True)
        os.makedirs(os.path.join(FAISS_FOLDER, "temp"), exist_ok=True)
        
        video_data = process_folder_content('videos', 'video')
        pdf_data = process_folder_content(os.path.join('documents', 'pdfs'), 'pdf')
        text_data = process_folder_content(os.path.join('documents', 'texts'), 'text')
        
        all_content = video_data + pdf_data + text_data
        
        if all_content:
            initialize_local_vectordb(all_content)
            print(f"Vector database created successfully with {len(all_content)} documents!")
        else:
            print("No content found to process!")
            
    except Exception as e:
        print(f"Error creating vector database: {str(e)}")
        raise

def get_platform_type(query: str) -> str:
    db_keywords = ['find', 'search', 'lookup', 'get', 'query', 'database', 
                  'record', 'entry', 'status', 'order', 'account']
    
    if any(keyword in query.lower() for keyword in db_keywords):
        return "mongodb"
    return "vector"

class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def get_collection_schema(collection_name: str) -> str:
    try:
        collection: Collection = mongo_db[collection_name]
        sample_doc = collection.find_one()
        
        if not sample_doc:
            return "Empty collection"
            
        if '_id' in sample_doc:
            del sample_doc['_id']
            
        fields = list(sample_doc.keys())
        schema_desc = f"Collection '{collection_name}' has the following fields:\n"
        for field in fields:
            value_type = type(sample_doc[field]).__name__
            schema_desc += f"- {field} ({value_type})\n"
            
        total_docs = collection.count_documents({})
        schema_desc += f"\nTotal number of documents in collection: {total_docs}"
            
        return schema_desc
        
    except Exception as e:
        return f"Error getting collection schema: {str(e)}"

def clean_query_text(query_text: str) -> str:
    query_text = query_text.replace("```json", "").replace("```", "")
    query_text = query_text.strip()
    return query_text

def generate_mongodb_query(collection_name: str, user_query: str) -> Dict[str, Any]:
    try:
        schema_context = get_collection_schema(collection_name)
        
        messages = [
            {"role": "system", "content": """You are an assistant that generates MongoDB queries.
                Generate only the query object as valid JSON. Do not include markdown formatting or code blocks.
                For counting queries, use an empty query object {}.
                The query should work with the provided collection schema."""},
            {"role": "user", "content": f"""
                Collection Schema:
                {schema_context}
                
                Generate a MongoDB query object for the following user query:
                {user_query}
                
                Return only the query object without any formatting or explanation.
            """}
        ]

        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=0.1
        )
        
        query_text = clean_query_text(response.choices[0].message.content)
        print(f"Generated query: {query_text}")
        
        try:
            query_dict = json.loads(query_text)
            return query_dict
        except json.JSONDecodeError:
            print(f"Failed to parse generated query: {query_text}")
            return {}
            
    except Exception as e:
        print(f"Error generating MongoDB query: {str(e)}")
        return {}

def search_mongodb_platform(query: str, collection_name: str) -> str:
    try:
        if not collection_name:
            return "No collection name provided for MongoDB search."
            
        collection: Collection = mongo_db[collection_name]
        
        if "how many" in query.lower() or ("total" in query.lower() and "documents" in query.lower()):
            mongo_query = generate_mongodb_query(collection_name, query)
            count = collection.count_documents(mongo_query)
            return json.dumps({"matching_documents": count}, indent=4)
        
        mongo_query = generate_mongodb_query(collection_name, query)
        
        if not mongo_query:
            return "Failed to generate a valid MongoDB query."
        
        content_results = list(collection.find(mongo_query).limit(5))
        
        if content_results:
            return json.dumps(content_results, indent=4, cls=MongoJSONEncoder)
            
        return f"No results found in collection '{collection_name}' for the given query."
            
    except Exception as e:
        print(f"Detailed error: {str(e)}")
        return f"Error searching MongoDB collection '{collection_name}': {str(e)}"

def search_vector_database(query: str) -> str:
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
    
def classify_response(text):
    prompt = f"Classify the following email response into one of the following categories: positive, unsubscribe, inquiry, other. '{text}'"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a Professional Email Categoriser, that specializes in categorizing email responses into positive, unsubscribe, inquiry, other."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        category = response.choices[0].message.content.strip().lower()
        return category
    except Exception as e:
        print(f"Error classifying response: {str(e)}")
        return "other"
    
def bot_reply(category):
    prompt = f"Compose an email response for the category: '{category}'. Return the subject and body of the email in a json value like this: {{\"subject\": \"Subject here\", \"body\": \"Body here\"}}. ALWAYS RETURN A SINGLE LINE JSON AND NOTHING ELSE."
    
    response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a Professional Email Writer, that specializes in writing email based on the user response category. You always return a single line JSON with the subject and body of the email."},
                {"role": "user", "content": prompt}
            ],
            temperature=1
        )

    content = response.choices[0].message.content.strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {content}")
        data = {
                "subject": f"Response to your {category} email",
                "body": f"Thank you for your {category} response. We have received your message and will process it accordingly."
            }

        print(f"Parsed data: {data}")
    return data

def query_agent(query: str) -> Tuple[int, str]:
    try:
        messages = [
            {"role": "system", "content": """You are a query classifier. Analyze the user's query and return ONLY a number (1-5) corresponding to the appropriate action:
                1: For queries seeking information or asking questions
                2: For requests to update or refresh the database
                3: For requests related to sending emails
                4: For requests to start or manage email monitoring
                5: For requests to exit or close
                
                Return ONLY the number, no explanations or additional text."""},
            {"role": "user", "content": query}
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
            max_tokens=1
        )
        
        option = response.choices[0].message.content.strip()
        
        match = re.search(r'\d', option)
        if match:
            option_num = int(match.group())
            if 1 <= option_num <= 5:
                return option_num, query
                
        return 1, query

    except Exception as e:
        print(f"Error in query agent: {str(e)}")
        return 1, query