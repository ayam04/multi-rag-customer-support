# multi-rag-customer-support

This repository contains a customer support system that utilizes a multi-retrieval approach to answer user queries by leveraging both MongoDB and a vector-based database. The system transcribes text from videos, PDFs, and text documents to create a searchable vector database using FAISS. Additionally, it employs OpenAI’s language models to generate relevant responses based on retrieved context. 

## Project Structure

- **`functions.py`**: This script handles content extraction from various media types, vector database initialization, MongoDB queries, and response generation using OpenAI’s API.
- **`server.py`**: A FastAPI server that defines two endpoints: one for querying the system and another for updating the vector database with new content.

## Features

1. **Content Extraction**:
   - Supports text extraction from video files (using Whisper) and PDFs, alongside direct text files.
   
2. **Vector Database**:
   - FAISS-based vector database stores content embeddings to enable similarity-based retrieval.
   - Embeddings are created with `SentenceTransformer`.
   
3. **MongoDB Querying**:
   - Basic keyword search functionality on MongoDB collections, allowing for structured query handling.
   
4. **Response Generation**:
   - Utilizes OpenAI’s language models to generate responses based on the retrieved context, delivering conversational support to users.

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ayam04/multi-rag-customer-support.git
   cd multi-rag-customer-support
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory with the following:
     ```plaintext
     OPENAI_API_KEY=<your_openai_api_key>
     MONGODB_URI=<your_mongodb_uri>
     ```

4. Run the server:
   ```bash
   uvicorn server:app --port 8080 --reload
   ```

## Usage

### Endpoints

1. **POST /query**  
   This endpoint processes a user query and returns a relevant response based on either MongoDB or vector database results.

   **Request**:
   ```json
   {
     "query": "How do I check my order status?"
   }
   ```
   
   **Response**:
   ```json
   {
     "response": "Your order status can be checked by accessing your account dashboard."
   }
   ```

2. **POST /update-db**  
   Updates the vector database by processing content from the specified folders.

   **Response**:
   ```json
   {
     "message": "Database updated successfully"
   }
   ```

## Key Functions

- **`process_folder_content()`**: Processes video, PDF, or text content from a folder, extracts text, and generates embeddings.
- **`initialize_local_vectordb()`**: Initializes the FAISS index with content embeddings.
- **`search_mongodb_platform()`**: Searches MongoDB collections based on a user query.
- **`search_vector_database()`**: Finds the most relevant content from the vector database using FAISS.
- **`generate_response()`**: Generates a response using OpenAI’s API based on the retrieved context.

## Folder Structure

- **`videos`**: Folder containing video files to be processed.
- **`documents/pdfs`**: Folder containing PDF files.
- **`documents/texts`**: Folder containing text files.
- **`FAISS`**: Stores the FAISS index and processed document embeddings.

## Error Handling

- Comprehensive exception handling in the data processing and querying functions to manage missing files, corrupted data, or connectivity issues.