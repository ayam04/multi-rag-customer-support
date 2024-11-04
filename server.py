import uvicorn
from typing import Dict
from fastapi import FastAPI, HTTPException, status
from functions import get_platform_type, search_mongodb_platform, search_vector_database, generate_response, create_vector_database

app = FastAPI()

@app.post("/query")
async def process_query(query: str) -> Dict[str, str]:
    try:
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        platform = get_platform_type(query)
        
        if platform == "mongodb":
            context = search_mongodb_platform(query, "assessment")
        else:
            context = search_vector_database(query)
        
        response = generate_response(context, query)
            
        return {"response": response}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    
@app.post("/update-db")
async def update_database():
    try:
        create_vector_database()
        return {"message": "Database updated successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )    
if __name__ == "__main__":
    uvicorn.run("server:app", port=8080, reload=True)