import uvicorn
from typing import Dict
from typing import List
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from utils import send_email_agent, continuous_monitoring
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from functions import get_platform_type, search_mongodb_platform, search_vector_database, generate_response, create_vector_database

app = FastAPI()

class EmailRequest(BaseModel):
    recipient_email: List[str]

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

@app.post("/send-emails")
def send_email(request: EmailRequest):
    try:
        for email in request.recipient_email:
            send_email_agent(email)
        return JSONResponse(content={"message": "Emails sent successfully"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"message": f"Failed to send emails: {str(e)}"}, status_code=500)

@app.post("/start-monitoring")
def start_monitoring(background_tasks: BackgroundTasks):
    background_tasks.add_task(continuous_monitoring)
    return {"message": "Continuous email monitoring started"}

if __name__ == "__main__":
    uvicorn.run("server:app", port=8080, reload=True)