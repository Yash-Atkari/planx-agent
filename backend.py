from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from planX import app as agent_app # Imports the agent from your planX.py file

# 1. Initialize API
api = FastAPI(title="PlanX API")

# 2. Enable CORS (Allows your frontend to talk to this backend)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Define the data format we expect
class ChatRequest(BaseModel):
    message: str

# 4. The Chat Endpoint
@api.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # A. Prepare Input
        input_message = HumanMessage(content=request.message)
        
        # B. Run Agent
        # We use a fresh thread_id for each request for now
        config = {"configurable": {"thread_id": "web-user"}}
        
        output = agent_app.invoke(
            {"messages": [input_message]},
            config=config
        )
        
        # C. Extract Response
        ai_msg = output["messages"][-1]
        response_text = ai_msg.content

        # Clean up Gemini's list format
        final_text = ""
        if isinstance(response_text, str):
            final_text = response_text
        elif isinstance(response_text, list):
            for part in response_text:
                if isinstance(part, dict) and "text" in part:
                    final_text += part["text"]
                elif isinstance(part, str):
                    final_text += part
                    
        return {"response": final_text}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn backend:api --reload