import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from google_auth_oauthlib.flow import InstalledAppFlow
from langchain_core.messages import SystemMessage
from planX import get_agent_app

# 1. Initialize API
api = FastAPI(title="PlanX API")

# 2. CORS (Frontend & Backend communication)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication config
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 
          'https://www.googleapis.com/auth/gmail.send',
          'https://www.googleapis.com/auth/calendar']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'

# Web auth endpoints
@api.get("/login")
async def login():
    """Generates the Google Auth URL and returns it to the frontend"""
    # Create the flow using the credentials file
    flow = InstalledAppFlow.from_client_secrets_file(
        CREDENTIALS_FILE, SCOPES,
        redirect_uri='http://localhost:8000/auth-callback' # MUST MATCH EXACTLY
    )
    
    # Generate the authorization URL
    auth_url, _ = flow.authorization_url(prompt='consent')
    
    return {"auth_url": auth_url}

@api.get("/auth-callback")
async def auth_callback(code: str = None, error: str = None):
    """Google redirects user here. We grab the code and save the token."""
    if error:
        return {"error": error}
        
    # Recreate the flow to exchange the code for a token
    flow = InstalledAppFlow.from_client_secrets_file(
        CREDENTIALS_FILE, SCOPES,
        redirect_uri='http://localhost:8000/auth-callback'
    )
    
    # Fetch the token using the code Google sent us
    flow.fetch_token(code=code)
    
    # Save the credentials to token.json
    with open(TOKEN_FILE, 'w') as token:
        token.write(flow.credentials.to_json())
        
    # Redirect the user back to your frontend (index.html)
    return RedirectResponse("http://127.0.0.1:5500/index.html") # Use your frontend URL

@api.get("/auth-check")
async def auth_check():
    """Checks if token.json exists"""
    if os.path.exists(TOKEN_FILE):
        return {"status": "connected"}
    return {"status": "disconnected"}

# Chat endpoint
class ChatRequest(BaseModel):
    message: str

@api.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. Unpack the tuple correctly
        agent_app, sys_prompt = get_agent_app() 
        
        # 2. Prepare the messages.
        input_messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=request.message)
        ]
        
        # 3. Use a thread_id to let LangGraph manage state/memory
        config = {"configurable": {"thread_id": "web-user-session"}}
        
        # 4. Invoke the agent
        output = agent_app.invoke(
            {"messages": input_messages},
            config=config
        )
        
        # 5. Extract the LAST message from the graph
        ai_msg = output["messages"][-1]
        response_text = ai_msg.content
        
        # Standardize the output format (handling Gemini's list/dict quirks)
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
        return {"response": f"I encountered an error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    # Render provides a PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(api, host="0.0.0.0", port=port)
    
# Run with: uvicorn backend:api --reload
