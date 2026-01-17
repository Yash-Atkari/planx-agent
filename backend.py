import os
import json
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from google_auth_oauthlib.flow import InstalledAppFlow
from langchain_core.messages import SystemMessage
from planX import get_agent_app
from supabase import create_client, Client
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'
# Add this line to force HTTPS redirect detection on Render
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '0'

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

# $ Supabase Connection (Add these to Render Env Variables)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Check if the code is running on Render (Render sets the 'RENDER' environment variable to 'true')
if os.environ.get('RENDER'):
    REDIRECT_URI = 'https://planx-backend-yvin.onrender.com/auth-callback'
    FRONTEND_URI = 'https://planx-agent.netlify.app'
else:
    # Use this for local testing
    REDIRECT_URI = 'http://localhost:8001/auth-callback'
    FRONTEND_URI = 'http://127.0.0.1:5500/index.html'
    
# Authentication config
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 
          'https://www.googleapis.com/auth/gmail.send',
          'https://www.googleapis.com/auth/calendar']
CREDENTIALS_FILE = 'credentials.json'

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    This function runs before your chat starts. 
    It catches the JWT token from the frontend and asks Supabase: 'Is this user valid?'
    """
    token = credentials.credentials
    try:
        # Ask Supabase to verify the token
        user = supabase.auth.get_user(token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid user session")
        return user.user.id # Return the unique Supabase UUID
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed")

@api.get("/login")
async def login(user_id: str):
    """
    Line 1: We take the user_id from the frontend.
    Line 2: We pass that ID into the 'state' parameter so Google remembers it.
    """
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES, redirect_uri=REDIRECT_URI)
    
    # Passing user_id as state
    auth_url, _ = flow.authorization_url(prompt='consent', state=user_id) 
    
    return {"auth_url": auth_url}

@api.get("/auth-callback")
async def auth_callback(code: str = None, state: str = None):
    """
    Line 1: Google sends back the 'code' AND the 'state' (which is our user_id).
    """
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES, redirect_uri=REDIRECT_URI)
    flow.fetch_token(code=code)
    
    token_json = flow.credentials.to_json()
    
    # Line 2: Now 'state' contains the user_id we need for the database!
    supabase.table("users_tokens").upsert({
        "user_id": state, 
        "google_token": json.loads(token_json)
    }).execute()
    
    return RedirectResponse(FRONTEND_URI)

@api.get("/auth-check")
async def auth_check(user_id: str):
    """
    Line 1: We check the DATABASE instead of the local folder.
    Line 2: If a row exists for this user_id, they are connected.
    """
    response = supabase.table("users_tokens").select("*").eq("user_id", user_id).execute()

    if response.data:
        return {"status": "connected"}
    return {"status": "disconnected"}

@api.post("/disconnect")
async def disconnect(user_id: str):
    """Deletes the user's Google token from the database"""
    supabase.table("users_tokens").delete().eq("user_id", user_id).execute()
    return {"status": "success"}

# Chat endpoint
class ChatRequest(BaseModel):
    message: str

@api.post("/chat")
async def chat_endpoint(request: ChatRequest, user_id: str = Depends(get_current_user)):
    try:
        # 1. Fetch the Google token from Supabase for THIS specific user
        db_response = supabase.table("users_tokens").select("google_token").eq("user_id", user_id).single().execute()
        
        if not db_response.data:
            return {"response": "Please connect your Google account first."}
        
        user_google_token = db_response.data["google_token"]

        # 2. Initialize the agent with the user's specific token
        # Note: You will need to modify planX.py to accept 'user_google_token'
        agent_app, sys_prompt = get_agent_app(user_google_token)
        
        # 3. Prepare the message sequence
        input_messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=request.message)
        ]
        
        # 4. State Management: Use user_id as thread_id so each user has their own memory
        config = {"configurable": {"thread_id": user_id}}
        
        # 5. Run the Agent
        output = agent_app.invoke(
            {"messages": input_messages},
            config=config
        )
        
        # 6. Extract the AI's final response
        ai_msg = output["messages"][-1]
        response_text = ai_msg.content
        
        # Handle Gemini's specific content formatting quirks
        final_text = ""
        if isinstance(response_text, str):
            final_text = response_text
        elif isinstance(response_text, list):
            final_text = "".join([part.get("text", part) if isinstance(part, dict) else part for part in response_text])
                    
        return {"response": final_text}

    except Exception as e:
        print(f"Chat Error for user {user_id}: {e}")
        return {"response": f"I encountered an error while processing your request."}

if __name__ == "__main__":
    import uvicorn
    # Render provides a PORT environment variable
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(api, host="0.0.0.0", port=port)
    
# Run with: uvicorn backend:api --reload
