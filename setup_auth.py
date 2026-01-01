import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# 1. Define the scopes for BOTH apps
SCOPES = [
    "https://mail.google.com/",                  # The "Nuclear" Gmail Scope
    "https://www.googleapis.com/auth/calendar"   # Full Calendar Access
]

def get_master_token():
    creds = None
    token_path = "token.json"
    
    # Check if we have an existing token
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        except Exception:
            print("Existing token is invalid. Generating new one...")
            creds = None

    # If no valid credentials, let's log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None

        if not creds:
            print("🚀 Launching Browser for Authorization...")
            print("⚠️  CHECK BOTH BOXES: Gmail AND Calendar!")
            
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
            
            # Save the new Master Token
            with open(token_path, "w") as token:
                token.write(creds.to_json())
                print("✅ Success! 'token.json' created with DOUBLE permissions.")

if __name__ == "__main__":
    # Ensure old token is gone
    if os.path.exists("token.json"):
        os.remove("token.json")
        print("🗑️  Deleted old token.json")
    
    get_master_token()
    