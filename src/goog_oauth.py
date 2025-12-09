from __future__ import print_function
import pathlib, os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

# Centralized token file locations
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
TOKEN_FILE = os.path.join(PROJECT_ROOT, "goog_token.json")
CREDENTIALS_FILE = os.path.join(PROJECT_ROOT, "credentials.json")

def get_token_path():
    """Get the centralized token file path"""
    return TOKEN_FILE

def get_credentials_path():
    """Get the centralized credentials file path"""
    return CREDENTIALS_FILE

def validate_token():
    """
    Validate the centralized token file.
    Returns True if token exists and is valid, False otherwise.
    """
    try:
        if not pathlib.Path(TOKEN_FILE).exists():
            print(f" Token file not found: {TOKEN_FILE}")
            return False
        
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        if creds and creds.valid:
            print(f" Token is valid: {TOKEN_FILE}")
            return True
        elif creds and creds.expired:
            print(f" Token is expired: {TOKEN_FILE}")
            return False
        else:
            print(f" Token is invalid: {TOKEN_FILE}")
            return False
    except Exception as e:
        print(f" Error validating token: {e}")
        return False

def get_token_status():
    """
    Get detailed status of the centralized token.
    Returns dict with token information.
    """
    status = {
        "token_file": TOKEN_FILE,
        "credentials_file": CREDENTIALS_FILE,
        "token_exists": pathlib.Path(TOKEN_FILE).exists(),
        "credentials_exists": pathlib.Path(CREDENTIALS_FILE).exists(),
        "token_valid": False,
        "token_expired": False,
        "scopes": SCOPES,
        "error": None
    }
    
    try:
        if status["token_exists"]:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            if creds:
                status["token_valid"] = creds.valid
                status["token_expired"] = creds.expired if hasattr(creds, 'expired') else False
                if hasattr(creds, 'expiry') and creds.expiry:
                    status["expiry"] = creds.expiry.isoformat()
    except Exception as e:
        status["error"] = str(e)
    
    return status

def get_service():
    """
    Get Google Calendar service using centralized token management.
    
    Token file: goog_token.json (in project root)
    Credentials file: credentials.json (in project root)
    """
    creds = None
    
    # Load existing token from centralized location
    if pathlib.Path(TOKEN_FILE).exists():
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            print(f" Loaded existing token from {TOKEN_FILE}")
        except Exception as e:
            print(f" Failed to load existing token: {e}")
            creds = None
    
    # Check if credentials need refresh or creation
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                print(" Refreshing expired token...")
                creds.refresh(Request())
                print(" Token refreshed successfully")
            except Exception as e:
                print(f" Failed to refresh token: {e}")
                creds = None
        
        # Create new credentials if needed
        if not creds:
            if not pathlib.Path(CREDENTIALS_FILE).exists():
                raise FileNotFoundError(
                    f" Credentials file not found: {CREDENTIALS_FILE}\n"
                    "Please download OAuth client JSON from Google Cloud Console and save as 'credentials.json'"
                )
            
            print("🔐 Starting OAuth flow...")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
            print(" OAuth flow completed")
        
        # Save token to centralized location
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
            
            with open(TOKEN_FILE, "w") as f:
                f.write(creds.to_json())
            print(f" Token saved to {TOKEN_FILE}")
        except Exception as e:
            print(f" Failed to save token: {e}")
    
    return build("calendar", "v3", credentials=creds)
