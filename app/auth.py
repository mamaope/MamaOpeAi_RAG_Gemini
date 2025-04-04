import os
import vertexai
from dotenv import load_dotenv
from google.auth import load_credentials_from_file
import traceback

# Load environment variables
load_dotenv()

PROJECT_ID = os.getenv("GCP_ID")
PROJECT_LOCATION = os.getenv("GCP_LOCATION")
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

def initialize_vertexai():
    if SERVICE_ACCOUNT_FILE and os.path.exists(SERVICE_ACCOUNT_FILE):
        credentials, project_id = load_credentials_from_file(SERVICE_ACCOUNT_FILE)
        vertexai.init(project=PROJECT_ID, location=PROJECT_LOCATION, credentials=credentials)
        print(f"Successfully loaded credentials from file for project: {project_id}")
    else:
        print("No credentials file found. Using application default credentials.")
        vertexai.init(project=PROJECT_ID, location=PROJECT_LOCATION)

initialize_vertexai()
