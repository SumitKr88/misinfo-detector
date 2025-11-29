import os
import boto3
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import modal
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
# In production, these should be environment variables set in Railway
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# App Name defined in modal_app.py
MODAL_APP_NAME = "deepfake-detector-mvp"
# Class Name defined in modal_app.py
MODAL_CLASS_NAME = "DeepfakeDetector"

app = FastAPI(title="Misinformation Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:4000",
        "http://localhost:5173",
        "http://localhost:8000",
    ],
    allow_origin_regex="https?://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

class AnalyzeRequest(BaseModel):
    file_key: str
    file_type: str

@app.get("/")
def health_check():
    return {"status": "healthy"}

@app.get("/generate-upload-url")
def generate_upload_url(file_type: str, extension: str):
    if not AWS_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="Server misconfigured: Missing S3 Bucket")

    file_uuid = str(uuid.uuid4())
    object_name = f"uploads/{file_uuid}.{extension}"

    try:
        presigned_url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': AWS_BUCKET_NAME,
                'Key': object_name,
                'ContentType': file_type
            },
            ExpiresIn=3600
        )
        return {
            "upload_url": presigned_url,
            "file_key": object_name,
            "file_id": file_uuid
        }
    except Exception as e:
        print(f"Error generating URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_media(request: AnalyzeRequest):
    try:
        print(f"Looking up Modal Class: {MODAL_CLASS_NAME} in App: {MODAL_APP_NAME}...")

        # FIX: Use Cls.from_name() which is the correct syntax for Modal 1.0+
        ModelClass = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)

        # Generate a read-url for the worker
        read_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': AWS_BUCKET_NAME, 'Key': request.file_key},
            ExpiresIn=300
        )

        print(f"Triggering inference on: {request.file_key}")

        # Instantiate the class and call the method remotely
        # Note: ModelClass() creates the remote object handle
        result = ModelClass().analyze_media.remote(read_url, request.file_type)

        return result

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"FULL ERROR: {error_details}")
        return {
            "status": "error",
            "message": f"Backend Error: {str(e)}",
            "debug_error": error_details
        }