# Veritas AI - Misinformation Detector

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)
![Modal](https://img.shields.io/badge/Modal%20Cloud%20AI%20-black?logo=icloud)
![PyTorch](https://img.shields.io/badge/pytorch-grey?logo=pytorch)

Veritas AI is a powerful tool designed to detect AI-generated images and deepfakes. It combines state-of-the-art AI models with digital forensics (Metadata analysis, Frequency domain analysis) to provide a comprehensive credibility score.

## Features

-   **Hybrid Analysis**: Uses an ensemble of AI models (HuggingFace Transformers) and traditional digital forensics.
-   **Modal Cloud Integration**: Offloads heavy AI inference to Modal's serverless GPU infrastructure.
-   **Real-time Analysis**: Fast and efficient detection pipeline.
-   **Detailed Reports**: Provides granular details on why an image was flagged (e.g., "Synthetic frequency patterns", "Missing metadata").

## Architecture Abstract

The **Veritas AI** architecture is designed for scalability and efficiency by decoupling the lightweight backend from the heavy AI processing.

*   **Frontend (User)**: Uploads files directly to S3 (via presigned URLs) to avoid bottling the backend.
*   **FastAPI Backend**: Acts as a control plane. It generates secure upload credentials and triggers remote jobs on Modal.
*   **AWS S3**: Serves as the central storage for media files, accessible by both the User and the GPU workers.
*   **Modal.com**: Provides serverless GPU infrastructure. It spins up containers on demand to run the **DeepfakeDetectorLogic** (AI Ensemble + Forensic Analysis) and shuts them down immediately after processing.

## Prerequisites

-   **Python 3.9+**
-   **Node.js** (Optional, for advanced frontend serving)
-   **Modal Account**: Sign up at [modal.com](https://modal.com)
-   **AWS S3 Bucket**: For storing uploaded media.

## Installation

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd misinfo-detector
    ```

2.  **Backend Setup**
    ```bash
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pip install python-dotenv
    ```

3.  **Modal Setup**
    To run the detection models on the cloud, you need to set up Modal:

    **Authentication**:
    1.  Create a new token:
        ```bash
        modal token new
        ```
        This will open your browser to authenticate.
    2.  Configure your local client:
        ```bash
        modal deploy backend/modal_app.py
        ```
        This will deploy the app to Modal (now you can run the modal execution from cloud)


## Configuration

Create a `.env` file in the project root (`/misinfo-detector/.env`) with your AWS credentials:

```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_BUCKET_NAME=your_bucket_name
AWS_REGION=us-east-1
```

## Running the Application

### 1. Start the Backend Server
**Important**: Run this command from the **project root** directory, not inside the `backend` folder.

```bash
# From project root
backend/venv/bin/python -m uvicorn backend.main:app --reload --port 8000
```

### 2. Run the Frontend
Since the frontend is a simple HTML/JS application, you can open `frontend/index.html` directly in your browser, or serve it using a simple HTTP server:

```bash
# From project root
python -m http.server 4000
```
Then visit `http://localhost:4000`.

### 3. Load frontend at http://localhost:4000 , Upload the image/video and Analyse
Media file will be uploaded to S3, and Modal will read the uploaded assest in S3 and apply the algorithm and share the result in UI.
<img width="487" height="358" alt="Screenshot 2025-11-29 at 11 10 20 PM" src="https://github.com/user-attachments/assets/a8f8aef6-ed90-4399-8a17-0278ec37b4c2" />


### 4. Final Result
After evaluation, final result based on modal execution should reflect in UI.
Sample screenshot:
<img width="933" height="685" alt="Screenshot 2025-11-24 at 6 12 08 PM" src="https://github.com/user-attachments/assets/a26c8cb3-d6bc-462a-848e-07469882da2b" />
<img width="458" height="726" alt="Screenshot 2025-11-29 at 7 43 05 PM" src="https://github.com/user-attachments/assets/60dbf7eb-0921-4372-8404-738756801463" />


## Running Local Evaluation (Optional)

To test the accuracy of the detection models locally (without deploying to Modal), we have provided an evaluation script.

1.  **Install Local Dependencies**
    Running the models locally requires heavy libraries (PyTorch, Transformers) which are not needed for the lightweight API server.
    ```bash
    backend/venv/bin/pip install -r backend/requirements-local.txt
    ```

2.  **Prepare Dataset**
    Place your test images in the following directories:
    -   `backend/dataset/real/`: Real images.
    -   `backend/dataset/fake/`: AI-generated images.

3.  **Run Evaluation**
    ```bash
    backend/venv/bin/python backend/evaluate.py
    ```
    This will generate a report showing Accuracy, Precision, Recall, and a list of False Positives/Negatives.

    Evaluation report Sample.
<img width="711" height="844" alt="Screenshot 2025-11-29 at 11 02 39 PM" src="https://github.com/user-attachments/assets/83184e1d-cb93-4e39-a455-eb3bbfe36eb8" />

## Project Structure

```
misinfo-detector/
├── backend/
│   ├── main.py                 # FastAPI Backend Server
│   ├── modal_app.py            # Modal Cloud Application
│   ├── detector_logic.py       # Core Detection Logic (Shared)
│   ├── evaluate.py             # Local Evaluation Script
│   ├── requirements.txt        # Server Dependencies
│   ├── requirements-local.txt  # Local ML Dependencies
│   └── dataset/                # Test Data
├── frontend/
│   └── index.html              # Frontend UI
└── .env                        # Environment Variables
```

## Troubleshooting

-   **ModuleNotFoundError: No module named 'backend'**: Make sure you run the uvicorn command from the project root.
-   **CORS Errors**: Ensure the backend is running and the `.env` file is correctly loaded.
-   **Modal Errors**: Ensure you have authenticated with `modal setup` and that `detector_logic.py` is correctly mounted (handled automatically in the code).

## Support

If you find this project useful, please give it a **Star ⭐️** and **Share** it with others!
