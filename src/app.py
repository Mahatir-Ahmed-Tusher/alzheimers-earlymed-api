from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import logging
from src.inference import run_inference

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Alzheimer's Detection API")

# Add CORS middleware to allow frontend requests
logger.info("Adding CORS middleware for origins: ['http://localhost:8080', 'https://earlymed.vercel.app']")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "https://earlymed.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    logger.info("Handling GET / request")
    return {
        "message": "Welcome to the Alzheimer's Detection API",
        "endpoints": {
            "health": "/health",
            "predict-alzheimers": "/predict-alzheimers (POST)",
            "docs": "/docs"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    logger.info("Handling GET /health request")
    return {"status": "healthy"}

@app.post("/predict-alzheimers")
async def predict_alzheimers(file: UploadFile = File(...)):
    """
    API endpoint to predict Alzheimer's stage from an image.
    """
    logger.info("Received POST /predict-alzheimers request")
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            logger.warning(f"Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="Only JPEG or PNG images are supported.")

        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Run inference
        result = run_inference(image)
        logger.info("Successfully processed /predict-alzheimers request")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in /predict-alzheimers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)