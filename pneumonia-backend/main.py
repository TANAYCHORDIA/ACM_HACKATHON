from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from datetime import datetime
from PIL import Image
import io
import numpy as np
from utils.model_loader import get_model

from utils.preprocessing import preprocess_image
from utils.clinical import calculate_clinical_metrics


# --------------------------------------------------
# Configure logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# Create FastAPI app
# --------------------------------------------------
app = FastAPI(
    title="Pneumonia Detection API",
    description="AI-powered pneumonia detection from chest X-rays",
    version="1.0.0"
)


# --------------------------------------------------
# Enable CORS
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "Pneumonia Detection API",
        "status": "online",
        "version": "1.0.0",
        "docs": "/docs"
    }


# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.get("/health")
async def health_check():
    model = get_model()
    model_info = model.get_info()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model.is_loaded(),
        **model_info
    }



# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Must be an image."
            )

        logger.info(f"Received file: {file.filename}, type: {file.content_type}")

        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess image
        # Get model and predict (auto-switches between dummy/real)
        model = get_model()
        result = model.predict(processed_image)
        
        prediction = result["prediction"]
        class_id = result["class_id"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]
        
        # Calculate clinical metrics
        clinical_metrics = calculate_clinical_metrics(
            probabilities=probabilities,
            prediction=prediction,
            confidence=confidence
        )

        # Build response
        response = {
            "success": True,
            "prediction": prediction,
            "class_id": class_id,
            "confidence": confidence,
            "probabilities": probabilities,
            "image_shape": list(processed_image.shape),
            "model_loaded": model.is_loaded(),
            "ensemble_size": result.get("ensemble_size", 0),
            "message": f"Prediction from {result['ensemble_size']}-model ensemble" if model.is_loaded() 
                      else "Dummy prediction - add .h5 files to models/ folder"
        }
        
        response.update(clinical_metrics)
        
        return response


    except Image.UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not identify image file")

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


# --------------------------------------------------
# Run server
# --------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
