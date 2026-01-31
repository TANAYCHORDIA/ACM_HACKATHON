from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime
from PIL import Image
import io
import numpy as np
from utils.preprocessing import preprocess_image
from utils.clinical import calculate_clinical_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Pneumonia Detection API",
    description="AI-powered pneumonia detection from chest X-rays",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Pneumonia Detection API",
        "status": "online",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": False
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Prediction endpoint with proper image handling"""
   
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Must be an image."
            )
       
        logger.info(f"Received file: {file.filename}, type: {file.content_type}")
       
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
       
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
       
        # Preprocess image
        processed_image = preprocess_image(image, target_size=(224, 224))
       
        logger.info(f"Image preprocessed: shape={processed_image.shape}")
       
        # Dummy prediction values (will replace with real model later)
        prediction = "Normal"
        class_id = 0
        confidence = 0.95
        probabilities = {
            "Normal": 0.95,
            "Bacterial Pneumonia": 0.03,
            "Viral Pneumonia": 0.02
        }
       
        # Calculate clinical metrics
        clinical_metrics = calculate_clinical_metrics(probabilities, prediction, confidence)
       
        logger.info(f"DEBUG - Clinical metrics: {clinical_metrics}")

        # Build response dictionary
        response_data = {
            "success": True,
            "prediction": prediction,
            "class_id": class_id,
            "confidence": confidence,
            "probabilities": probabilities,
            "image_shape": list(processed_image.shape),
            "message": "Dummy prediction - model not loaded yet"
        }
        
        # Merge clinical metrics into response
        response_data.update(clinical_metrics)

        # Return enhanced response
        return JSONResponse(response_data)
   
    except Image.UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not identify image file")
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
