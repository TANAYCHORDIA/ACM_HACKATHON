# ⚡ 24-HOUR HACKATHON - FINAL BATTLE PLAN
## AI-Assisted Pneumonia Detection System

**Start Time:** 1:30 PM IST  
**End Time:** 1:30 PM IST (Next Day)  
**Strategy:** Adaptive checkpoints - scale up if ahead, scale back if behind

---

## 🚨 CRITICAL SUCCESS METRICS

### HOUR 2 CHECKPOINT (3:30 PM)
- [ ] ResNet50 Phase 1 training running (Tanay)
- [ ] FastAPI `/health` endpoint responding (Akshayaa)
- [ ] Streamlit file upload working (Dhanvi)
- [ ] GitHub repo created, demo images collected (Sajal)

**IF ANY FAIL:** Emergency team huddle - reassign tasks

### HOUR 8 CHECKPOINT (9:30 PM)
- [ ] ResNet50 trained with >88% validation accuracy (Tanay)
- [ ] API `/predict` endpoint works with model (Akshayaa)
- [ ] Frontend calls API, displays predictions (Dhanvi)
- [ ] End-to-end test successful (Everyone)

**IF FAIL:** Cut to Plan B (see bottom)

### HOUR 16 CHECKPOINT (5:30 AM)
- [ ] Grad-CAM heatmaps working (Tanay)
- [ ] Backend deployed OR local with ngrok (Akshayaa)
- [ ] UI polished, heatmaps displayed (Dhanvi)
- [ ] Demo rehearsed once (Everyone)

**IF FAIL:** Focus on demo prep with what you have

---

## 👤 TANAY - ML ENGINEER

### 🔥 HOUR 0-2 (1:30-3:30 PM) - CRITICAL PATH
**Goal:** Training running, no excuses

**MUST DO (IN ORDER):**
1. Open Google Colab, enable T4 GPU
2. Install dependencies:
   ```python
   !pip install tensorflow==2.15 kaggle opencv-python-headless pandas scikit-learn
   ```
3. Download dataset to Colab:
   ```python
   !kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   !unzip chest-xray-pneumonia.zip
   ```
4. Quick data split (stratified 70/15/15):
   ```python
   from sklearn.model_selection import train_test_split
   # Split code - keep it simple, save CSVs
   ```
5. **START RESNET50 PHASE 1 TRAINING** (freeze base, train head):
   ```python
   from tensorflow.keras.applications import ResNet50
   from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
   from tensorflow.keras.models import Model
   from tensorflow.keras.optimizers import Adam
   
   base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
   base_model.trainable = False
   
   x = base_model.output
   x = GlobalAveragePooling2D()(x)
   x = Dense(512, activation='relu')(x)
   x = Dropout(0.5)(x)
   x = Dense(256, activation='relu')(x)
   predictions = Dense(3, activation='softmax')(x)
   
   model = Model(inputs=base_model.input, outputs=predictions)
   model.compile(optimizer=Adam(0.001), 
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
   
   # Train for 10 epochs - should take 20-30 min
   history = model.fit(train_data, epochs=10, validation_data=val_data)
   model.save('resnet50_phase1.h5')
   ```

**CHECKPOINT (3:30 PM):**
- Training running? ✅
- Phase 1 accuracy >70%? ✅
- If NO: Debug immediately, team helps

---

### ⚡ HOUR 2-8 (3:30-9:30 PM) - CORE TRAINING
**Goal:** One working model with >88% accuracy

**MUST DO:**
1. **Phase 2 Training** (unfreeze last 50 layers):
   ```python
   base_model.trainable = True
   for layer in base_model.layers[:-50]:
       layer.trainable = False
   
   model.compile(optimizer=Adam(1e-5), 
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
   
   # Train for 40 epochs with early stopping
   from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
   
   callbacks = [
       EarlyStopping(patience=5, restore_best_weights=True),
       ModelCheckpoint('resnet50_best.h5', save_best_only=True)
   ]
   
   history = model.fit(train_data, epochs=40, 
                       validation_data=val_data,
                       callbacks=callbacks)
   ```

2. **Evaluate on validation set:**
   ```python
   from sklearn.metrics import classification_report, confusion_matrix
   
   val_preds = model.predict(val_data)
   print(classification_report(y_val, val_preds.argmax(axis=1)))
   ```

3. **Save model for backend:**
   ```python
   model.save('resnet50_final.h5')
   # Download to local, push to GitHub
   ```

**IF AHEAD OF SCHEDULE:**
- Start VGG16 training in parallel (new Colab notebook)
- Add basic data augmentation (rotation ±10°, flip)

**CHECKPOINT (9:30 PM):**
- Model saved? ✅
- Validation accuracy >88%? ✅
- Confusion matrix looks reasonable? ✅

---

### 🎯 HOUR 8-16 (9:30 PM-5:30 AM) - GRAD-CAM + OPTIMIZATION
**Goal:** Explainability working, second model if time

**MUST DO:**
1. **Implement Grad-CAM** (2-3 hours):
   ```python
   !pip install tf-keras-vis
   
   from tf_keras_vis.gradcam import Gradcam
   from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
   
   def generate_gradcam(model, image, class_idx):
       gradcam = Gradcam(model, 
                         model_modifier=ReplaceToLinear(),
                         clone=True)
       
       cam = gradcam(lambda output: output[:, class_idx],
                     image,
                     penultimate_layer=-1)
       
       return cam[0]
   
   # Test on 5 sample images
   # Overlay heatmap on original image
   import cv2
   heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
   overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
   ```

2. **Package as utility function:**
   ```python
   # Create src/gradcam.py
   def get_gradcam_heatmap(model_path, image_path, class_idx):
       # Load model, generate heatmap, return base64 encoded
       pass
   ```

3. **Share with backend team** (Akshayaa needs this)

**IF AHEAD:**
- Train VGG16 or MobileNetV2
- Implement simple ensemble (average probabilities)
- Add CLAHE preprocessing

**IF BEHIND:**
- Skip second model
- Get basic Grad-CAM working (even if not perfect)

**CHECKPOINT (5:30 AM):**
- Grad-CAM generates heatmaps? ✅
- Overlays look medically plausible? ✅

---

### 🏁 HOUR 16-24 (5:30 AM-1:30 PM) - POLISH + DEMO
**Goal:** Model performance documented, demo ready

**MUST DO:**
1. **Generate evaluation metrics:**
   ```python
   # Test set evaluation
   test_acc = model.evaluate(test_data)
   test_preds = model.predict(test_data)
   
   # Calculate precision, recall, F1 for each class
   # Create confusion matrix visualization
   # Save as images for slides
   ```

2. **Prepare demo images:**
   - 2 Normal cases (with low confidence scores)
   - 2 Bacterial cases (with heatmaps showing consolidation)
   - 2 Viral cases (with heatmaps showing diffuse patterns)
   - Test each image end-to-end

3. **Document real results** (update README with actual numbers)

4. **Demo rehearsal:**
   - Explain model architecture (2 min)
   - Show training curves (1 min)
   - Explain Grad-CAM (2 min)
   - Answer Q&A about limitations

**DELIVERABLES:**
- `resnet50_final.h5` (and VGG16 if trained)
- `src/gradcam.py`
- Evaluation metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization
- 6 tested demo images

---

## 👤 AKSHAYAA - BACKEND ENGINEER

### 🔥 HOUR 0-3 (1:30-4:30 PM) - API SKELETON
**Goal:** FastAPI responding to requests

**MUST DO (IN ORDER):**
1. **Create project structure:**
   ```bash
   mkdir backend && cd backend
   touch main.py requirements.txt
   ```

2. **Install dependencies:**
   ```bash
   pip install fastapi==0.104.1 uvicorn python-multipart pillow numpy tensorflow==2.15
   ```

3. **Create basic FastAPI app:**
   ```python
   # backend/main.py
   from fastapi import FastAPI, File, UploadFile, HTTPException
   from fastapi.middleware.cors import CORSMiddleware
   import uvicorn
   
   app = FastAPI(title="Pneumonia Detection API")
   
   # Enable CORS for frontend
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   
   @app.get("/health")
   async def health_check():
       return {
           "status": "healthy",
           "message": "API is running"
       }
   
   @app.post("/predict")
   async def predict(file: UploadFile = File(...)):
       # Dummy response for now
       return {
           "prediction": "Normal",
           "class_id": 0,
           "confidence": 0.95,
           "probabilities": {
               "Normal": 0.95,
               "Bacterial": 0.03,
               "Viral": 0.02
           }
       }
   
   if __name__ == "__main__":
       uvicorn.run(app, host="0.0.0.0", port=8000)
   ```

4. **Test locally:**
   ```bash
   uvicorn main:app --reload --port 8000
   # Visit http://localhost:8000/docs
   # Test /health endpoint
   ```

5. **Share API URL with Dhanvi** (http://localhost:8000)

**CHECKPOINT (4:30 PM):**
- API running? ✅
- `/health` returns 200? ✅
- Swagger docs accessible? ✅

---

### ⚡ HOUR 3-10 (4:30-11:30 PM) - REAL INFERENCE
**Goal:** Load model, make real predictions

**MUST DO:**
1. **Wait for Tanay's Phase 1 model** (~6:30 PM)

2. **Create preprocessing function:**
   ```python
   # backend/utils.py
   import numpy as np
   from PIL import Image
   import io
   
   def preprocess_image(image_bytes):
       """Convert uploaded image to model input format"""
       image = Image.open(io.BytesIO(image_bytes))
       image = image.convert('RGB')
       image = image.resize((224, 224))
       image_array = np.array(image) / 255.0
       image_array = np.expand_dims(image_array, axis=0)
       return image_array
   ```

3. **Load model and update `/predict`:**
   ```python
   # backend/main.py
   from tensorflow import keras
   import numpy as np
   from utils import preprocess_image
   
   # Load model at startup
   MODEL = keras.models.load_model('models/resnet50_phase1.h5')
   CLASS_NAMES = ['Normal', 'Bacterial Pneumonia', 'Viral Pneumonia']
   
   @app.post("/predict")
   async def predict(file: UploadFile = File(...)):
       try:
           # Read image
           contents = await file.read()
           image = preprocess_image(contents)
           
           # Make prediction
           predictions = MODEL.predict(image)
           class_id = int(np.argmax(predictions[0]))
           confidence = float(np.max(predictions[0]))
           
           return {
               "prediction": CLASS_NAMES[class_id],
               "class_id": class_id,
               "confidence": confidence,
               "probabilities": {
                   CLASS_NAMES[i]: float(predictions[0][i])
                   for i in range(3)
               }
           }
       except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))
   ```

4. **Test with real X-ray images:**
   ```bash
   curl -X POST http://localhost:8000/predict \
     -F "file=@test_xray.jpg"
   ```

**CHECKPOINT (11:30 PM):**
- Model loaded successfully? ✅
- Predictions look reasonable? ✅
- Response time <2s? ✅

---

### 🎯 HOUR 10-18 (11:30 PM-7:30 AM) - INTEGRATION + DEPLOYMENT
**Goal:** Grad-CAM integrated, API deployed

**MUST DO:**
1. **Integrate Grad-CAM** (wait for Tanay's code ~2 AM):
   ```python
   # Copy Tanay's gradcam.py to backend/
   from gradcam import get_gradcam_heatmap
   import base64
   
   @app.post("/predict")
   async def predict(file: UploadFile = File(...)):
       # ... existing prediction code ...
       
       # Generate heatmap
       heatmap = get_gradcam_heatmap(MODEL, image, class_id)
       
       # Convert to base64
       import cv2
       _, buffer = cv2.imencode('.png', heatmap)
       heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
       
       return {
           "prediction": CLASS_NAMES[class_id],
           "confidence": confidence,
           "probabilities": {...},
           "heatmap_base64": heatmap_base64
       }
   ```

2. **Deployment to Render:**
   ```yaml
   # render.yaml
   services:
     - type: web
       name: pneumonia-api
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

3. **Update requirements.txt:**
   ```
   fastapi==0.104.1
   uvicorn[standard]
   python-multipart
   pillow
   numpy
   tensorflow==2.15
   opencv-python-headless
   tf-keras-vis
   ```

4. **Deploy and test:**
   - Push to GitHub
   - Connect to Render
   - Test deployed endpoint
   - Share live URL with team

**IF DEPLOYMENT FAILS:**
- Use ngrok for local tunnel:
   ```bash
   pip install pyngrok
   ngrok http 8000
   # Share ngrok URL
   ```

**CHECKPOINT (7:30 AM):**
- Grad-CAM returns in API response? ✅
- Deployed OR ngrok URL working? ✅

---

### 🏁 HOUR 18-24 (7:30 AM-1:30 PM) - POLISH + DOCS
**Goal:** API stable, documented, tested

**MUST DO:**
1. **Error handling improvements:**
   ```python
   # Validate file type
   if not file.content_type.startswith('image/'):
       raise HTTPException(400, "File must be an image")
   
   # Handle model errors gracefully
   try:
       predictions = MODEL.predict(image)
   except Exception as e:
       logger.error(f"Prediction failed: {e}")
       raise HTTPException(500, "Model inference failed")
   ```

2. **API documentation** (auto-generated by FastAPI):
   - Add descriptions to endpoints
   - Add example responses
   - Test Swagger UI at `/docs`

3. **Performance testing:**
   ```python
   import time
   
   @app.post("/predict")
   async def predict(file: UploadFile = File(...)):
       start_time = time.time()
       # ... prediction code ...
       inference_time = time.time() - start_time
       
       return {
           ...,
           "inference_time_ms": round(inference_time * 1000, 2)
       }
   ```

4. **Load testing:**
   ```bash
   # Test with 10 concurrent requests
   # Ensure <2s response time
   ```

**DELIVERABLES:**
- Working API (deployed or ngrok)
- `/predict`, `/health` endpoints
- Swagger docs at `/docs`
- Error handling for edge cases

---

## 👤 DHANVI - FRONTEND LEAD

### 🔥 HOUR 0-4 (1:30-5:30 PM) - UI SKELETON
**Goal:** File upload working, connected to API

**MUST DO (IN ORDER):**
1. **Create Streamlit app:**
   ```bash
   mkdir frontend && cd frontend
   touch app.py requirements.txt
   ```

2. **Install dependencies:**
   ```bash
   pip install streamlit==1.29 requests pillow
   ```

3. **Basic UI with file upload:**
   ```python
   # frontend/app.py
   import streamlit as st
   from PIL import Image
   import requests
   
   st.set_page_config(page_title="Pneumonia Detection", layout="wide")
   
   st.title("🫁 AI-Assisted Pneumonia Detection")
   st.markdown("Upload a chest X-ray for instant analysis")
   
   # File uploader
   uploaded_file = st.file_uploader(
       "Choose an X-ray image", 
       type=['jpg', 'jpeg', 'png']
   )
   
   if uploaded_file is not None:
       # Display uploaded image
       image = Image.open(uploaded_file)
       st.image(image, caption="Uploaded X-Ray", width=400)
       
       # Predict button
       if st.button("🔍 Analyze X-Ray"):
           st.write("Connecting to API...")
           # Will connect to backend later
   ```

4. **Test locally:**
   ```bash
   streamlit run app.py
   # Visit http://localhost:8501
   ```

**CHECKPOINT (5:30 PM):**
- Streamlit app loads? ✅
- File upload widget works? ✅
- Image displays correctly? ✅

---

### ⚡ HOUR 4-12 (5:30 PM-1:30 AM) - API INTEGRATION
**Goal:** Show real predictions from backend

**MUST DO:**
1. **Connect to Akshayaa's API:**
   ```python
   # frontend/app.py
   API_URL = "http://localhost:8000"  # Update with deployed URL later
   
   if st.button("🔍 Analyze X-Ray"):
       with st.spinner("Analyzing X-ray..."):
           # Send to backend
           files = {"file": uploaded_file.getvalue()}
           response = requests.post(f"{API_URL}/predict", files=files)
           
           if response.status_code == 200:
               result = response.json()
               
               # Display results
               st.success("✅ Analysis Complete!")
               
               col1, col2 = st.columns(2)
               
               with col1:
                   st.metric("Diagnosis", result['prediction'])
                   st.metric("Confidence", f"{result['confidence']*100:.1f}%")
               
               with col2:
                   st.write("**Class Probabilities:**")
                   for class_name, prob in result['probabilities'].items():
                       st.progress(prob, text=f"{class_name}: {prob*100:.1f}%")
           else:
               st.error("❌ Error analyzing image")
   ```

2. **Test end-to-end:**
   - Upload test image
   - Click "Analyze"
   - Verify prediction displays

**CHECKPOINT (1:30 AM):**
- Frontend calls backend successfully? ✅
- Predictions display correctly? ✅
- Error messages show for failures? ✅

---

### 🎯 HOUR 12-20 (1:30-9:30 AM) - HEATMAPS + POLISH
**Goal:** Beautiful UI with Grad-CAM visualization

**MUST DO:**
1. **Display Grad-CAM heatmap:**
   ```python
   import base64
   from io import BytesIO
   
   if response.status_code == 200:
       result = response.json()
       
       # Create 2 columns
       col1, col2 = st.columns(2)
       
       with col1:
           st.subheader("Original X-Ray")
           st.image(image, use_column_width=True)
       
       with col2:
           st.subheader("AI Attention Heatmap")
           # Decode base64 heatmap
           heatmap_bytes = base64.b64decode(result['heatmap_base64'])
           heatmap_image = Image.open(BytesIO(heatmap_bytes))
           st.image(heatmap_image, use_column_width=True)
       
       # Results below
       st.markdown("---")
       st.subheader("📊 Diagnosis Results")
       
       # Styled metrics
       if result['prediction'] == 'Normal':
           st.success(f"✅ {result['prediction']}")
       else:
           st.warning(f"⚠️ {result['prediction']}")
       
       st.metric("Confidence Score", f"{result['confidence']*100:.1f}%")
   ```

2. **UI improvements:**
   ```python
   # Add sidebar
   with st.sidebar:
       st.header("ℹ️ About")
       st.write("""
       This AI system detects pneumonia from chest X-rays.
       
       **Classes:**
       - Normal
       - Bacterial Pneumonia
       - Viral Pneumonia
       
       **Model:** ResNet50 Transfer Learning
       **Accuracy:** 90%+
       """)
       
       st.markdown("---")
       st.write("Built for Nokia Health Tech Hackathon 2026")
   ```

3. **Color coding based on prediction:**
   ```python
   # Custom CSS
   if result['prediction'] != 'Normal':
       st.markdown("""
       <div style='padding:10px; background-color:#fff3cd; border-radius:5px;'>
       ⚠️ <b>Attention Required:</b> Pneumonia detected. Please consult a radiologist.
       </div>
       """, unsafe_allow_html=True)
   ```

**IF AHEAD:**
- Add model selector dropdown (if multiple models available)
- Add download results button
- Add batch upload feature

**CHECKPOINT (9:30 AM):**
- Heatmap displays correctly? ✅
- UI looks professional? ✅
- Color coding works? ✅

---

### 🏁 HOUR 20-24 (9:30 AM-1:30 PM) - DEPLOYMENT + FINAL TESTING
**Goal:** Deployed app, demo-ready

**MUST DO:**
1. **Deploy to Streamlit Cloud:**
   ```bash
   # Create requirements.txt
   streamlit==1.29.0
   requests
   pillow
   ```

2. **Update API URL to deployed backend:**
   ```python
   API_URL = "https://pneumonia-api.onrender.com"  # or ngrok URL
   ```

3. **Test deployed app with all demo images:**
   - 2 Normal cases
   - 2 Bacterial cases
   - 2 Viral cases

4. **Screenshot for slides:**
   - Main interface
   - Prediction results
   - Heatmap visualization

**DELIVERABLES:**
- Deployed Streamlit app (URL)
- Working end-to-end demo
- Screenshots for presentation

---

## 👤 SAJAL - FRONTEND SUPPORT & TESTING

### 🔥 HOUR 0-4 (1:30-5:30 PM) - SETUP & RESOURCES
**Goal:** Infrastructure ready, demo materials prepared

**MUST DO (IN ORDER):**
1. **Create GitHub repository:**
   ```bash
   # Create repo: pneumonia-ai-detector
   git init
   mkdir -p data/raw models src backend frontend docs tests
   touch README.md .gitignore
   
   # .gitignore
   *.h5
   *.pyc
   __pycache__/
   .env
   data/raw/
   ```

2. **Find demo X-ray images:**
   - Google: "chest x-ray pneumonia examples"
   - Collect 10 images:
     - 3-4 Normal
     - 3-4 Bacterial pneumonia
     - 3 Viral pneumonia
   - Save to `demo_images/` folder with clear names

3. **Create project structure:**
   ```bash
   pneumonia-ai-detector/
   ├── data/
   │   ├── raw/
   │   └── processed/
   ├── models/
   ├── src/
   │   ├── data_prep.py
   │   ├── train.py
   │   └── gradcam.py
   ├── backend/
   │   ├── main.py
   │   └── requirements.txt
   ├── frontend/
   │   ├── app.py
   │   └── requirements.txt
   ├── demo_images/
   ├── docs/
   └── README.md
   ```

4. **Initial commit:**
   ```bash
   git add .
   git commit -m "Initial project structure"
   git push origin main
   ```

**CHECKPOINT (5:30 PM):**
- Repo created and shared with team? ✅
- 10 demo images collected? ✅
- Folder structure ready? ✅

---

### ⚡ HOUR 4-12 (5:30 PM-1:30 AM) - API CLIENT & UTILITIES
**Goal:** Helper functions for frontend

**MUST DO:**
1. **Create API client utility:**
   ```python
   # frontend/utils/api_client.py
   import requests
   from typing import Dict, Any
   
   class PneumoniaAPI:
       def __init__(self, base_url: str):
           self.base_url = base_url
       
       def health_check(self) -> Dict[str, Any]:
           """Check if API is healthy"""
           response = requests.get(f"{self.base_url}/health")
           return response.json()
       
       def predict(self, image_file) -> Dict[str, Any]:
           """Send image for prediction"""
           files = {"file": image_file}
           response = requests.post(
               f"{self.base_url}/predict",
               files=files,
               timeout=30
           )
           response.raise_for_status()
           return response.json()
   ```

2. **Help Dhanvi integrate this:**
   ```python
   # In app.py
   from utils.api_client import PneumoniaAPI
   
   api = PneumoniaAPI("http://localhost:8000")
   
   # Check API health
   if api.health_check()['status'] == 'healthy':
       st.success("✅ Connected to AI backend")
   ```

3. **Create image preprocessing utilities:**
   ```python
   # frontend/utils/image_utils.py
   from PIL import Image
   import io
   
   def validate_image(uploaded_file):
       """Validate uploaded image"""
       try:
           image = Image.open(uploaded_file)
           # Check dimensions
           if image.width < 100 or image.height < 100:
               return False, "Image too small"
           return True, "Valid image"
       except:
           return False, "Invalid image file"
   
   def resize_for_display(image, max_width=400):
       """Resize image for display"""
       ratio = max_width / image.width
       new_height = int(image.height * ratio)
       return image.resize((max_width, new_height))
   ```

**IF AHEAD:**
- Create batch processing helper
- Add image format conversion utilities

**CHECKPOINT (1:30 AM):**
- API client working? ✅
- Utilities integrated in frontend? ✅

---

### 🎯 HOUR 12-20 (1:30-9:30 AM) - TESTING & BONUS FEATURES
**Goal:** Comprehensive testing, add nice-to-haves

**MUST DO:**
1. **Test all demo images:**
   ```python
   # Create test_demo.py
   import os
   from utils.api_client import PneumoniaAPI
   
   api = PneumoniaAPI("http://localhost:8000")
   
   demo_dir = "demo_images/"
   for filename in os.listdir(demo_dir):
       print(f"\nTesting: {filename}")
       with open(os.path.join(demo_dir, filename), 'rb') as f:
           result = api.predict(f)
           print(f"Prediction: {result['prediction']}")
           print(f"Confidence: {result['confidence']}")
   ```

2. **Document edge cases:**
   - Non-image files → Should show error
   - Very small images → Should handle gracefully
   - Network timeout → Should show retry option

3. **Create error log:**
   ```python
   # Track any errors during testing
   errors = []
   # Document in errors.md
   ```

**IF AHEAD - Add batch processing UI:**
```python
# frontend/app.py - add tab
tab1, tab2 = st.tabs(["Single Image", "Batch Upload"])

with tab2:
    uploaded_files = st.file_uploader(
        "Upload multiple X-rays",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )
    
    if st.button("Analyze Batch"):
        results = []
        for file in uploaded_files:
            result = api.predict(file)
            results.append({
                'filename': file.name,
                'prediction': result['prediction'],
                'confidence': result['confidence']
            })
        
        # Display results table
        st.dataframe(results)
```

**CHECKPOINT (9:30 AM):**
- All demo images tested? ✅
- Edge cases documented? ✅
- Batch feature added (if time)? ✅/❌

---

### 🏁 HOUR 20-24 (9:30 AM-1:30 PM) - DEMO PREP & PRESENTATION
**Goal:** Perfect demo, slides ready

**MUST DO:**
1. **Record demo video (backup plan):**
   ```bash
   # Use OBS or screen recorder
   # Show:
   - Opening app
   - Uploading each demo image
   - Predictions + heatmaps
   - Different classes
   # Length: 2-3 minutes
   ```

2. **Create presentation slides:**
   - Slide 1: Problem statement
   - Slide 2: Our solution (system architecture)
   - Slide 3: Model performance (real numbers from Tanay)
   - Slide 4: Live demo screenshots
   - Slide 5: Technical implementation
   - Slide 6: Future improvements

3. **Prepare demo script:**
   ```
   [0:00-0:30] Introduction
   "We built an AI system to help doctors triage pneumonia cases faster."
   
   [0:30-1:00] System overview
   "ResNet50 model, FastAPI backend, Streamlit frontend."
   
   [1:00-3:00] Live demo
   - Upload Normal X-ray → "See, classified as Normal with 92% confidence"
   - Upload Bacterial → "Detected Bacterial Pneumonia, heatmap shows consolidation"
   - Upload Viral → "Viral Pneumonia, diffuse pattern visible in heatmap"
   
   [3:00-4:00] Results
   "Achieved 90% accuracy, 92% recall - critical for not missing sick patients"
   
   [4:00-5:00] Q&A
   ```

4. **Rehearse with team:**
   - Everyone runs through demo once
   - Time each section
   - Prepare answers for common questions:
     - "Why ResNet50?" → Proven architecture, good balance speed/accuracy
     - "How accurate?" → [Real numbers from testing]
     - "Can this be used in hospitals?" → Yes, API can integrate with PACS
     - "What about false negatives?" → High recall minimizes missed cases

**DELIVERABLES:**
- Demo video (2-3 min)
- Presentation slides (5-6 slides)
- Demo script
- Team rehearsed at least once

---

## 🚨 EMERGENCY PLAN B (If Behind at Hour 8)

### MINIMAL VIABLE DEMO
If by 9:30 PM you don't have all pieces working:

**Cut these immediately:**
- ❌ Second model (VGG16/MobileNet)
- ❌ Ensemble
- ❌ Deployment (run locally)
- ❌ Batch processing
- ❌ CLAHE preprocessing

**Focus ONLY on:**
- ✅ ONE ResNet50 model (even if 85% accuracy)
- ✅ Basic `/predict` endpoint (local backend)
- ✅ Basic UI (upload + show prediction)
- ✅ Simple Grad-CAM (even if heatmap quality is mediocre)

**Revised timeline for Plan B:**

| Hours 8-16 | All hands on integration |
| Hours 16-20 | Polish what works |
| Hours 20-24 | Demo prep with lower expectations |

---

## 📊 SUCCESS CRITERIA

### MINIMUM (Must have to present):
- [ ] One trained model (>85% accuracy)
- [ ] Working API endpoint
- [ ] Basic UI (upload → prediction)
- [ ] At least 3 demo images working

### TARGET (Good demo):
- [ ] ResNet50 with >90% accuracy
- [ ] Grad-CAM heatmaps
- [ ] Polished UI with confidence scores
- [ ] Deployed backend (or ngrok)
- [ ] 6 demo images (all classes)

### STRETCH (Impressive demo):
- [ ] 2+ models with ensemble
- [ ] Both deployed (Render + Streamlit Cloud)
- [ ] Batch processing
- [ ] Model comparison feature
- [ ] Real clinical validation discussion

---

## 🎯 FINAL CHECKLIST (Hour 22-24)

**2 Hours Before Demo:**
- [ ] Test complete flow 5 times
- [ ] All demo images work without errors
- [ ] Presentation slides finalized
- [ ] Backup video recorded
- [ ] GitHub README updated with real results

**1 Hour Before Demo:**
- [ ] Team members know their speaking parts
- [ ] Laptop/projector tested
- [ ] Internet connection verified (if using deployed version)
- [ ] Backup plan ready (local version + video)

**During Demo:**
- [ ] Start with most impressive result (clear pneumonia case)
- [ ] Show heatmap explanation
- [ ] Mention accuracy/recall numbers
- [ ] End with Normal case (shows balance)

---

## 💡 COMMUNICATION PROTOCOL

**Hourly Check-ins (Everyone):**
- 3:30 PM, 5:30 PM, 7:30 PM, 9:30 PM, 11:30 PM
- 1:30 AM, 3:30 AM, 5:30 AM, 7:30 AM, 9:30 AM, 11:30 AM

**Format:**
```
[Name] Status Update:
✅ Completed: [what's done]
🔄 In Progress: [what's being worked on]
⚠️ Blocked: [what's blocking you, if anything]
🎯 Next: [what's next in the next 2 hours]
```

**Slack/Discord channel:**
- `#general` - Team coordination
- `#ml` - Tanay model updates
- `#backend` - Akshayaa API updates
- `#frontend` - Dhanvi/Sajal UI updates
- `#integration` - Cross-team issues

**Emergency flag:**
If anyone is stuck >30 minutes, type "🚨 HELP NEEDED" and others jump in

---

## 🏆 LET'S GO!

Remember:
- ⏰ **Time is your enemy** - every minute counts
- 🎯 **Working > Perfect** - functional demo beats beautiful failure
- 🤝 **Help each other** - blocked teammates slow everyone down
- 📊 **Real results > Promises** - use actual numbers, not hopes

**You've got this. Now EXECUTE!** 💪

Last updated: 1:30 PM IST | Go time! 🚀
