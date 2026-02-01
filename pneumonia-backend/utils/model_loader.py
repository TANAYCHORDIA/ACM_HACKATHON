import os
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class EnsemblePneumoniaModel:
    def __init__(self, models_dir="models"):
        """
        Auto-load ensemble models from directory.
        Works with dummy predictions until real models are added.
        """
        self.models_dir = Path(models_dir)
        self.models = []
        self.metadata = None
        self.class_names = ["Normal", "Bacterial Pneumonia", "Viral Pneumonia"]
        self.input_size = (224, 224)
        
        # Try to import TensorFlow
        self.tf_available = False
        self.keras = None
        
        if self._models_exist():
            try:
                from tensorflow import keras as tf_keras
                self.keras = tf_keras
                self.tf_available = True
                logger.info("✅ TensorFlow loaded")
            except ImportError:
                logger.warning("⚠️ TensorFlow not installed. Install with: pip install tensorflow")
        
        self.load_models()
    
    def _models_exist(self):
        """Check if any model files exist."""
        if not self.models_dir.exists():
            return False
        h5_files = list(self.models_dir.glob("*.h5"))
        return len(h5_files) > 0
    
    def load_models(self):
        """Auto-load all models from directory."""
        try:
            self.models_dir.mkdir(exist_ok=True)
            
            h5_files = list(self.models_dir.glob("*.h5"))
            
            if not h5_files:
                logger.info("📁 No model files found in models/ - using dummy predictions")
                logger.info("👉 To use real models: Place .h5 files in pneumonia-backend/models/")
                return
            
            if not self.tf_available:
                logger.error("❌ TensorFlow not available but models found")
                logger.error("👉 Install with: pip install tensorflow")
                return
            
            for model_path in h5_files:
                try:
                    model = self.keras.models.load_model(str(model_path))
                    self.models.append({
                        'name': model_path.name,
                        'model': model
                    })
                    logger.info(f"✅ Loaded: {model_path.name}")
                except Exception as e:
                    logger.error(f"❌ Failed to load {model_path.name}: {str(e)}")
            
            npy_files = list(self.models_dir.glob("*.npy"))
            if npy_files:
                try:
                    self.metadata = np.load(str(npy_files[0]), allow_pickle=True).item()
                    logger.info(f"✅ Loaded metadata: {npy_files[0].name}")
                    
                    if isinstance(self.metadata, dict):
                        if 'class_names' in self.metadata:
                            self.class_names = self.metadata['class_names']
                        if 'input_size' in self.metadata:
                            self.input_size = tuple(self.metadata['input_size'])
                except Exception as e:
                    logger.warning(f"⚠️ Metadata file found but couldn't load: {str(e)}")
            
            if self.models:
                logger.info(f"🎉 Ensemble ready: {len(self.models)} models loaded")
            else:
                logger.warning("⚠️ No models loaded successfully - using dummy predictions")
                
        except Exception as e:
            logger.error(f"❌ Error during model loading: {str(e)}")
    
    def predict(self, preprocessed_image):
        """Make prediction (ensemble if models loaded, dummy otherwise)."""
        if not self.models:
            return self._dummy_prediction()
        
        try:
            all_predictions = []
            
            for model_info in self.models:
                model = model_info['model']
                prediction = model.predict(preprocessed_image, verbose=0)
                all_predictions.append(prediction[0])
            
            ensemble_probs = np.mean(all_predictions, axis=0)
            
            if self.metadata and isinstance(self.metadata, dict):
                if 'weights' in self.metadata:
                    weights = self.metadata['weights']
                    if len(weights) == len(all_predictions):
                        ensemble_probs = np.average(all_predictions, axis=0, weights=weights)
            
            predicted_class_idx = np.argmax(ensemble_probs)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(ensemble_probs[predicted_class_idx])
            
            probs_dict = {
                self.class_names[i]: float(ensemble_probs[i])
                for i in range(len(self.class_names))
            }
            
            return {
                "prediction": predicted_class,
                "class_id": int(predicted_class_idx),
                "confidence": confidence,
                "probabilities": probs_dict,
                "ensemble_size": len(self.models)
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            return self._dummy_prediction()
    
    def _dummy_prediction(self):
        """Dummy prediction when no models available."""
        return {
            "prediction": "Normal",
            "class_id": 0,
            "confidence": 0.95,
            "probabilities": {
                "Normal": 0.95,
                "Bacterial Pneumonia": 0.03,
                "Viral Pneumonia": 0.02
            },
            "ensemble_size": 0
        }
    
    def is_loaded(self):
        """Check if real models are loaded."""
        return len(self.models) > 0
    
    def get_info(self):
        """Get model information."""
        return {
            "models_loaded": len(self.models),
            "model_names": [m['name'] for m in self.models],
            "class_names": self.class_names,
            "input_size": self.input_size,
            "tensorflow_available": self.tf_available
        }

_model_instance = None

def get_model():
    """Get or create model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = EnsemblePneumoniaModel()
    return _model_instance
