# src/utils/sentence_transformer_cache.py
import os
from sentence_transformers import SentenceTransformer
import shutil

class SentenceTransformerCache:
    # Class variable to cache the model
    _model = None
    _model_path = "models_cache/all-MiniLM-L6-v2"
    
    @classmethod
    def get_model(cls):
        """Get or create cached sentence transformer model"""
        if cls._model is None:
            cls._model = cls._load_or_download_model()
        return cls._model
    
    @classmethod
    def _load_or_download_model(cls):
        """Load model from local cache or download if not exists"""
        try:
            # Create cache directory if it doesn't exist
            os.makedirs("models_cache", exist_ok=True)
            
            # Check if model is already downloaded
            if os.path.exists(cls._model_path) and os.path.isdir(cls._model_path):
                print(f"🔄 Loading sentence transformer model from local cache: {cls._model_path}")
                model = SentenceTransformer(cls._model_path)
                print("✅ Sentence transformer model loaded from cache!")
            else:
                print("📥 Downloading sentence transformer model (one-time setup)...")
                # Download model
                model = SentenceTransformer('all-MiniLM-L6-v2')
                
                # Save model locally for future use
                print(f"💾 Saving model to local cache: {cls._model_path}")
                model.save(cls._model_path)
                print("✅ Sentence transformer model downloaded and cached!")
                
            return model
        except Exception as e:
            print(f"❌ Error loading sentence transformer model: {e}")
            print("💡 Falling back to online loading...")
            return SentenceTransformer('all-MiniLM-L6-v2')