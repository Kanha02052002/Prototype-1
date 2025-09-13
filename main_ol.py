# main.py
import os
from src.embedder.build_embeddings import build_embeddings
from src.classifier.train_classifier import train_hybrid_model
from src.rag_chatbot_ollama import RAGChatbotOllama
import warnings
warnings.filterwarnings("ignore")   

def main():
    # Check if Ollama is running
    try:
        import requests
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if response.status_code != 200:
            return
    except:
        print("Cannot connect to Ollama")
        print("Please install Ollama from: https://ollama.ai")
        print("And pull a model: ollama pull llama2")
        return

    if not os.path.exists('embeddings/faiss_index.bin'):
        print("Setting up embeddings...")
        build_embeddings()

    if not os.path.exists('models/hybrid_classifier.pkl'):
        print("Training hybrid classifier...")
        train_hybrid_model()

    print("Starting IT Support Chatbot (Ollama Version)...")
    bot = RAGChatbotOllama()
    bot.run()

if __name__ == "__main__":
    main()