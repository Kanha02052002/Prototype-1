import os
from src.embedder.build_embeddings import build_embeddings
from src.classifier.train_classifier import train_hybrid_model
from src.logic import ChatLogic
import warnings
warnings.filterwarnings("ignore")   



def main():
    if not os.path.exists('embeddings/faiss_index.bin'):
        print("Setting up embeddings...")
        build_embeddings()

    if not os.path.exists('models/hybrid_classifier.pkl'):
        print("Training hybrid classifier...")
        train_hybrid_model()

    print("Starting IT Support Chatbot.........")


    bot = ChatLogic()
    bot.run()

if __name__ == "__main__":
    main()