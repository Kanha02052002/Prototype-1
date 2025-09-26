import os
from src.embedder.build_embeddings import build_embeddings
from src.classifier.train_classifier import train_hybrid_model
from src.logic import ChatLogic
import pandas as pd
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
    # bot.run()
    bot.run()

    # response = bot.jira_create_issue(initial_query, description)
    # if response.status_code == 201:
    #     print("\nIssue created successfully.")
    #     print("\nResponse:", response.text)
        
    # else:
    #     print("\nFailed to create issue.")
    

if __name__ == "__main__":
    main()