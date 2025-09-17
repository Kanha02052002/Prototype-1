# rag_chatbot_2.py
import os
import pandas as pd
import faiss
import pickle
import json
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import joblib
import requests

load_dotenv()

# LM Studio configuration
LM_STUDIO_API_BASE = "http://127.0.0.1:1234/v1" #http://localhost:1234/v1
LM_STUDIO_MODEL = "qwen/qwen3-4b-2507"  # "openai/gpt-oss-20b" -> almost 6 mins with reasoning

class RAGChatbotLM:
    _sentence_model = None
    _model_cache_path = "models_cache/all-MiniLM-L6-v2"
    
    def __init__(self, embed_dir='embeddings'):
        self.model = self._get_sentence_model()
        self.index = faiss.read_index(os.path.join(embed_dir, 'faiss_index.bin'))
        with open(os.path.join(embed_dir, 'chunks.pkl'), 'rb') as f:
            self.chunks = pickle.load(f)
        self.conversation_log = []
        self.conversation_state = []

    @classmethod
    def _get_sentence_model(cls):
        """Get or create cached sentence transformer model"""
        if cls._sentence_model is None:
            cls._sentence_model = cls._load_or_cache_model()
        return cls._sentence_model

    @classmethod
    def _load_or_cache_model(cls):
        """Load model from local cache or download and cache"""
        try:
            os.makedirs("models_cache", exist_ok=True)
            
            if os.path.exists(cls._model_cache_path) and os.path.isdir(cls._model_cache_path):
                # print("Loading sentence transformer model from cache...")
                model = SentenceTransformer(cls._model_cache_path)
                # print("Model loaded from cache!")
            else:
                # print("Downloading sentence transformer model...")
                model = SentenceTransformer('all-MiniLM-L6-v2')
                # print("Caching model locally...")
                model.save(cls._model_cache_path)
                # print("Model downloaded and cached!")
            return model
        except Exception as e:
            print(f"Error with model caching: {e}")
            # print("Loading model directly...")
            return SentenceTransformer('all-MiniLM-L6-v2')
        
    def get_initial_response(self, data=QUERY_SRC, query=""):
        """Get top 5 matching categories based on user query"""
        top_n = 5
        try:
            # Read CSV file
            df = pd.read_csv(data)
            if not query or df.empty:
                return []
            
            categories = df['Category'].tolist()
            questions = df['Q1'].tolist()
            
            # Encode questions and user query
            question_embeddings = self.model.encode(questions, convert_to_tensor=True)
            user_embedding = self.model.encode(query, convert_to_tensor=True)
            
            # Calculate similarities
            similarities = util.cos_sim(user_embedding, question_embeddings)[0]
            top_indices = similarities.argsort(descending=True)[:top_n]
            
            # Return list of top 5 categories with similarity scores
            top_categories = []
            for i in top_indices:
                top_categories.append({
                    'category': categories[i],
                    'similarity': similarities[i].item(),
                    'q1': questions[i]
                })
            
            return top_categories
        except Exception as e:
            print(f"Error in get_initial_response: {e}")
            return []

    def get_category_questions(self, data=QUERY_SRC, category=""):
        """Get all questions for a specific category"""
        try:
            df = pd.read_csv(data)
            if not category or df.empty:
                return []
            
            # Find the row matching the category
            category_row = df[df['Category'] == category]
            if category_row.empty:
                return []
            
            row = category_row.iloc[0]
            
            # Extract all questions for this category
            questions = []
            if pd.notna(row['Q1']):
                questions.append(row['Q1'])
            if pd.notna(row.get('Q2 (if Yes)', '')):
                questions.append(row.get('Q2 (if Yes)', ''))
            if pd.notna(row.get('Q2a (Deeper probing if Yes)', '')):
                questions.append(row.get('Q2a (Deeper probing if Yes)', ''))
            if pd.notna(row.get('Q3 (if No at any stage)', '')):
                questions.append(row.get('Q3 (if No at any stage)', ''))
            
            return questions
        except Exception as e:
            print(f"Error in get_category_questions: {e}")
            return []


    def retrieve(self, query, k=3):
        embedding = self.model.encode([query])
        _, indices = self.index.search(embedding.astype('float32'), k)
        return [self.chunks[i] for i in indices[0]]

    # def generate_response(self, prompt):
    #     """Generate response using LM Studio"""
    #     try:
    #         response = requests.post(
    #             url=f"{LM_STUDIO_API_BASE}/chat/completions",
    #             headers={
    #                 "Content-Type": "application/json",
    #             },
    #             data=json.dumps({
    #                 "model": LM_STUDIO_MODEL,
    #                 "messages": [{"role": "user", "content": prompt}],
    #                 "temperature": 0.3,
    #                 "max_tokens": 200,
    #                 "top_p": 0.9,
    #                 "frequency_penalty": 0,
    #                 "presence_penalty": 0
    #             }),
    #             timeout=120  
    #         )
    #         response.raise_for_status()
    #         return response.json()['choices'][0]['message']['content']
    #     except Exception as e:
    #         print(f"Error generating response: {e}")
    #         return "I'm sorry, I'm having trouble generating a response right now."
    def generate_response(self, prompt):
        try:
            response = requests.post(
                url=f"{LM_STUDIO_API_BASE}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": LM_STUDIO_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 200,
                    "top_p": 0.9,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"HTTP Error: {e}")
            if e.response is not None:
                print("Response content:", e.response.text)
                return "I'm sorry, I'm having trouble generating a response right now."

    def log_interaction(self, user_input, bot_response):
        self.conversation_log.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_input,
            "bot": bot_response
        })

    def save_log(self, filename="logs/conversation_logs.txt"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'a', encoding='utf-8') as f:
            for entry in self.conversation_log:
                f.write(f"[{entry['timestamp']}] User: {entry['user']}\n")
                f.write(f"[{entry['timestamp']}] Bot: {entry['bot']}\n")
            f.write("\n" + "="*60 + "\n")

    def predict_category(self, text, model_path='models/hybrid_classifier.pkl'):
        try:
            clf, vectorizer = joblib.load(model_path)
            vec = vectorizer.transform([text])
            pred = clf.predict(vec)[0]
            return pred
        except Exception as e:
            print(f"Error predicting category: {e}")
            return "Unknown"

    def run(self):
        print("Hello! I'm your IT Support Assistant. Please describe your issue.")
        # print("Type 'quit' to exit")
        
        initial_query = input("You: ")
        top_categories = self.get_initial_response(QUERY_SRC, initial_query)
        if not top_categories:
            # print("No matching categories found.")
            return False
        
        print("\nTop matching categories:")
        for i, cat in enumerate(top_categories, 1):
            print(f"{i}. {cat['category']}")

        try:
            choice = int(input("\nPlease select a category (1-5): "))
            if 1 <= choice <= 5:
                selected_category = top_categories[choice-1]['category']
                print(f"\nSelected category: {selected_category}")
            else:
                # print("Invalid choice. Using the first category.")
                selected_category = top_categories[0]['category']
        except (ValueError, IndexError):
            # print("Invalid input. Using the first category.")
            selected_category = top_categories[0]['category']

        category_questions = self.get_category_questions(QUERY_SRC, selected_category)
        
        if not category_questions:
            # print("No questions found for the selected category.")
            return False
    
        # print(f"\nFound {len(category_questions)} questions for {selected_category}")
        
        # if initial_query.lower().strip() in ['quit', 'exit', 'q']:
        #     print("Goodbye!")
        #     return False
        
        # Sequential execution
        category = self.predict_category(initial_query)
        retrieved_chunks = self.retrieve(initial_query)
        context = "\n".join(retrieved_chunks)
        # print(f"Category: {category}")

        # Question 1
        prompt1 = f"""
           <prompt>
            <context>
                 {context}
             </context>
            
             <task>
                 Generate a follow-up question to narrow down the issue. Focus on quick understanding.
             </task>
           
             <constraints>
                <focus>{initial_query} and question pattern as {category_questions[0]}</focus>
                 <format>single question, and it need to be in format of yes/no or just 2-3 words of response</format>
             </constraints>
             </prompt>
        """
        
        q1 = self.generate_response(prompt1)
        print(f"Bot: {q1}")
        self.log_interaction(initial_query, q1)

        answer1 = input("You: ")
        # if answer1.lower().strip() in ['quit', 'exit', 'q']:
        #     print("Goodbye!")
        #     return False
            
        self.conversation_state.append({"Q1": q1, "A1": answer1})
        self.log_interaction(answer1, "")

        # Question 2
        prompt2 = f"""
            <prompt>
            <context>
                Previously user said: '{initial_query}'.
                Then bot asked: '{q1}'.
                User replied: '{answer1}'.
                Refer context for suggestions: {context}.
            </context>
            
            <task>
                Generate another follow-up question to further narrow down the issue. Quick and to the point.
            </task>
            
            <constraints>
                <focus>question pattern as {category_questions[1]}</focus>
                <format>single question,and it need to be in format of yes/no or just 2-3 words of response</format>
                <purpose>diagnostic narrowing</purpose>
                <action>to get better understanding of the issue</action>
                <avoid>Avoid repeating similar question like {q1}</avoid>
            </constraints>
            </prompt>
        """
        # <focus>{context}</focus>
        q2 = self.generate_response(prompt2)
        print(f"Bot: {q2}")
        self.log_interaction(answer1, q2)

        answer2 = input("You: ")
        # if answer2.lower().strip() in ['quit', 'exit', 'q']:
        #     print("Goodbye!")
        #     return False
            
        self.conversation_state.append({"Q2": q2, "A2": answer2})
        self.log_interaction(answer2, "")

        # Question 3
        prompt3 = f"""
            <prompt>
            <context>
                Previously user said: '{initial_query}'.
                Then bot asked: '{q1}' and user replied: '{answer1}'.
                Then bot asked: '{q2}' and user replied: '{answer2}'.
                Refer context for suggestions: {context}.
            </context>
            
            <task>
                Generate another follow-up question to further narrow down the issue. Quick and to the point.
            </task>
            
            <constraints>
                <focus>question pattern as {category_questions[2]}</focus>
                <format>single question,and it need to be in format of yes/no or just 2-3 words of response</format>
                <purpose>more diagnostic narrowing, to get better understanding of the issue</purpose>
                
                <action>to get better understanding of the issue</action>
                <avoid>Avoid repeating similar question like {q2}</avoid>
            </constraints>
            </prompt>
        """
        # <focus>{context}</focus>
        q3 = self.generate_response(prompt3)
        print(f"Bot: {q3}")
        self.log_interaction(answer2, q3)

        answer3 = input("You: ")
        # if answer3.lower().strip() in ['quit', 'exit', 'q']:
        #     print("Goodbye!")
        #     return False
            
        self.conversation_state.append({"Q3": q3, "A3": answer3})
        self.log_interaction(answer3, "")

        q4 = "Please provide some more context or details about the issue, in form of text or screenshots. (optional)"
        print(f"Bot: {q4}")
        answer4 = input("You: ")

        answer4_processed = answer4 if answer4.strip() != "" else "No additional details provided"

        self.conversation_state.append({"Q4": q4, "A4": answer4_processed})
        self.log_interaction(answer4_processed, "Additional details collected")
        # final_input = input("Did this solve your issue? (Type 'Issue solved' or anything else): ")
        # if final_input.lower().strip() in ['quit', 'exit', 'q']:
        #     print("Goodbye!")
        #     return False
            
        # if "solved" in final_input.lower():
        #     flag = "Solved by Bot"
        #     print("Great! Happy to help!")
        # else:
        flag = f"Ticket raised for the issue: {initial_query}"
        print(f"I'll create a ticket for this issue.\n{flag}")
        
        final_log_entry = f"""{flag} | Category: {category} 
                                | Backed by LM Studio Model: {LM_STUDIO_MODEL} | Initial Category: {selected_category}
                            """
        self.log_interaction("Ticket raised automatically", final_log_entry)
        self.save_log()
        print(f"\nFinal status: {flag}")
        print(f"Issue Category: {category}")

if __name__ == "__main__":
    bot = RAGChatbotLM()
    bot.run()