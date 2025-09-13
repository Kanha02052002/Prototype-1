import os
import faiss
import pickle
import json
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import joblib
import requests

load_dotenv()

# LM Studio configuration
LM_STUDIO_API_BASE = "http://127.0.0.1:1234/v1" #http://localhost:1234/v1
LM_STUDIO_MODEL = "qwen/qwen3-4b-2507"  # "openai/gpt-oss-20b" -> almost 6 mins with reasoning

class RAGChatbotLM:
    _sentence_model = None
    
    def __init__(self, embed_dir='embeddings'):
        self.model = self._get_sentence_model()
        self.index = faiss.read_index(os.path.join(embed_dir, 'faiss_index.bin'))
        with open(os.path.join(embed_dir, 'chunks.pkl'), 'rb') as f:
            self.chunks = pickle.load(f)
        self.conversation_log = []

    @classmethod
    def _get_sentence_model(cls):
        """Get or create cached sentence transformer model"""
        if cls._sentence_model is None:
            print("Loading sentence transformer model...")
            cls._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Sentence transformer model loaded!")
        return cls._sentence_model

    def retrieve(self, query, k=3):
        embedding = self.model.encode([query])
        _, indices = self.index.search(embedding.astype('float32'), k)
        return [self.chunks[i] for i in indices[0]]

    def generate_response(self, prompt):
        """
            Lower temp - faster + deterministic response
            max_token - less = faster

        """
        try:
            response = requests.post(
                url=f"{LM_STUDIO_API_BASE}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": LM_STUDIO_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5,    # 0.7
                    "max_tokens": 300,      # 500
                    "top_p": 0.9,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                }),
                timeout=120  
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error generating response: {e}")
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
        initial_query = input("You: ")
        
        category = self.predict_category(initial_query)
        
        retrieved_chunks = self.retrieve(initial_query)
        context = "\n".join(retrieved_chunks)

        # prompt = (
        #     f"Based on the following context:\n{context}\n\n"
        #     f"Generate a follow-up question to narrow down the issue described: '{initial_query}'"
        # )
        prompt = f"""
            <prompt>
            <context>
                {context}
            </context>
            
            <task>
                Generate a follow-up question to narrow down the issue.
            </task>
            
            <constraints>
                <focus>{initial_query}</focus>
                <format>single question</format>
            </constraints>
            </prompt>
         """
        # prompt = f"""
        #     <prompt>
        #     <context>
        #         {context}
        #     </context>
            
        #     <task>
        #         Generate a follow-up question to narrow down the issue.
        #     </task>
            
        #     <constraints>
        #         <focus>{initial_query}</focus>
        #         <format>single question</format>
        #     </constraints>
        #     </prompt>
        #  """

        q1 = self.generate_response(prompt)
        print(f"Bot: {q1}")
        self.log_interaction(initial_query, q1)

        answer1 = input("You: ")
        self.log_interaction(answer1, "")

        # prompt2 = (
        #     f"Previously user said: '{initial_query}'.\n"
        #     f"Then bot asked: '{q1}'\n"
        #     f"User replied: '{answer1}'\n\n"
        #     f"Now generate another follow-up question to further narrow down the issue."
        # )
        prompt2 = f"""
            <prompt>
            <context>
                Previously user said: '{initial_query}'.
                Then bot asked: '{q1}'.
                User replied: '{answer1}'.
            </context>
            
            <task>
                Generate another follow-up question to further narrow down the issue.
            </task>
            
            <constraints>
                <format>single question</format>
                <purpose>diagnostic narrowing</purpose>
                <focus>{context}</focus>
            </constraints>
            </prompt>
        """

        q2 = self.generate_response(prompt2)
        print(f"Bot: {q2}")
        self.log_interaction(answer1, q2)

        answer2 = input("You: ")
        self.log_interaction(answer2, "")

        # prompt3 = (
        #     f"Previously user said: '{initial_query}'.\n"
        #     f"Then bot asked: '{q1}' and user replied: '{answer1}'.\n"
        #     f"Then bot asked: '{q2}' and user replied: '{answer2}'.\n\n"
        #     f"Now suggest 3-4 quick non-technical fixes or diagnostics related to this issue by observing the {context}. Give point-wise suggestions with navigation if needed."
        # )
        # prompt3 = f"""
        #     <prompt>
        #     <context>
        #         Previously user said: '{initial_query}'.
        #         Then bot asked: '{q1}' and user replied: '{answer1}'.
        #         Then bot asked: '{q2}' and user replied: '{answer2}'.
        #     </context>
        #     <role>
        #         Act as a IT person who work with resolving issue raised.
        #     </role>
        #     <task>
        #         Suggest 3-4 quick non-technical fixes or diagnostics related to this issue.
        #     </task>
            
        #     <constraints>
        #         <format>point-wise</format>
        #         <detail>include navigation if needed</detail>
        #         <focus>{context}</focus>
        #     </constraints>
        #     </prompt>
        # """
        prompt3 = f"""
            <prompt>
            <context>
                Previously user said: '{initial_query}'.
                Then bot asked: '{q1}' and user replied: '{answer1}'.
                Then bot asked: '{q2}' and user replied: '{answer2}'.
            </context>
            <role>
                Act as a IT person who work with resolving issue raised.
            </role>
            <task>
                Suggest 3-4 quick non-technical fixes or diagnostics related to this issue.
            </task>
            
            <constraints>
                <format>point-wise</format>
                <detail>include navigation if needed</detail>
                <focus>{context}</focus>
            </constraints>
            </prompt>
        """

        final_response = self.generate_response(prompt3)
        print(f"Bot: {final_response}")
        self.log_interaction(answer2, final_response)

        final_input = input("Did this solve your issue? (Type 'Issue solved' or anything else): ")
        if "solved" in final_input.lower():
            flag = "Solved by Bot"
        else:
            flag = f"Ticket raised for the issue: {initial_query}"

        final_log_entry = f"{flag} | Category: {category} | Backed by LM Studio Model: {LM_STUDIO_MODEL}"
        self.log_interaction(final_input, final_log_entry)
        self.save_log()
        print(f"\nFinal status: {flag}")
        print(f"Issue Category: {category}")

if __name__ == "__main__":
    bot = RAGChatbotLM()
    bot.run()