import os
import pandas as pd
import faiss
import pickle
import json
import time
import uuid
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import joblib
import requests
from requests.auth import HTTPBasicAuth
import json
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError, DuplicateKeyError

load_dotenv(".env")

QUERY_SRC="data\Queries.csv"

URL = os.getenv("JIRA_INSTANCE_URL")
USERNAME = os.getenv("JIRA_INSTANCE_USERNAME")
API_TOKEN = os.getenv("JIRA_RESTAPI_KEY")
JIRA_INSTANCE_PROJECT = os.getenv("JIRA_INSTANCE_PROJECT")
JIRA_PROJECT_ID = os.getenv("JIRA_PROJECT_ID")

LM_STUDIO_API = os.getenv("LM_STUDIO_API_BASE")
LM_STUDIO_MODEL = "qwen/qwen3-4b-2507"  

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DATABASE")
GLOBAL_CLUSTER = os.getenv("GLOBAL_CLUSTER")
INSTANCE_CLUSTER = os.getenv("INSTANCE_CLUSTER")

class ChatLogic:
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
                model = SentenceTransformer(cls._model_cache_path)
            else:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                model.save(cls._model_cache_path)
            return model
        except Exception as e:
            print(f"Error with model caching: {e}")
            return SentenceTransformer('all-MiniLM-L6-v2')
        
    def get_initial_response(self, data=QUERY_SRC, query=""):
        """Get top 3 matching categories based on user query"""
        top_n = 3
        try:
            df = pd.read_csv(data)
            if not query or df.empty:
                return []
            
            categories = df['Category'].tolist()
            questions = df['Q1'].tolist()
            
            question_embeddings = self.model.encode(questions, convert_to_tensor=True)
            user_embedding = self.model.encode(query, convert_to_tensor=True)
            
            similarities = util.cos_sim(user_embedding, question_embeddings)[0]
            top_indices = similarities.argsort(descending=True)[:top_n]
            
            top_categories = []
            for i in top_indices:
                top_categories.append({
                    'category': categories[i]
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
            
            category_row = df[df['Category'] == category]
            if category_row.empty:
                return []
            
            row = category_row.iloc[0]
            
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

    def generate_response(self, prompt):
        try:
            response = requests.post(
                url=f"{LM_STUDIO_API}/chat/completions",
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

    # def log_interaction(self, user_input, bot_response):
    #     self.conversation_log.append({
    #         "user": user_input,
    #         "bot": bot_response
    #     })

    # def save_log(self, filename="logs/conversation_logs.txt"):
    #     os.makedirs(os.path.dirname(filename), exist_ok=True)
    #     with open(filename, 'a', encoding='utf-8') as f:
    #         for entry in self.conversation_log:
    #             f.write(f"User: {entry['user']}\n")
    #             f.write(f"Bot: {entry['bot']}\n")
    #         f.write("\n" + "="*60 + "\n")

    def predict_category(self, text, model_path='models/hybrid_classifier.pkl'):
        try:
            clf, vectorizer = joblib.load(model_path)
            vec = vectorizer.transform([text])
            pred = clf.predict(vec)[0]
            return pred
        except Exception as e:
            print(f"Error predicting category: {e}")
            return "Unknown"
        


    def question_generation(self,**kwargs):
        context=kwargs.get("context","")
        initial_query=kwargs.get("initial_query","")
        prev_que_1=kwargs.get("prev_que_1","")
        prev_que_2=kwargs.get("prev_que_2","")
        question_list=kwargs.get("question_list","")
        answer_1=kwargs.get("answer_1","")
        answer_2=kwargs.get("answer_2","")
        id=kwargs.get("id","")

        prompt=""

        if id==0:
            prompt=f"""
           <prompt>
            <context>
                 {context}
             </context>
            
             <task>
                 Generate a follow-up question to narrow down the issue. Focus on quick understanding.
             </task>
           
             <constraints>
                <focus>{initial_query} and question pattern as {question_list[id]}</focus>
                 <format>single question, and it need to be in format of yes/no or just 2-3 words of response</format>
             </constraints>
             </prompt>
        """
        elif id==1:
            prompt = f"""
            <prompt>
            <context>
                Previously user said: '{initial_query}'.
                Then bot asked: '{prev_que_1}'.
                User replied: '{answer_1}'.
                Refer context for suggestions: {context}.
            </context>
            
            <task>
                Generate another follow-up question to further narrow down the issue. Quick and to the point.
            </task>
            
            <constraints>
                <focus>question pattern as {question_list[id]}</focus>
                <format>single question,and it need to be in format of yes/no or just 2-3 words of response</format>
                <purpose>diagnostic narrowing</purpose>
                <action>to get better understanding of the issue</action>
                <avoid>Avoid repeating similar question like {prev_que_1}</avoid>
            </constraints>
            </prompt>
        """
        elif id==2:
            prompt = f"""
            <prompt>
            <context>
                Previously user said: '{initial_query}'.
                Then bot asked: '{prev_que_1}' and user replied: '{answer_1}'.
                Then bot asked: '{prev_que_2}' and user replied: '{answer_2}'.
                Refer context for suggestions: {context}.
            </context>
            
            <task>
                Generate another follow-up question that closely mirrors the style and structure of {question_list[id]}, 
                but keeps alignment with the given context.
            </task>
            
            <constraints>
                <focus>question must be almost the same as {question_list[id]} but adapted to current context</focus>
                <format>single question, short, yes/no type or 2-3 words response</format>
                <purpose>maintain consistency in questioning style while narrowing down the issue further</purpose>
                <avoid>Avoid repeating the exact same wording as {prev_que_2}, but keep it as close as possible to {question_list[id]}</avoid>
            </constraints>
            </prompt>
        """

        return self.generate_response(prompt=prompt)


    def run(self):
        print("Hello! I'm your IT Support Assistant. Please describe your issue.")
        
        self.key=self.create_unique_key()
        print("Session ID:", self.key)
        
        initial_query = input("You: ")
        
        top_categories = self.get_initial_response(QUERY_SRC, initial_query)
        if not top_categories:
            return False
        
        print("\nTop matching categories:")
        for i, cat in enumerate(top_categories, 1):
            print(f"{i}. {cat['category']}")

        try:
            choice = int(input("\nPlease select a category (1-3): "))
            if 1 <= choice <= 5:
                selected_category = top_categories[choice-1]['category']
                print(f"Selected category: {selected_category}")
            else:
                selected_category = top_categories[0]['category']
        except (ValueError, IndexError):
            selected_category = top_categories[0]['category']

        category_questions = self.get_category_questions(QUERY_SRC, selected_category)
        
        if not category_questions:
            return False
    
        category = self.predict_category(initial_query)
        retrieved_chunks = self.retrieve(initial_query)
        context = "\n".join(retrieved_chunks)
        
        q1=self.question_generation(context=context, 
                                    initial_query=initial_query, 
                                    question_list=category_questions, 
                                    id=0)
        print(f"Bot: {q1}")
        answer1 = input("You: ")
        
        q2=self.question_generation(context=context, 
                                    initial_query=initial_query, 
                                    question_list=category_questions, 
                                    id=1,
                                    prev_que_1=q1,
                                    answer_1=answer1)
        print(f"Bot: {q2}")
        answer2 = input("You: ")

        q3=self.question_generation(context=context, 
                                    initial_query=initial_query, 
                                    question_list=category_questions, 
                                    id=2,
                                    prev_que_1=q1,
                                    answer_1=answer1,
                                    prev_que_2=q2,
                                    answer_2=answer2)
        print(f"Bot: {q3}")
        answer3 = input("You: ")

        q4 = "Please provide some more context or details about the issue, in form of text or screenshots. (optional)"
        print(f"Bot: {q4}")
        answer4 = input("You: ")

        answer4_processed = answer4 if answer4.strip() != "" else "No additional details provided"

        flag = f"Ticket raised for the issue: {initial_query}"
        
        json_data = {
            "session_id": self.key,
            "user_initial_query": initial_query,
            "similarity_category": top_categories,
            "user_selected_category": selected_category,
            "predefined_category_questions": category_questions,
            "q_a": {
                "Q1": {"question": q1, "answer": answer1},
                "Q2": {"question": q2, "answer": answer2},
                "Q3": {"question": q3, "answer": answer3},
                "Q4": {"question": q4, "answer": answer4_processed}
            },
            "upload": None,
            "predicted_category": category,
            "status": flag,
            "id": None,
            "key": None,
            "datetime": datetime.now(),
            "deleted_temp_data": None
        }
        

        conversation_summary = f"""
        User_Selected_Category: {selected_category}
        Q1: {q1}
        A1: {answer1}
        Q2: {q2}
        A2: {answer2}   
        Q3: {q3}    
        A3: {answer3}   
        Q4: {q4}    
        A4: {answer4_processed}
        """
        
        self.add_to_mongo_instance(initial_query, conversation_summary, session_id=self.key)
        response=self.jira_create_issue()
        if response.status_code==201:
            resp_data = response.json()
            json_data["id"] = resp_data.get("id")
            json_data["key"] = resp_data.get("key")
            json_data["deleted_temp_data"]=self.clear_instance_cluster(session_id=self.key).get("timestamp")
            self.add_to_mongo_global(json_data, self.key)
            
            print("Ticket is up")
        else:
            print("Issue")

        
    
    def create_unique_key(self):
        return "air-"+str(uuid.uuid4())

    def initialize_mongo(self):
        try:
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            db = client[MONGODB_DB]
            db[INSTANCE_CLUSTER].create_index("session_id", unique=True)
            db[GLOBAL_CLUSTER].create_index("session_id", unique=True)
            return client
        except ConnectionFailure:
            return None
        
    def add_to_mongo_instance(self, initial_query, description, session_id=None, upload_url=None):
        client=self.initialize_mongo()
        if client is None:
            return
        db=client[MONGODB_DB]
        collection_instance=db[INSTANCE_CLUSTER]
        collection_instance.create_index("session_id", unique=True)
        if not collection_instance.find_one({"session_id": session_id}):
            collection_instance.insert_one({
                "session_id": session_id,
                "initial_query": initial_query,
                "description": description,
                "uploads": upload_url,
            })
            return {"status": "success", "session_id": session_id}
        else:
            return {"status": "exists", "session_id": session_id}
        
    def add_to_mongo_global(self, json_data, session_id):
        client = self.initialize_mongo()
        if client is None:
            return {"status": "error", "message": "MongoDB not connected"}

        db = client[MONGODB_DB]
        collection_global = db[GLOBAL_CLUSTER]

        try:
            # 🔹 Ensure session_id is always present and consistent
            if not json_data.get("session_id"):
                json_data["session_id"] = session_id

            # 🔹 Insert document
            collection_global.insert_one(json_data)
            return {"status": "success", "session_id": json_data["session_id"]}

        except DuplicateKeyError:
            return {
                "status": "exists",
                "session_id": json_data.get("session_id"),
                "message": "Document with this session_id already exists"
            }

        except PyMongoError as e:
            return {"status": "error", "message": str(e)}
    
    def fetch_mongo_data(self):
        client=self.initialize_mongo()
        if client is None:
            return
        db=client[MONGODB_DB]
        collection=db[INSTANCE_CLUSTER]
        data=collection.find_one({"session_id": str(self.key)})
        if data:
            summary = data.get("initial_query")
            description = data.get("description")
        else:
            summary, description = None, None
        return summary, description


    def jira_create_issue(self):
        summary, desc = self.fetch_mongo_data()
        
        if not summary or not desc:
            return {"status":"error","message":"No data found in database, issue not created"}
        
        auth = HTTPBasicAuth(USERNAME, API_TOKEN)

        headers={
            "Accept":"application/json",
            "Content-Type":"application/json"
        }
        
        payload = json.dumps({
        "fields": {
            "project": {
            "key": JIRA_INSTANCE_PROJECT        
            },
            "issuetype": {
            "id": JIRA_PROJECT_ID        
            },
            "summary": summary,
            "description": {
            "type": "doc",
            "version": 1,
            "content": [
                {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": desc}
                ]
                }
            ]
            }
        }
        })
        response = requests.request(
        "POST",
        URL + "rest/api/3/issue",
        data=payload,
        headers=headers,
        auth=auth
        )
        return response
    
    def clear_instance_cluster(self, session_id):
        client=self.initialize_mongo()
        if not client:
            return {"status": "error", "Message":"DB not connected"}
        db=client[MONGODB_DB]
        collection=db[INSTANCE_CLUSTER]
        try:
            collection.delete_one({"session_id": session_id})
            return {"status":"success", "Message": "DB cleared", "timestamp": datetime.now()}
        except Exception as e:
            return {"status": "error", "Message": str(e)}
        