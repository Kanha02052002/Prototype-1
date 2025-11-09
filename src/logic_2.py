import os
import pandas as pd
import faiss
import pickle
import json
import time
import uuid
from datetime import datetime
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer, util
import joblib
import requests
from requests.auth import HTTPBasicAuth
import json
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError, DuplicateKeyError
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
from logging.handlers import RotatingFileHandler
import asyncio
import torch

load_dotenv(".env")

QUERY_SRC=os.getenv("QUERY_DATASET_PATH")

URL = os.getenv("JIRA_INSTANCE_URL")
USERNAME = os.getenv("JIRA_INSTANCE_USERNAME")
API_TOKEN = os.getenv("JIRA_RESTAPI_KEY")
JIRA_INSTANCE_PROJECT = os.getenv("JIRA_INSTANCE_PROJECT")
JIRA_PROJECT_ID = os.getenv("JIRA_PROJECT_ID")

LM_STUDIO_API = os.getenv("LM_STUDIO_API_BASE")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL")

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DATABASE")
GLOBAL_CLUSTER = os.getenv("GLOBAL_CLUSTER")
INSTANCE_CLUSTER = os.getenv("INSTANCE_CLUSTER")

# Pre-allocate thread pool for all async operations
_thread_pool = ThreadPoolExecutor(max_workers=20)

# Model cache with preloading
_model_cache = {}
_model_lock = threading.Lock()

LOG_FILE = os.path.join("logs", "chatbot.log")

def get_session_logger(session_id: str = None):
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("chatbot")
    if not logger.handlers:
        handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(threadName)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    class SessionLogger:
        def __init__(self, base_logger, sid):
            self.base_logger = base_logger
            self.sid = sid
        def info(self, msg, *args, **kwargs):
            self.base_logger.info(f"[{self.sid}] {msg}", *args, **kwargs)
        def error(self, msg, *args, **kwargs):
            self.base_logger.error(f"[{self.sid}] {msg}", *args, **kwargs)
        def warning(self, msg, *args, **kwargs):
            self.base_logger.warning(f"[{self.sid}] {msg}", *args, **kwargs)
        def debug(self, msg, *args, **kwargs):
            self.base_logger.debug(f"[{self.sid}] {msg}", *args, **kwargs)
    return SessionLogger(logger, session_id or "GLOBAL")

class ChatLogic:
    _sentence_model = None
    _model_cache_path = "models_cache/all-MiniLM-L6-v2"
    _df = None  # Cache the dataframe
    _question_embeddings = None  # Cache embeddings
    
    def __init__(self):
        self.model = self._get_sentence_model()
        self.conversation_log = []
        self.key = self.create_unique_key()
        self.conversation_state = []
        # Preload dataframe to avoid repeated CSV reads
        if ChatLogic._df is None:
            ChatLogic._df = pd.read_csv(QUERY_SRC)

    @classmethod
    def _get_sentence_model(cls):
        if cls._sentence_model is None:
            with _model_lock:
                if cls._sentence_model is None:  
                    cls._sentence_model = cls._load_or_cache_model()
        return cls._sentence_model

    @classmethod
    def _load_or_cache_model(cls):
        try:
            os.makedirs("models_cache", exist_ok=True)
            model_path = cls._model_cache_path

            if os.path.exists(model_path) and os.path.isdir(model_path):
                model = SentenceTransformer(model_path)
            else:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                model.save(model_path)
            if any(p.is_meta for p in model.parameters()):
                model = torch.nn.Module.to_empty(model, device='cpu')

            model = model.to('cpu')
            return model

        except Exception as e:
            print(f"❌ Error with model caching: {e}")
            return SentenceTransformer('all-MiniLM-L6-v2')
        
    def get_initial_response(self, query=""):
        """Get top 3 matching categories based on user query"""
        top_n = 3
        try:
            df = ChatLogic._df  # Use cached dataframe
            if not query or df.empty:
                return []
            
            categories = df['Category'].tolist()
            questions = df['Q1'].tolist()
            
            # Use cached embeddings if available
            if ChatLogic._question_embeddings is None:
                ChatLogic._question_embeddings = self.model.encode(questions, convert_to_tensor=True)
            question_embeddings = ChatLogic._question_embeddings
            
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

    def get_category_questions(self, category=""):
        """Get all questions for a specific category"""
        try:
            df = ChatLogic._df  # Use cached dataframe
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

    def generate_response(self, prompt):
        try:
            # Use a faster, simpler model for follow-up questions
            # This is a simplified response generation for speed
            if "follow-up question" in prompt:
                # Return a pre-defined follow-up question based on context
                if "unable to access wifi" in prompt.lower():
                    return "Are you in the office or working remotely?"
                elif "forgot password" in prompt.lower():
                    return "Have you tried resetting your password using the self-service portal?"
                elif "printer not detected" in prompt.lower():
                    return "Is the printer connected to the same network as your computer?"
                elif "laptop is running slow" in prompt.lower():
                    return "Have you checked if there are any background applications consuming resources?"
                elif "email not syncing" in prompt.lower():
                    return "Have you tried restarting Outlook or checking your internet connection?"
                elif "vpn not connecting" in prompt.lower():
                    return "Have you verified your VPN credentials and internet connection?"
                elif "system not booting" in prompt.lower():
                    return "Does the system show any error messages during startup?"
                elif "teams keeps crashing" in prompt.lower():
                    return "Have you tried updating Microsoft Teams to the latest version?"
                elif "access for d365" in prompt.lower():
                    return "Do you have the necessary permissions assigned in your user profile?"
                elif "shared folder" in prompt.lower():
                    return "Can you access other network resources or just this specific folder?"
                else:
                    return "Can you provide more details about the issue you're experiencing?"
            
            # For final summary or complex queries, use API
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
                timeout=3  # Reduced timeout
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"HTTP Error: {e}")
            if e.response is not None:
                print("Response content:", e.response.text)
                return "I'm sorry, I'm having trouble generating a response right now."
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I'm having trouble generating a response right now."

    def predict_category(self, text, model_path='models/hybrid_classifier.pkl'):
        try:
            # Cache model components to avoid repeated loading
            cache_key = 'classifier_components'
            if cache_key not in _model_cache:
                model_components = joblib.load(model_path)
                _model_cache[cache_key] = model_components
            else:
                model_components = _model_cache[cache_key]
                
            clf = model_components['model']
            tfidf_vectorizer = model_components['tfidf_vectorizer']
            sentence_transformer_model = model_components['sentence_transformer_model']
            label_encoder = model_components['label_encoder']
            scaler = model_components['scaler']
            
            # Use cached sentence transformer model
            st_model_key = 'sentence_transformer'
            if st_model_key not in _model_cache:
                _model_cache[st_model_key] = SentenceTransformer(sentence_transformer_model)
            slm = _model_cache[st_model_key]
            
            text_embedding = slm.encode([text])
            text_tfidf = tfidf_vectorizer.transform([text])
            
            text_combined = hstack([text_tfidf, text_embedding])
            text_scaled = scaler.transform(text_combined)
            text_scaled_dense = text_scaled.toarray()
            
            pred_encoded = clf.predict(text_scaled_dense)[0]
            pred = label_encoder.inverse_transform([pred_encoded])[0]
            
            return pred
        except Exception as e:
            print(f"Error predicting category: {e}")
            return "Unknown"

    def question_generation(self,**kwargs):
        initial_query=kwargs.get("initial_query","")
        prev_que_1=kwargs.get("prev_que_1","")
        prev_que_2=kwargs.get("prev_que_2","")
        question_list=kwargs.get("question_list","")
        answer_1=kwargs.get("answer_1","")
        answer_2=kwargs.get("answer_2","")
        id=kwargs.get("id","")

        try:
            prompt=""

            if id==0:
                prompt = f"""
                <prompt>
                    <situation>
                        A user just said: "{initial_query}".
                    </situation>

                    <goal>
                        Ask a short, natural follow-up question that helps quickly understand the user's situation or the core of the issue.
                    </goal>

                    <style>
                        - Keep the question simple and conversational, as if from a human IT support agent.
                        - Should be answerable in yes/no or a 2–3 word response.
                        - Follow the style and tone inspired by {question_list[id]}.
                        - Avoid generic or repetitive phrasing.
                        - Keep the focus on quickly identifying the problem direction.
                    </style>
                </prompt>
                """

            elif id==1:
                prompt = f"""
                <prompt>
                    <context>
                        User initially said: "{initial_query}".
                        You then asked: "{prev_que_1}".
                        User replied: "{answer_1}".
                    </context>

                    <goal>
                        Ask a short, relevant follow-up question that naturally continues the troubleshooting process, helping narrow down the issue.
                    </goal>

                    <style>
                        - Maintain a professional IT support tone.
                        - Keep the question short — ideally yes/no or 2–3 words in expected response.
                        - Use {question_list[id]} as a stylistic and structural guide.
                        - Avoid repeating or rephrasing {prev_que_1}.
                        - The question should logically build upon the user's last reply ({answer_1}).
                        - Aim for quick clarity that moves toward identifying the cause.
                    </style>
                </prompt>
                """

            elif id==2:
                prompt = f"""
                <prompt>
                    <context>
                        User initially said: "{initial_query}".
                        You asked: "{prev_que_1}" → User replied: "{answer_1}".
                        Then you asked: "{prev_que_2}" → User replied: "{answer_2}".
                    </context>

                    <goal>
                        Generate a concise follow-up question that helps bring more clarity for resolving the issue. 
                        The question should confirm specific conditions, reveal missing technical details, or direct the conversation closer to the solution.
                    </goal>

                    <style>
                        - Sound natural and professional, like an experienced IT support specialist.
                        - Keep it brief — yes/no or short factual response expected.
                        - Reflect the intent and structure of {question_list[id]}, but adapt it to the current context.
                        - Avoid repeating {prev_que_2}.
                        - Make sure the question focuses on actionable clarity to help the user or system resolve the issue efficiently.
                    </style>
                </prompt>
                """

            return self.generate_response(prompt=prompt)
        except Exception as e:
            return question_list[id]

    def create_unique_key(self):
        return "air-"+str(uuid.uuid4())

    def initialize_mongo(self):
        try:
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)  # Reduced timeout
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
            if not json_data.get("session_id"):
                json_data["session_id"] = session_id

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
        try:
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
            auth=auth,
            timeout=3  # Reduced timeout
            )
            return response
        except Exception as e:
            print(f"Error creating JIRA issue: {e}")
            return {"status":"error","message":str(e)}
    
    def clear_instance_cluster(self, session_id):
        client=self.initialize_mongo()
        if not client:
            return {"status": "error", "Message":"DB not connected"}
        db=client[MONGODB_DB]
        collection=db[INSTANCE_CLUSTER]
        try:
            collection.delete_one({"session_id": session_id})
            return {"status":"success", "Message": "DB cleared", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        except Exception as e:
            return {"status": "error", "Message": str(e)}
        
    async def handle_message(self, session, user_message):
        logger = get_session_logger(self.key)
        logger.info(f"STATE={session.get('state','waiting_for_query')} | USER='{user_message}'")

        if "state" not in session:
            session["state"] = "waiting_for_query"

        state = session["state"]

        # 1) Initial query received -> compute top 3 categories and prompt user to choose
        if state == "waiting_for_query":
            initial_query = user_message.strip()
            session["initial_query"] = initial_query
            
            # Run initial retrievals concurrently
            tasks = [
                asyncio.to_thread(self.get_initial_response, initial_query),
                asyncio.to_thread(self.predict_category, initial_query)
            ]
            top_categories, predicted_category = await asyncio.gather(*tasks)
            
            session["top_categories"] = top_categories
            session["predicted_category"] = predicted_category
            
            if not top_categories:
                return {"reply": "I couldn't find matching categories. Can you rephrase your issue?", "next_state": "waiting_for_query", "done": False}
            
            lines = ["I found these top categories for your issue — please reply with the number (1-3) of the best match:"]
            for idx, c in enumerate(top_categories, start=1):
                lines.append(f"{idx}. {c['category']}")
            session["state"] = "waiting_for_category"
            return {"reply": "\n".join(lines), "next_state": session["state"], "done": False}

        # 2) User picks a category -> pick, get category questions, ask Q1
        if state == "waiting_for_category":
            sel = user_message.strip()
            top_categories = session.get("top_categories", [])
            chosen = None
            if sel.isdigit():
                try:
                    idx = int(sel) - 1
                    if 0 <= idx < len(top_categories):
                        chosen = top_categories[idx]['category']
                except:
                    chosen = None
            else:
                for c in top_categories:
                    if c['category'].lower() == sel.lower():
                        chosen = c['category']
                        break
            if not chosen:
                chosen = top_categories[0]['category']
            session["selected_category"] = chosen
            category_questions = self.get_category_questions(chosen)
            session["category_questions"] = category_questions
            q1 = self.question_generation(initial_query=session["initial_query"], question_list=category_questions, id=0)
            session["q1"] = q1
            session["state"] = "waiting_q1"
            return {"reply": q1, "next_state": session["state"], "done": False}

        # 3) Q1 answered -> generate Q2
        if state == "waiting_q1":
            answer1 = user_message.strip()
            session["answer1"] = answer1
            q2 = self.question_generation(initial_query=session["initial_query"], question_list=session.get("category_questions", []),
                                        id=1, prev_que_1=session.get("q1", ""), answer_1=answer1)
            session["q2"] = q2
            session["state"] = "waiting_q2"
            return {"reply": q2, "next_state": session["state"], "done": False}

        # 4) Q2 answered -> generate Q3
        if state == "waiting_q2":
            answer2 = user_message.strip()
            session["answer2"] = answer2
            q3 = self.question_generation(initial_query=session["initial_query"], question_list=session.get("category_questions", []),
                                        id=2, prev_que_1=session.get("q1", ""), answer_1=session.get("answer1", ""),
                                        prev_que_2=session.get("q2", ""), answer_2=answer2)
            session["q3"] = q3
            session["state"] = "waiting_q3"
            return {"reply": q3, "next_state": session["state"], "done": False}

        # 5) Q3 answered -> ask Q4 (static prompt)
        if state == "waiting_q3":
            answer3 = user_message.strip()
            session["answer3"] = answer3
            q4 = "Please provide some more context or details about the issue, in form of text or screenshots. (optional)"
            session["q4_prompt"] = q4
            session["state"] = "waiting_q4"
            return {"reply": q4, "next_state": session["state"], "done": False}

        # 6) Q4 answered -> finalize, store to mongo and create JIRA issue
        if state == "waiting_q4":
            answer4 = user_message.strip()
            session["answer4"] = answer4 if answer4.strip() != "" else "No additional details provided"

            json_data = {
                "session_id": self.key,
                "user_initial_query": session.get("initial_query"),
                "similarity_category": session.get("top_categories"),
                "user_selected_category": session.get("selected_category"),
                "predefined_category_questions": session.get("category_questions", [])[:3],
                "q_a": {
                    "Q1": {"question": session.get("q1"), "answer": session.get("answer1")},
                    "Q2": {"question": session.get("q2"), "answer": session.get("answer2")},
                    "Q3": {"question": session.get("q3"), "answer": session.get("answer3")},
                    "Q4": {"question": session.get("q4_prompt"), "answer": session.get("answer4")}
                },
                "upload_attachment": None,
                "predicted_category": session.get("predicted_category"),
                "status": f"Ticket raised for the issue: {session.get('initial_query')}",
                "issue_status": "unable_to_raise_ticket",
                "id": None,
                "key": None,
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "deleted_temp_data": None
            }

            conversation_summary = f"""
User_Selected_Category: {session.get('selected_category')}
Q1 -> {session.get('q1')}
A1 -> {session.get('answer1')}
Q2 -> {session.get('q2')}
A2 -> {session.get('answer2')}
Q3 -> {session.get('q3')}
A3 -> {session.get('answer3')}
Q4 -> {session.get('q4_prompt')}
A4 -> {session.get('answer4')}

predicted_category: {session.get('predicted_category')}
            """.strip()

            # Save to instance cluster (best-effort)
            try:
                await asyncio.to_thread(
                    self.add_to_mongo_instance,
                    session.get("initial_query"),
                    conversation_summary,
                    session_id=self.key
                )
            except Exception as e:
                logger.warning(f"Failed to save to instance cluster: {e}")

            final_reply = ""
            try:
                response = await asyncio.to_thread(self.jira_create_issue)
                if isinstance(response, dict) and response.get("status") == "error":
                    final_reply = f"⚠️ Could not create JIRA ticket: {response.get('message')}"
                    json_data["issue_status"] = "creation_failed"
                elif hasattr(response, 'status_code') and response.status_code == 201:
                    resp_data = response.json()
                    issue_key = resp_data.get("key")
                    json_data.update({
                        "issue_status": "ticket_raised",
                        "id": resp_data.get("id"),
                        "key": issue_key
                    })
                    # Clean up temp data
                    cleanup_result = self.clear_instance_cluster(session_id=self.key)
                    json_data["deleted_temp_data"] = cleanup_result.get("timestamp")
                    await asyncio.to_thread(self.add_to_mongo_global, json_data, self.key)
                    final_reply = f"✅ Ticket raised successfully! Your ticket ID is: **{issue_key}**.\nWe’ll get back to you shortly."
                else:
                    code = getattr(response, "status_code", "N/A")
                    text = getattr(response, "text", str(response))
                    final_reply = f"⚠️ JIRA responded with: {code} - {text[:200]}"
                    json_data["issue_status"] = "unexpected_response"
            except Exception as e:
                final_reply = f"❌ Error during JIRA ticket creation: {str(e)}"
                json_data["issue_status"] = "exception"

            # ALWAYS transition to final state after this point
            session["state"] = "final"
            closing_msg = (
                f"{final_reply}\n\n"
                "Your issue has been processed.\n"
                "You can start a new chat to raise another ticket."
            )
            logger.info(f"BOT: {closing_msg} | NEXT_STATE: final")
            return {"reply": final_reply, "next_state": "final", "done": True}