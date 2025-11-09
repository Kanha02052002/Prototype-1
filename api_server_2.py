import os
import json
import time
import asyncio
import datetime
from datetime import timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Query
from concurrent.futures import ThreadPoolExecutor
from src.logic import ChatLogic, get_session_logger

_executor = ThreadPoolExecutor(max_workers=15)
_sessions = {}
_sessions_lock = asyncio.Lock()
SESSION_TIMEOUT = timedelta(minutes=2)

GREETING = "Hello! 👋 I'm your IT Support Assistant. Please describe your issue."
GLOBAL_LOGGER = get_session_logger("global")


async def cleanup_stale_sessions():
    """Periodically remove sessions idle for more than SESSION_TIMEOUT."""
    while True:
        try:
            await asyncio.sleep(30)
            now = datetime.datetime.utcnow()

            async with _sessions_lock:
                stale = [
                    sid for sid, s in _sessions.items()
                    if (now - s.get("last_active", now)) > SESSION_TIMEOUT
                ]

                for sid in stale:
                    try:
                        GLOBAL_LOGGER.info(f"🧹 Cleaning stale session {sid}")
                        s = _sessions.pop(sid)
                        await asyncio.to_thread(s["logic"].clear_instance_cluster, sid)
                    except Exception as e:
                        GLOBAL_LOGGER.error(f"Cleanup error for {sid}: {e}", exc_info=True)
        except asyncio.CancelledError:
            GLOBAL_LOGGER.info("cleanup_stale_sessions task was cancelled")
            break
        except Exception as e:
            GLOBAL_LOGGER.error(f"Unexpected error in cleanup_stale_sessions: {e}", exc_info=True)
            await asyncio.sleep(30)  
            
async def cleanup_session_after_completion(session_id: str):
    """Gracefully cleanup after a session is finished."""
    await asyncio.sleep(10)
    async with _sessions_lock:
        if session_id in _sessions:
            del _sessions[session_id]
    GLOBAL_LOGGER.info(f"Session {session_id} cleaned up after completion.")

def check_readiness():
    """Check dataset, env vars, and model cache availability."""
    from dotenv import load_dotenv
    load_dotenv(".env")

    required_envs = [
        "QUERY_DATASET_PATH",
        "MONGODB_URI",
        "MONGODB_DATABASE",
    ]

    missing = [e for e in required_envs if not os.getenv(e)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}")

    dataset_path = os.getenv("QUERY_DATASET_PATH")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    os.makedirs("models_cache", exist_ok=True)
    GLOBAL_LOGGER.info("✅ Readiness check passed — environment and dataset OK.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    GLOBAL_LOGGER.info("🚀 Starting IT Support Chatbot API (async version)...")
    try:
        check_readiness()
    except Exception as e:
        GLOBAL_LOGGER.error(f"Startup readiness failed: {e}", exc_info=True)
        raise e

    cleanup_task = asyncio.create_task(cleanup_stale_sessions())
    GLOBAL_LOGGER.info("🧹 Background cleanup task started.")
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass  

        # On shutdown
        GLOBAL_LOGGER.info("🔻 Shutting down API...")
        async with _sessions_lock:
            for sid, s in list(_sessions.items()):
                try:
                    await asyncio.to_thread(s["logic"].clear_instance_cluster, sid)
                except Exception:
                    pass
            _sessions.clear()
        GLOBAL_LOGGER.info("✅ Shutdown cleanup complete.")

app = FastAPI(title="IT Support Chatbot API", lifespan=lifespan)

@app.post("/chat/start")
async def start_chat():
    logic = ChatLogic()
    sid = logic.key
    session = {
        "logic": logic,
        "state": "waiting_for_query",
        "history": [{"role": "bot", "content": GREETING}],
        "data": {},
        "last_active": datetime.datetime.utcnow()
    }

    async with _sessions_lock:
        _sessions[sid] = session

    GLOBAL_LOGGER.info(f"🆕 New session started: {sid}")
    return {"session_id": sid, "greeting": GREETING}

@app.post("/chat/message")
async def chat_message(
    session_id: str = Query(..., description="Session ID"),
    user_message: str = Query(..., description="User message content")
):
    async with _sessions_lock:
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail=f"Invalid session_id: {session_id}")
        session = _sessions[session_id]

    logic: ChatLogic = session["logic"]

    try:
        GLOBAL_LOGGER.info(f"[{session_id}] USER: {user_message}")
        result = await logic.handle_message(session["data"], user_message)
    except Exception as e:
        GLOBAL_LOGGER.error(f"❌ Error processing message for {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing message: {e}")

    reply = result.get("reply", "Sorry, something went wrong.")
    next_state = result.get("next_state", session["state"])
    done = result.get("done", False)

    async with _sessions_lock:
        session["state"] = next_state
        session["last_active"] = datetime.datetime.utcnow()
        session["history"].append({"role": "user", "content": user_message})
        session["history"].append({"role": "bot", "content": reply})

    GLOBAL_LOGGER.info(f"[{session_id}] BOT: {reply}")

    if done or next_state == "final":
        GLOBAL_LOGGER.info(f"[{session_id}] ✅ Chat completed.")
        asyncio.create_task(cleanup_session_after_completion(session_id))

    return {
        "session_id": session_id,
        "reply": reply,
        "next_state": next_state,
        "done": done
    }

@app.get("/chat/history/{session_id}")
async def get_history(session_id: str):
    async with _sessions_lock:
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail="Invalid session_id")
        session = _sessions[session_id]
    return {
        "session_id": session_id,
        "state": session["state"],
        "history": session["history"]
    }

@app.get("/")
async def health():
    try:
        check_readiness()
        return {"status": "ok", "message": "🚀 IT Support Chatbot API is fully operational."}
    except Exception as e:
        return {"status": "error", "message": str(e)}