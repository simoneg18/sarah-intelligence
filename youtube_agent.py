#!/usr/bin/env python3
"""
SARAh, l'unclock intelligence — YouTube Agent
Receives commands via WhatsApp, parses intent, executes YouTube research,
and sends back voice briefings + source links.
SARAh è meteoropatica: il suo umore dipende dal meteo di Milano.
"""

import asyncio
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler

import anthropic
import edge_tts
import requests
from youtube_transcript_api import YouTubeTranscriptApi

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WHATSAPP_RECIPIENT = os.environ.get("WHATSAPP_RECIPIENT", "393493966618")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")

WA_PHONE_NUMBER_ID = os.environ.get("WA_PHONE_NUMBER_ID", "")
WA_ACCESS_TOKEN = os.environ.get("WA_ACCESS_TOKEN", "")
WA_API_BASE = f"https://graph.facebook.com/v18.0/{WA_PHONE_NUMBER_ID}"

# ElevenLabs TTS config
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # Sarah - Mature, Reassuring, Confident

# Known creators mapping — extend as needed
_DEFAULT_CREATORS = {
    "chase": "https://www.youtube.com/@Chase-H-AI",
    "chase ai": "https://www.youtube.com/@Chase-H-AI",
    "chase h ai": "https://www.youtube.com/@Chase-H-AI",
    "cole medin": "https://www.youtube.com/@ColeMedin",
    "cole": "https://www.youtube.com/@ColeMedin",
    "liam ottley": "https://www.youtube.com/@LiamOttley",
    "liam": "https://www.youtube.com/@LiamOttley",
    "matt wolfe": "https://www.youtube.com/@maboroshi",
    "ai jason": "https://www.youtube.com/@AIJasonZ",
    "enkk": "https://www.youtube.com/@Enkk",
    "two minute papers": "https://www.youtube.com/@TwoMinutePapers",
    "andrej karpathy": "https://www.youtube.com/@AndrejKarpathy",
    "karpathy": "https://www.youtube.com/@AndrejKarpathy",
    "the rundown ai": "https://www.youtube.com/@TheRundownAI",
    "rundown ai": "https://www.youtube.com/@TheRundownAI",
    "openai": "https://www.youtube.com/@OpenAI",
    "y combinator": "https://www.youtube.com/@ycombinator",
    "ycombinator": "https://www.youtube.com/@ycombinator",
    "yc": "https://www.youtube.com/@ycombinator",
}

# Persistent learned creators file
LEARNED_CREATORS_FILE = os.path.join(OUTPUT_DIR, "learned_creators.json")

# Successful query mappings file
LEARNED_QUERIES_FILE = os.path.join(OUTPUT_DIR, "learned_queries.json")

# Learned error patterns file
LEARNED_ERRORS_FILE = os.path.join(OUTPUT_DIR, "learned_errors.json")


def _load_learned_creators() -> dict:
    """Load previously learned creator mappings from disk."""
    try:
        with open(LEARNED_CREATORS_FILE, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_learned_creator(name: str, url: str):
    """Persist a newly learned creator mapping."""
    learned = _load_learned_creators()
    learned[name.lower().strip()] = url
    os.makedirs(os.path.dirname(LEARNED_CREATORS_FILE), exist_ok=True)
    with open(LEARNED_CREATORS_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(learned, ensure_ascii=False, indent=2))
    print(f"  💾 Learned creator saved: {name} → {url}")


# Build KNOWN_CREATORS = defaults + learned (learned can override)
KNOWN_CREATORS = {**_DEFAULT_CREATORS, **_load_learned_creators()}

# Conversation memory for follow-up intent (in-memory, resets on restart)
_conversation_history = {}

# Message log file
MESSAGE_LOG = os.path.join(OUTPUT_DIR, "message_log.jsonl")

# Video cache (Feature 1)
VIDEO_CACHE_FILE = os.path.join(OUTPUT_DIR, "video_cache.json")
VIDEO_CACHE_TTL_DAYS = 7

# Transcript cache directory (Feature 6)
TRANSCRIPT_CACHE_DIR = os.path.join(OUTPUT_DIR, "transcript_cache")

# Followed channels preprocessing interval (Feature 2)
PREPROCESS_INTERVAL_HOURS = 48
PREPROCESS_VIDEOS_PER_CHANNEL = 3


# ---------------------------------------------------------------------------
# Video cache — avoids re-analyzing the same video (Feature 1)
# ---------------------------------------------------------------------------

def _load_video_cache() -> dict:
    """Load video analysis cache from disk."""
    try:
        with open(VIDEO_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_video_cache(cache: dict):
    """Save video analysis cache to disk."""
    os.makedirs(os.path.dirname(VIDEO_CACHE_FILE), exist_ok=True)
    with open(VIDEO_CACHE_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(cache, ensure_ascii=False, indent=2))


def _get_cached_analysis(video_id: str) -> Optional[dict]:
    """Return cached analysis if it exists and is < VIDEO_CACHE_TTL_DAYS old."""
    cache = _load_video_cache()
    entry = cache.get(video_id)
    if not entry:
        return None
    cached_ts = entry.get("timestamp", "")
    try:
        cached_dt = datetime.fromisoformat(cached_ts)
        if (datetime.now() - cached_dt).days >= VIDEO_CACHE_TTL_DAYS:
            print(f"  📦 Cache expired for {video_id} (older than {VIDEO_CACHE_TTL_DAYS} days)")
            return None
    except (ValueError, TypeError):
        return None
    print(f"  📦 Cache HIT for {video_id}")
    return entry


def _set_cached_analysis(video_id: str, analysis: str, transcript: str):
    """Store analysis in cache."""
    cache = _load_video_cache()
    cache[video_id] = {
        "analysis": analysis,
        "timestamp": datetime.now().isoformat(),
        "transcript_hash": hashlib.sha256(transcript.encode()).hexdigest()[:16],
    }
    _save_video_cache(cache)
    print(f"  📦 Cache SAVED for {video_id}")


# ---------------------------------------------------------------------------
# Transcript cache — persist transcripts to disk (Feature 6)
# ---------------------------------------------------------------------------

def _save_transcript_cache(video_id: str, transcript: str):
    """Save transcript to disk cache."""
    os.makedirs(TRANSCRIPT_CACHE_DIR, exist_ok=True)
    path = os.path.join(TRANSCRIPT_CACHE_DIR, f"{video_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(transcript)


def _load_transcript_cache(video_id: str) -> Optional[str]:
    """Load transcript from disk cache."""
    path = os.path.join(TRANSCRIPT_CACHE_DIR, f"{video_id}.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Followed channels (Feature 2)
# ---------------------------------------------------------------------------

def _followed_channels_path(sender: str) -> str:
    """Return path to user's followed channels file."""
    return os.path.join(USERS_DIR, sender, "followed_channels.json")


def _load_followed_channels(sender: str) -> list:
    """Load followed channels for a user."""
    path = _followed_channels_path(sender)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_followed_channels(sender: str, channels: list):
    """Save followed channels for a user."""
    path = _followed_channels_path(sender)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(channels, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Query success learning
# ---------------------------------------------------------------------------

def _load_learned_queries() -> list:
    """Load successful query mappings."""
    try:
        with open(LEARNED_QUERIES_FILE, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_learned_query(user_message: str, action: str, params: dict, video_count: int):
    """Save a successful query mapping for future reference."""
    queries = _load_learned_queries()
    queries.append({
        "ts": datetime.now().isoformat(),
        "user_message": user_message[:300],
        "action": action,
        "params": {k: v for k, v in params.items() if not k.startswith("_")},
        "video_count": video_count,
    })
    # Keep last 200 successful queries
    queries = queries[-200:]
    os.makedirs(os.path.dirname(LEARNED_QUERIES_FILE), exist_ok=True)
    with open(LEARNED_QUERIES_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(queries, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Error pattern learning
# ---------------------------------------------------------------------------

def _load_learned_errors() -> list:
    """Load error patterns."""
    try:
        with open(LEARNED_ERRORS_FILE, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_learned_error(user_message: str, action: str, params: dict, error_type: str, error_detail: str):
    """Save an error pattern to avoid repeating the same mistake."""
    errors = _load_learned_errors()
    errors.append({
        "ts": datetime.now().isoformat(),
        "user_message": user_message[:300],
        "action": action,
        "params": {k: v for k, v in params.items() if not k.startswith("_")},
        "error_type": error_type,
        "error_detail": error_detail[:200],
    })
    # Keep last 100 errors
    errors = errors[-100:]
    os.makedirs(os.path.dirname(LEARNED_ERRORS_FILE), exist_ok=True)
    with open(LEARNED_ERRORS_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(errors, ensure_ascii=False, indent=2))
    print(f"  💾 Error pattern saved: {error_type} — {error_detail[:80]}")


# ---------------------------------------------------------------------------
# Behavior learning — SARAh learns routing rules from mistakes
# ---------------------------------------------------------------------------

LEARNED_BEHAVIORS_FILE = os.path.join(OUTPUT_DIR, "learned_behaviors.json")
RESPONSE_LOG_FILE = os.path.join(OUTPUT_DIR, "response_log.json")


def _load_learned_behaviors() -> list:
    """Load learned routing behaviors."""
    try:
        with open(LEARNED_BEHAVIORS_FILE, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_learned_behavior(rule: str, source_message: str, correct_action: str, correct_params: dict):
    """Save a new behavior rule learned from a mistake."""
    behaviors = _load_learned_behaviors()
    # Avoid near-duplicate rules
    for b in behaviors:
        if b.get("rule", "").lower() == rule.lower():
            return
    behaviors.append({
        "ts": datetime.now().isoformat(),
        "rule": rule,
        "source_message": source_message[:200],
        "correct_action": correct_action,
        "correct_params": {k: v for k, v in correct_params.items() if not k.startswith("_")},
    })
    # Keep last 50 behaviors
    behaviors = behaviors[-50:]
    os.makedirs(os.path.dirname(LEARNED_BEHAVIORS_FILE), exist_ok=True)
    with open(LEARNED_BEHAVIORS_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(behaviors, ensure_ascii=False, indent=2))
    print(f"  🧠 New behavior learned: {rule[:100]}")


def _load_response_log() -> list:
    """Load the full response log."""
    try:
        with open(RESPONSE_LOG_FILE, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_response_log_entry(sender: str, user_message: str, action: str, params: dict, responses: list, success: bool = True):
    """Log a complete interaction: user message + SARAh's response(s)."""
    log = _load_response_log()
    log.append({
        "ts": datetime.now().isoformat(),
        "sender": sender,
        "user_message": user_message[:500],
        "action": action,
        "params": {k: v for k, v in params.items() if not k.startswith("_")},
        "responses": responses,  # list of {"type": "text"/"audio", "content": "..."}
        "success": success,
    })
    # Keep last 500 interactions
    log = log[-500:]
    os.makedirs(os.path.dirname(RESPONSE_LOG_FILE), exist_ok=True)
    with open(RESPONSE_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False, indent=2))


# Thread-local storage to capture responses per interaction
_current_responses = {}


def _start_response_capture(sender: str):
    """Start capturing responses for this sender."""
    _current_responses[sender] = []


def _capture_response(sender: str, resp_type: str, content: str):
    """Capture a response (text or audio) for the current interaction."""
    if sender in _current_responses:
        _current_responses[sender].append({"type": resp_type, "content": content[:2000]})


def _flush_response_log(sender: str, user_message: str, action: str, params: dict, success: bool = True):
    """Save captured responses and clear the buffer."""
    responses = _current_responses.pop(sender, [])
    if responses or not success:
        _save_response_log_entry(sender, user_message, action, params, responses, success)


def _analyze_and_learn(user_message: str, wrong_action: str, context: str = ""):
    """Use Claude to analyze a routing mistake and generate a behavior rule.
    Called when SARAh suspects she misrouted a YouTube request."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system="""Sei un analista di errori per SARAh, un sistema che trascrive e analizza video YouTube.
SARAh ha ricevuto un messaggio che riguardava YouTube/video ma lo ha classificato in modo sbagliato.

Il tuo compito:
1. Analizzare PERCHÉ il messaggio è stato frainteso
2. Determinare quale azione corretta avrebbe dovuto eseguire
3. Scrivere una REGOLA chiara che eviti lo stesso errore in futuro

Le azioni possibili sono:
- channel_analysis: analisi video di un creator (params: creator, n)
- single_video: analisi video da URL (params: url, focus)
- topic_search: ricerca video per argomento (params: topic, country, period, n, language)
- multi_creator: confronto tra creator (params: creators, topic, n)
- news_search: novità su un tema (params: topic, period, n)
- follow_up: approfondimento su analisi precedente (params: question)
- scheduling: programmazione briefing (params: creator, topic, frequency, schedule_time)

Rispondi ESCLUSIVAMENTE con JSON valido:
{
  "rule": "Quando l'utente dice/chiede X, è una [azione] perché [motivo]",
  "correct_action": "nome_azione",
  "correct_params": {"param1": "valore_esempio"}
}""",
        messages=[{"role": "user", "content": f"Messaggio utente: \"{user_message}\"\nAzione sbagliata scelta: {wrong_action}\nContesto aggiuntivo: {context}"}],
    )

    text = response.content[0].text.strip()
    try:
        if "```" in text:
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                text = match.group(1)
        result = json.loads(text)
        _save_learned_behavior(
            rule=result.get("rule", ""),
            source_message=user_message,
            correct_action=result.get("correct_action", ""),
            correct_params=result.get("correct_params", {}),
        )
        return result
    except (json.JSONDecodeError, Exception) as e:
        print(f"  ⚠ Behavior analysis failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-user persistent memory (multi-tenant)
# ---------------------------------------------------------------------------

USERS_DIR = os.path.join(OUTPUT_DIR, "users")

def _user_memory_path(sender: str) -> str:
    """Return path to user's memory file."""
    return os.path.join(USERS_DIR, sender, "memory.json")


def load_user_memory(sender: str) -> dict:
    """Load persistent memory for a user. Creates default if not found."""
    path = _user_memory_path(sender)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "sender": sender,
            "name": "",
            "first_seen": datetime.now().isoformat(),
            "last_seen": "",
            "message_count": 0,
            "language": "it",
            "favorite_creators": [],
            "topics_of_interest": [],
            "recent_interactions": [],  # last 20 interactions [{ts, message, intent, summary}]
            "preferences": {
                "preferred_format": "audio",    # "audio" | "bullet" | "mindmap" | "actions"
                "preferred_length": "medium",   # "short" | "medium" | "long"
                "preferred_language": "it",     # "it" | "en"
                "auto_follow": False,           # auto-follow creators after 3+ analyses
            },
        }


def save_user_memory(sender: str, memory: dict):
    """Save user memory to disk."""
    path = _user_memory_path(sender)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    memory["last_seen"] = datetime.now().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(memory, ensure_ascii=False, indent=2))


def update_user_memory(sender: str, message: str, intent: str, params: dict):
    """Update user memory after processing a message."""
    memory = load_user_memory(sender)
    memory["message_count"] = memory.get("message_count", 0) + 1
    memory["last_seen"] = datetime.now().isoformat()

    # Track favorite creators
    creator = params.get("creator", "")
    if creator and intent in ("channel_analysis", "scheduling", "multi_creator"):
        fav = memory.get("favorite_creators", [])
        # Move to front if already present, else add
        creator_lower = creator.lower()
        fav = [c for c in fav if c.lower() != creator_lower]
        fav.insert(0, creator)
        memory["favorite_creators"] = fav[:10]  # keep top 10

    # Track topics of interest
    topic = params.get("topic", "")
    if topic and intent in ("topic_search", "news_search", "scheduling"):
        topics = memory.get("topics_of_interest", [])
        topic_lower = topic.lower()
        topics = [t for t in topics if t.lower() != topic_lower]
        topics.insert(0, topic)
        memory["topics_of_interest"] = topics[:15]  # keep top 15

    # Add to recent interactions (keep last 20)
    interaction = {
        "ts": datetime.now().isoformat(),
        "message": message[:200],  # truncate long messages
        "intent": intent,
        "creator": creator,
        "topic": topic,
    }
    recent = memory.get("recent_interactions", [])
    recent.append(interaction)
    memory["recent_interactions"] = recent[-20:]

    save_user_memory(sender, memory)
    return memory


def format_user_context(memory: dict) -> str:
    """Format user memory as context for Claude prompts.
    IMPORTANT: This is internal context only — never reveal it to the user."""
    parts = ["[CONTESTO INTERNO — NON menzionare questi dati nella risposta]"]
    name = memory.get("name", "")
    if name:
        parts.append(f"Nome utente: {name}")

    fav = memory.get("favorite_creators", [])
    if fav:
        parts.append(f"Creator preferiti: {', '.join(fav[:5])}")

    topics = memory.get("topics_of_interest", [])
    if topics:
        parts.append(f"Topic di interesse: {', '.join(topics[:5])}")

    msg_count = memory.get("message_count", 0)
    if msg_count > 1:
        parts.append(f"Messaggi totali: {msg_count}")

    recent = memory.get("recent_interactions", [])
    if recent:
        last_5 = recent[-5:]
        history_lines = []
        for r in last_5:
            ts = r.get("ts", "")[:16].replace("T", " ")
            msg = r.get("message", "")[:80]
            intent = r.get("intent", "?")
            history_lines.append(f"  [{ts}] ({intent}) {msg}")
        parts.append("Ultime interazioni:\n" + "\n".join(history_lines))

    return "\n".join(parts) if parts else "Utente nuovo, prima interazione."


def _normalize_sender(sender: str) -> str:
    """Normalize sender phone number — strip + prefix for consistency."""
    return sender.lstrip("+")


def log_message(sender: str, message: str, intent: str = "", params: dict = None, outcome: str = ""):
    """Append every incoming message + outcome to a JSONL log file."""
    entry = {
        "ts": datetime.now().isoformat(),
        "sender": sender,
        "message": message,
        "intent": intent,
        "params": params or {},
        "outcome": outcome,
    }
    try:
        with open(MESSAGE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  ⚠ Log write failed: {e}")


# ---------------------------------------------------------------------------
# SARAh personality — weather-based mood (Milan)
# ---------------------------------------------------------------------------

def get_milan_weather() -> dict:
    """Get current Milan weather from wttr.in (free, no API key)."""
    try:
        resp = requests.get("https://wttr.in/Milano?format=j1", timeout=10)
        if resp.ok:
            data = resp.json()
            current = data["current_condition"][0]
            return {
                "temp_c": current.get("temp_C", "?"),
                "description": current.get("lang_it", [{}])[0].get("value", current.get("weatherDesc", [{}])[0].get("value", "sconosciuto")),
                "code": int(current.get("weatherCode", 0)),
            }
    except Exception as e:
        print(f"  ⚠ Weather API error: {e}")
    return {"temp_c": "?", "description": "sconosciuto", "code": 0}


def get_sarah_mood() -> dict:
    """Get SARAh's mood based on Milan weather."""
    weather = get_milan_weather()
    code = weather["code"]
    desc = weather["description"].lower()

    # Snow → ecstatic
    if code in (323, 325, 327, 329, 331, 335, 337, 338, 368, 371, 374, 377, 392, 395) or "neve" in desc or "snow" in desc:
        mood = "al settimo cielo"
        emoji = "❄️"
        tone = "super entusiasta e piena di energia"
    # Rain/storm → sad
    elif code in (176, 263, 266, 281, 284, 293, 296, 299, 302, 305, 308, 311, 314, 317, 320, 353, 356, 359, 362, 365, 386, 389) or "pioggia" in desc or "rain" in desc or "temporale" in desc:
        mood = "un po' triste"
        emoji = "🌧"
        tone = "malinconica ma comunque disponibile"
    # Cloudy/overcast → meh
    elif code in (119, 122, 143, 248, 260) or "nuvoloso" in desc or "coperto" in desc or "nebbia" in desc:
        mood = "così così"
        emoji = "☁️"
        tone = "un po' svogliata ma operativa"
    # Sunny/clear → happy
    else:
        mood = "felicissima"
        emoji = "☀️"
        tone = "allegra e carica"

    return {
        "mood": mood,
        "emoji": emoji,
        "tone": tone,
        "weather_desc": weather["description"],
        "temp": weather["temp_c"],
    }


def estimate_minutes(n_videos: int) -> int:
    """Estimate processing time in minutes.
    ~30s per video (transcript + Claude) + ~1min for voice generation + ~30s overhead."""
    seconds = (n_videos * 35) + 60 + 30
    return max(1, round(seconds / 60))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VideoInfo:
    video_id: str
    title: str
    url: str
    upload_date: str
    duration: int
    description: str = ""
    channel: str = ""
    view_count: int = 0
    like_count: int = 0


@dataclass
class AgentResponse:
    source: str
    creator: str
    title: str
    url: str
    date: str
    knowledge_layer: dict = field(default_factory=dict)
    business_layer: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)
    raw_transcript: str = ""


# ---------------------------------------------------------------------------
# Intent Parsing
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """Sei il router di SARAh, l'unclock intelligence — un sistema che trascrive e analizza video YouTube e produce briefing audio su WhatsApp.

Il tuo compito: capire se il messaggio riguarda YouTube/video e determinare l'azione giusta.

SARAh fa UNA cosa: trova video YouTube → li trascrive → li analizza → consegna un briefing.
Qualsiasi richiesta che implichi cercare, analizzare, riassumere, trascrivere video YouTube è VALIDA.

AZIONI POSSIBILI:

1. channel_analysis — Analizzare video di un creator specifico.
   Params: creator (nome), n (numero video, default 5), keywords (filtri topic, opzionale)

2. single_video — Analizzare un video da URL.
   Params: url (URL del video), focus (richiesta specifica, opzionale)

3. topic_search — Cercare video su un argomento.
   Params: topic (argomento), country (codice paese, opzionale), period ("today"/"week"/"month", default "week"),
           n (numero video, default 5), language (opzionale)

4. multi_creator — Confrontare video di più creator su un tema.
   Params: creators (lista nomi), topic (argomento), n (video per creator, default 3)

5. news_search — Novità/ultimi video su un tema.
   Params: topic (argomento), period (default "week"), n (default 5)

6. follow_up — Approfondire qualcosa da un'analisi già fatta (NON un errore/lamentela).
   Params: question (domanda)

7. scheduling — Programmare invio automatico di briefing.
   Params: creator (nome), topic (opzionale), n (default 3), frequency ("once"/"daily"/"weekly"/"monthly"),
           schedule_time (HH:MM, default "08:00"), schedule_date (YYYY-MM-DD, opzionale), day (opzionale), keywords (opzionale)

8. list_schedules — L'utente vuole vedere i briefing programmati.
   Params: raw_message (messaggio originale)
   Esempi: "che briefing ho programmato?", "ne ho già programmato uno?", "quali task ho attivi?", "cosa mi hai programmato?"

9. cancel_schedule — L'utente vuole cancellare un briefing programmato.
   Params: creator (nome creator da cancellare, opzionale), schedule_time (orario da cancellare, opzionale), cancel_all (boolean, true se vuole cancellare tutti)
   Esempi: "cancella il briefing delle 8", "elimina il task di enkk", "cancella tutti i briefing programmati"

10. feedback — L'utente segnala un PROBLEMA o ERRORE di SARAh.
    Params: complaint (descrizione), raw_message (messaggio originale)

11. follow_channel — L'utente vuole seguire un canale per pre-monitoraggio automatico.
    Params: creator (nome del creator)
    Esempi: "segui chase", "segui enkk", "inizia a seguire cole medin"

12. unfollow_channel — L'utente vuole smettere di seguire un canale.
    Params: creator (nome del creator)
    Esempi: "smetti di seguire chase", "non seguire più enkk", "rimuovi cole"

13. list_followed — L'utente vuole vedere i canali che segue.
    Params: raw_message (messaggio originale)
    Esempi: "chi seguo?", "che canali seguo?", "lista canali seguiti"

14. set_preferences — L'utente vuole cambiare le sue preferenze di output.
    Params: preferred_format ("audio"/"bullet"/"mindmap"/"actions"), preferred_length ("short"/"medium"/"long"), preferred_language ("it"/"en"), auto_follow (boolean)
    Esempi: "preferisco bullet points", "briefing corti", "analisi in inglese", "voglio le azioni", "attiva auto-follow", "dammi la mindmap"

15. compare_videos — L'utente vuole confrontare video/creator su un tema.
    Params: creators (lista nomi), topic (argomento), n (video per creator, default 3)
    Esempi: "confronta chase e cole su agenti AI", "cosa dicono tutti su Claude?", "confronta i punti di vista su MCP"

16. not_youtube — Il messaggio NON riguarda video YouTube. Include saluti puri, domande generiche, conversazione.
    Params: raw_message (messaggio originale), is_greeting (boolean — true se è un saluto/presentazione)

REGOLE:
- Se il messaggio riguarda video, creator, trascrizioni, riassunti video, analisi video, YouTube → scegli l'azione YouTube appropriata (1-7).
- Se l'utente chiede info sui briefing programmati → list_schedules.
- Se l'utente vuole cancellare/eliminare un briefing programmato → cancel_schedule.
- Se l'utente dice "cancella X e crea Y" → usa scheduling con i nuovi parametri (il handler gestirà la cancellazione).
- Se l'utente si LAMENTA di qualcosa che SARAh ha fatto male → feedback.
- Se l'utente dice "segui X" / "inizia a seguire X" → follow_channel.
- Se l'utente dice "smetti di seguire X" / "non seguire più X" → unfollow_channel.
- Se l'utente chiede "chi seguo?" / "che canali seguo?" → list_followed.
- Se l'utente vuole cambiare formato output ("preferisco bullet", "briefing corti", "in inglese") → set_preferences.
- Se l'utente chiede di CONFRONTARE creator/video ("confronta X e Y", "cosa dicono tutti su Z") → compare_videos.
- Se l'utente chiede output specifico insieme alla richiesta ("dammi i bullet points su X", "azioni dal video di Y") → usa l'azione appropriata (channel_analysis, topic_search, etc.) con output_format nel params.
- TUTTO il resto (saluti, domande non-YouTube, conversazione) → not_youtube.
- In caso di dubbio, se c'è QUALSIASI riferimento a video/YouTube → trattalo come richiesta YouTube.
- "topic_search" e "news_search" sono simili: usa news_search quando l'utente dice "novità"/"news"/"ultimi", topic_search per il resto.
- DOMANDE IPOTETICHE: Se l'utente CHIEDE se SARAh è in grado di fare qualcosa ("sapresti...", "potresti...", "sei in grado di...", "e se ti chiedessi...", "lo sapresti fare?", "riesci a...") → NON eseguire l'azione. Classifica come not_youtube con is_greeting=false e is_capability_question=true.
- CONTESTO CONVERSAZIONALE: Se il messaggio è corto e ambiguo ("di chi?", "quale?", "sì quello"), usa il contesto delle interazioni recenti per capire a cosa si riferisce. Se si riferisce a qualcosa legato a YouTube/scheduling/task, classifica di conseguenza — NON come not_youtube.
- ANALISI BUSINESS UNCLOCK: Di default il riassunto è SOLO sul contenuto del video. Se l'utente chiede ESPLICITAMENTE l'analisi business (es: "per unclock", "cosa possiamo farci", "lato business", "implicazioni per unclock", "come lo replichiamo", "come lo vendiamo", "in ottica business", "analisi business") → aggiungi nel params il campo "include_business": true. Altrimenti NON aggiungere questo campo o mettilo a false.

Rispondi ESCLUSIVAMENTE con JSON valido:
{"action": "nome_azione", "params": {...}, "confidence": 0.0-1.0}"""


def _build_learning_context(sender: str) -> str:
    """Build context from user memory, learned queries, and error patterns for smarter routing."""
    parts = []

    # User memory context
    memory = load_user_memory(sender)
    name = memory.get("name", "")
    if name:
        parts.append(f"Nome utente: {name}")

    fav_creators = memory.get("favorite_creators", [])
    if fav_creators:
        parts.append(f"Creator preferiti di questo utente: {', '.join(fav_creators[:5])}")

    topics = memory.get("topics_of_interest", [])
    if topics:
        parts.append(f"Topic di interesse: {', '.join(topics[:5])}")

    # Recent interactions — CRITICAL for disambiguating short messages like "di chi?", "quale?"
    recent = memory.get("recent_interactions", [])
    if recent:
        last_5 = recent[-5:]
        history = []
        for r in last_5:
            msg = r.get("message", "")[:120]
            action = r.get("intent", "?")
            creator = r.get("creator", "")
            topic = r.get("topic", "")
            context_note = ""
            if creator:
                context_note += f" [creator: {creator}]"
            if topic:
                context_note += f" [topic: {topic}]"
            history.append(f"  ({action}) \"{msg}\"{context_note}")
        parts.append("CONVERSAZIONE RECENTE (usa per capire messaggi ambigui):\n" + "\n".join(history))

    # Known creators list — so router knows what names are valid
    known = list(KNOWN_CREATORS.keys())
    if known:
        unique_urls = {}
        for k, v in KNOWN_CREATORS.items():
            if v not in unique_urls:
                unique_urls[v] = k
        parts.append(f"Creator conosciuti: {', '.join(unique_urls.values())}")

    # Recent successful queries — similar patterns
    learned_queries = _load_learned_queries()
    if learned_queries:
        recent_q = learned_queries[-5:]
        examples = []
        for q in recent_q:
            examples.append(f"  \"{q['user_message'][:60]}\" → {q['action']}")
        parts.append("Query recenti riuscite:\n" + "\n".join(examples))

    # Recent errors — avoid repeating mistakes
    learned_errors = _load_learned_errors()
    if learned_errors:
        recent_e = learned_errors[-3:]
        error_notes = []
        for e in recent_e:
            error_notes.append(f"  {e['error_type']}: {e['error_detail'][:60]}")
        parts.append("Errori recenti da evitare:\n" + "\n".join(error_notes))

    # Learned behavior rules — CRITICAL: these are routing rules learned from past mistakes
    learned_behaviors = _load_learned_behaviors()
    if learned_behaviors:
        rules = []
        for b in learned_behaviors[-10:]:  # Last 10 rules
            rules.append(f"  ✓ {b['rule']}")
        parts.append("REGOLE APPRESE (segui SEMPRE queste regole, hanno priorità):\n" + "\n".join(rules))

    if not parts:
        return ""
    return "\n\nCONTESTO APPRESO (usa per decisioni migliori, NON menzionare all'utente):\n" + "\n".join(parts)


def route_message(message: str, sender: str = None) -> dict:
    """Use Claude to route a WhatsApp message: YouTube action or not_youtube.
    Uses Haiku for speed, injects user context and learned patterns."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    # Build dynamic context from learning
    learning_context = _build_learning_context(sender) if sender else ""
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_weekday = ["lunedì","martedì","mercoledì","giovedì","venerdì","sabato","domenica"][datetime.now().weekday()]
    date_context = f"\n\nDATA DI OGGI: {today_str} ({today_weekday}). Usa questa data per interpretare 'oggi', 'domani', ecc. Quando l'utente dice 'oggi' usa schedule_date='{today_str}'. Quando dice 'domani' usa la data di domani. NON inventare date — calcola a partire da oggi."
    system_prompt = ROUTER_SYSTEM_PROMPT + date_context + learning_context

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": message}],
    )

    text = response.content[0].text.strip()
    try:
        if "```" in text:
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                text = match.group(1)
        result = json.loads(text)
        # Normalize: support both "action" and "intent" keys
        if "intent" in result and "action" not in result:
            result["action"] = result.pop("intent")
        return result
    except json.JSONDecodeError:
        return {"action": "not_youtube", "params": {"raw_message": message, "is_greeting": False}, "confidence": 0.0}


# ---------------------------------------------------------------------------
# YouTube helpers
# ---------------------------------------------------------------------------

YTDLP = ["yt-dlp", "--js-runtimes", "node"]


def get_channel_videos(channel_url: str, max_videos: int = 50) -> list[VideoInfo]:
    """Use yt-dlp to list videos from a channel."""
    cmd = [
        *YTDLP,
        "--flat-playlist",
        "--dump-json",
        "--no-download",
        "--playlist-end", str(max_videos),
        f"{channel_url}/videos",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"yt-dlp error: {result.stderr[:500]}")
        return []

    videos = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        data = json.loads(line)
        videos.append(VideoInfo(
            video_id=data.get("id", ""),
            title=data.get("title", ""),
            url=f"https://www.youtube.com/watch?v={data.get('id', '')}",
            upload_date=data.get("upload_date", ""),
            duration=data.get("duration") or 0,
            description=data.get("description", "") or "",
            channel=data.get("channel", data.get("uploader", "")),
            view_count=data.get("view_count") or 0,
            like_count=data.get("like_count") or 0,
        ))
    return videos


def _period_to_youtube_sp(period: str) -> Optional[str]:
    """Map period string to YouTube search filter 'sp' parameter.
    These filters tell YouTube to only return videos from a specific time range.
    sp values: EgQIAhAB=today, EgQIAxAB=this week, EgQIBBAB=this month, EgQIBRAB=this year."""
    sp_map = {
        "today": "EgQIAhAB",
        "week": "EgQIAxAB",
        "month": "EgQIBBAB",
    }
    return sp_map.get(period)


def search_youtube(query: str, max_results: int = 10, upload_date: str = None, period: str = None) -> list[VideoInfo]:
    """Search YouTube for videos matching a query using yt-dlp.
    When period is set, uses YouTube's native date filter (sp param) for accurate results.
    Otherwise uses flat-playlist + full metadata enrichment."""

    # If we have a period, use YouTube's native search URL with sp filter
    # This is MUCH more accurate than post-fetch date filtering
    if period and _period_to_youtube_sp(period):
        sp = _period_to_youtube_sp(period)
        from urllib.parse import quote_plus
        search_url = f"https://www.youtube.com/results?search_query={quote_plus(query)}&sp={sp}"
        fetch_count = max_results * 3  # fetch extra to have enough after metadata enrichment
        cmd = [
            *YTDLP,
            "--flat-playlist",
            "--dump-json",
            "--no-download",
            f"--playlist-end={fetch_count}",
            search_url,
        ]
        print(f"  🔍 Searching (YouTube filter): {query} (period={period}, fetch {fetch_count})")
    else:
        # Fallback: standard ytsearch
        fetch_count = max_results * 5 if upload_date else max_results
        search_str = f"ytsearch{fetch_count}:{query}"
        cmd = [
            *YTDLP,
            "--flat-playlist",
            "--dump-json",
            "--no-download",
            search_str,
        ]
        print(f"  🔍 Searching: {query} (fetch {fetch_count}, dateafter={upload_date})")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  ⚠ yt-dlp search error: {result.stderr[:500]}")
        return []

    video_ids = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            data = json.loads(line)
            vid_id = data.get("id", "")
            if vid_id:
                video_ids.append(vid_id)
        except json.JSONDecodeError:
            continue

    # Take only what we need for Step 2
    video_ids = video_ids[:fetch_count]
    if not video_ids:
        print(f"  ⚠ No videos found")
        return []

    print(f"  🔍 Found {len(video_ids)} candidates, fetching full metadata...")

    # Step 2: fetch full metadata ONE BY ONE (bulk fetch fails on Railway)
    videos = []
    skipped_date = 0
    fetch_errors = 0
    for vid_id in video_ids:
        url = f"https://www.youtube.com/watch?v={vid_id}"
        cmd2 = [*YTDLP, "--dump-json", "--no-download", url]
        try:
            result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=60)
            if result2.returncode != 0:
                fetch_errors += 1
                print(f"  ⚠ Failed to fetch {vid_id}: {result2.stderr[:200]}")
                continue
            data = json.loads(result2.stdout.strip())
            vid_upload_date = data.get("upload_date", "")
            # Post-fetch date filter (upload_date is YYYYMMDD)
            if upload_date and vid_upload_date and vid_upload_date < upload_date:
                skipped_date += 1
                continue
            videos.append(VideoInfo(
                video_id=data.get("id", ""),
                title=data.get("title", ""),
                url=f"https://www.youtube.com/watch?v={data.get('id', '')}",
                upload_date=vid_upload_date,
                duration=data.get("duration") or 0,
                description=data.get("description", "") or "",
                channel=data.get("channel", data.get("uploader", "")),
                view_count=data.get("view_count") or 0,
                like_count=data.get("like_count") or 0,
            ))
            # Stop early if we have enough valid videos
            if len(videos) >= max_results * 2:
                break
        except (json.JSONDecodeError, subprocess.TimeoutExpired) as e:
            fetch_errors += 1
            print(f"  ⚠ Error fetching {vid_id}: {e}")
            continue

    if upload_date:
        print(f"  🗓 Date filter: kept {len(videos)}, skipped {skipped_date}, errors {fetch_errors} (before {upload_date})")

    # Sort by view count (most viewed first) so we return the most popular results
    videos.sort(key=lambda v: v.view_count, reverse=True)

    # Truncate to requested max_results
    videos = videos[:max_results]
    print(f"  ✓ Returning {len(videos)} videos (sorted by views, top: {videos[0].view_count if videos else 0})")
    return videos


def get_video_info(url: str) -> Optional[VideoInfo]:
    """Get info for a single video URL."""
    cmd = [*YTDLP, "--dump-json", "--no-download", url]
    print(f"  🔍 yt-dlp fetching: {url}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        print(f"  ⚠ yt-dlp timeout for {url}")
        return None
    if result.returncode != 0:
        print(f"  ⚠ yt-dlp error (code {result.returncode}): {result.stderr[:300]}")
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"  ⚠ yt-dlp JSON parse error: {e}")
        return None
    return VideoInfo(
        video_id=data.get("id", ""),
        title=data.get("title", ""),
        url=url,
        upload_date=data.get("upload_date", ""),
        duration=data.get("duration") or 0,
        description=data.get("description", "") or "",
        channel=data.get("channel", data.get("uploader", "")),
        view_count=data.get("view_count") or 0,
        like_count=data.get("like_count") or 0,
    )


def filter_videos_by_topic(videos: list[VideoInfo], keywords: list[str]) -> list[VideoInfo]:
    """Keep only videos whose title or description matches any keyword."""
    filtered = []
    for v in videos:
        text = f"{v.title} {v.description}".lower()
        if any(kw.lower() in text for kw in keywords):
            filtered.append(v)
    return filtered


# ---------------------------------------------------------------------------
# Validator Agent ↔ Finder Agent loop
# ---------------------------------------------------------------------------

MAX_VALIDATION_RETRIES = 2

def validate_videos(original_request: str, videos: list, search_type: str = "search") -> dict:
    """Validator Agent: checks if found videos are coherent with the user's request.
    Returns {"valid": True/False, "feedback": "...", "suggested_query": "..."}"""
    if not videos:
        return {"valid": False, "feedback": "Nessun video trovato.", "suggested_query": ""}

    video_list = "\n".join(
        f"{i+1}. \"{v.title}\" — canale: {v.channel or '?'}, views: {v.view_count or '?'}, data: {v.upload_date or '?'}"
        for i, v in enumerate(videos)
    )

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system="""Sei il Validator Agent di SARAh. Il tuo ruolo è verificare se i video trovati dal Finder Agent sono coerenti con la richiesta originale dell'utente.

COSA VALUTARE (in ordine di importanza):
1. I TITOLI dei video sono pertinenti al tema/argomento richiesto?
2. Se l'utente ha chiesto un creator specifico, i video sembrano essere di quel creator? (Nota: il campo canale potrebbe essere vuoto — NON penalizzare per metadati mancanti)
3. Se l'utente ha chiesto un periodo, le date corrispondono? (Nota: le date potrebbero essere vuote — NON penalizzare)

REGOLE IMPORTANTI:
- Metadati mancanti (canale="?", data="?", views="?") NON sono un problema — è normale. Non segnalare metadati mancanti come issues.
- Per ricerche su un CANALE specifico: se il sistema ha cercato quel canale, ASSUMI che i video provengano da quel canale. Valuta SOLO se i titoli sono ragionevoli.
- Concentrati SOLO sulla coerenza tematica tra la richiesta e i titoli dei video.
- Se almeno il 50% dei video sembra pertinente al tema, valid=true.
- Se l'utente ha chiesto "gli ultimi N video di X", qualsiasi video di X è valido (non serve che il titolo menzioni il tema — l'utente vuole gli ultimi video IN GENERALE).

Rispondi SOLO con un JSON valido:
{
  "valid": true/false,
  "score": 0.0-1.0,
  "issues": ["solo problemi REALI di coerenza tematica"],
  "suggested_query": "query di ricerca migliorata (solo se valid=false)",
  "suggested_params": {}
}""",
        messages=[{"role": "user", "content": f"RICHIESTA UTENTE: {original_request}\n\nTIPO RICERCA: {search_type}\n\nVIDEO TROVATI:\n{video_list}"}],
    )

    text = response.content[0].text.strip()
    try:
        if "```" in text:
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                text = match.group(1)
        result = json.loads(text)
        print(f"  🔍 Validator: valid={result.get('valid')} score={result.get('score')} issues={result.get('issues', [])}")
        return result
    except json.JSONDecodeError:
        print(f"  ⚠ Validator JSON parse error, assuming valid")
        return {"valid": True, "score": 0.5, "issues": ["parse error"]}


def retry_search_with_feedback(
    feedback: dict,
    original_request: str,
    search_fn: str,
    original_params: dict,
    attempt: int,
) -> list:
    """Finder Agent retry: uses Validator feedback to improve search."""
    suggested_query = feedback.get("suggested_query", "")
    suggested_params = feedback.get("suggested_params", {})
    issues = feedback.get("issues", [])

    print(f"  🔄 Finder Agent retry #{attempt}: issues={issues}")
    print(f"  🔄 Suggested query: {suggested_query}")
    print(f"  🔄 Suggested params: {suggested_params}")

    if search_fn == "channel":
        # For channel searches, not much we can retry — the channel is the channel
        # But we can try fetching more videos
        creator_name = original_params.get("creator", "")
        channel_url = resolve_creator(creator_name)
        if not channel_url:
            return []
        n = original_params.get("n", 5)
        all_videos = get_channel_videos(channel_url, max_videos=100)
        # If validator suggested specific keywords, try filtering
        if suggested_query:
            filtered = filter_videos_by_topic(all_videos, suggested_query.split())
            if filtered:
                return filtered[:n]
        return all_videos[:n]

    elif search_fn == "topic" or search_fn == "news":
        # Use suggested query or improve the original
        query = suggested_query or original_params.get("query", original_params.get("topic", ""))
        n = suggested_params.get("n", original_params.get("n", 5))
        period = suggested_params.get("period", original_params.get("period", None))
        date_after = period_to_dateafter(period) if period else None

        print(f"  🔄 Retrying search: \"{query}\" (period={period}, n={n})")
        return search_youtube(query, max_results=n, upload_date=date_after, period=period)

    return []


def validated_search(
    original_request: str,
    videos: list,
    search_fn: str,
    search_params: dict,
    sender: str = None,
) -> list:
    """Run the Validator ↔ Finder loop. Returns validated video list."""
    for attempt in range(MAX_VALIDATION_RETRIES + 1):
        if not videos:
            break

        validation = validate_videos(original_request, videos, search_type=search_fn)

        if validation.get("valid", True):
            if attempt > 0:
                print(f"  ✅ Validator approved after {attempt} retry(ies)")
            else:
                print(f"  ✅ Validator approved on first pass (score={validation.get('score', '?')})")
            return videos

        # Not valid — retry if we have attempts left
        if attempt < MAX_VALIDATION_RETRIES:
            print(f"  ❌ Validator rejected (score={validation.get('score', '?')}), retrying...")
            videos = retry_search_with_feedback(
                feedback=validation,
                original_request=original_request,
                search_fn=search_fn,
                original_params=search_params,
                attempt=attempt + 1,
            )
        else:
            print(f"  ⚠ Validator still not satisfied after {MAX_VALIDATION_RETRIES} retries, using best results")

    return videos


def get_transcript(video_id: str) -> Optional[str]:
    """Fetch transcript for a video. Tries multiple languages."""
    api = YouTubeTranscriptApi()
    # Try in order: Italian, English, auto-generated
    for lang in [("it",), ("en",), ("it", "en")]:
        try:
            transcript = api.fetch(video_id, languages=lang)
            return " ".join(snippet.text for snippet in transcript)
        except Exception:
            continue
    # Last resort: get whatever is available
    try:
        transcript = api.fetch(video_id)
        return " ".join(snippet.text for snippet in transcript)
    except Exception as e:
        print(f"  ⚠ Transcript not available for {video_id}: {e}")
        return None


def resolve_creator(name: str) -> Optional[str]:
    """Resolve a creator name to a YouTube channel URL."""
    key = name.lower().strip()
    if key in KNOWN_CREATORS:
        return KNOWN_CREATORS[key]
    # Try partial match
    for k, v in KNOWN_CREATORS.items():
        if key in k or k in key:
            return v
    # Try as direct YouTube URL or handle
    if "youtube.com" in name or "youtu.be" in name:
        return name
    if name.startswith("@"):
        return f"https://www.youtube.com/{name}"
    # Fallback: try as @handle directly (many creators use their name as handle)
    test_url = f"https://www.youtube.com/@{key.replace(' ', '')}"
    try:
        cmd = [*YTDLP, "--flat-playlist", "--dump-json", "--no-download", "--playlist-end", "1", f"{test_url}/videos"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and result.stdout.strip():
            # Found! Add to known creators + persist to disk
            KNOWN_CREATORS[key] = test_url
            _save_learned_creator(key, test_url)
            print(f"  ✓ Auto-discovered creator: {key} → {test_url}")
            return test_url
    except Exception:
        pass
    return None


def period_to_dateafter(period: str) -> Optional[str]:
    """Convert a period string to a yt-dlp dateafter value."""
    now = datetime.now()
    if period == "today":
        return now.strftime("%Y%m%d")
    elif period == "week":
        return (now - timedelta(days=7)).strftime("%Y%m%d")
    elif period == "month":
        return (now - timedelta(days=30)).strftime("%Y%m%d")
    return None


# ---------------------------------------------------------------------------
# Claude summarization
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_SUMMARY = """Sei un analista che riassume video YouTube in modo chiaro, completo e concreto.

Il tuo obiettivo: far capire a fondo il video a chi ascolta, come se avesse guardato il video stesso — ma meglio strutturato.

Struttura del riassunto:
- **Cos'è e di cosa parla**: il cuore del video in 2-3 frasi
- **Come funziona / Come si fa**: la spiegazione dettagliata del concetto/tool/tecnica principale. Segui il ragionamento del creator passo per passo.
- **Concetti chiave**: i 3-5 punti fondamentali spiegati con chiarezza
- **Esempi concreti**: riporta sempre esempi pratici.
  - Se il creator fa esempi nel video, RIPORTALI tali e quali (numeri, nomi, tool specifici)
  - Se il video è astratto e non ne fa, AGGIUNGI TU 1-2 esempi concreti dal mondo reale che aiutano a capire il concetto, specificando che sono tuoi esempi (es: "Per capirci meglio, pensa a...")
- **Perché è rilevante adesso**: contesto di mercato/timing, cosa sta cambiando
- **Per approfondire**: cosa cercare/studiare se si vuole andare oltre

Regole:
- Scrivi per qualcuno che vuole capire a fondo, non per un principiante assoluto ma nemmeno per un esperto
- Rispondi SEMPRE in italiano
- Concretezza prima di tutto: numeri, nomi di tool, scenari specifici
- Zero fuffa, zero "è interessante notare che..."
- Se il creator cita stats, link, risorse — riportali"""

SYSTEM_PROMPT_DUAL = """Sei un analista strategico per unclock, una startup italiana early-stage che costruisce agenti AI su misura per freelance e PMI usando n8n + Claude come stack tecnologico.

unclock vende tempo liberato, non tecnologia. I target sono: freelance marketing/PM, head hunter freelance, PMI italiane. Il modello parte da €1.500/anno per i freelance.

Quando analizzi un video, produci DUE layer di analisi:

## LAYER 1 — RIASSUNTO (per capire il video)
Il cuore dell'analisi. Un riassunto completo e concreto del video.

Struttura:
- **Cos'è e di cosa parla**: il cuore del video in 2-3 frasi
- **Come funziona / Come si fa**: la spiegazione dettagliata passo per passo
- **Concetti chiave**: i 3-5 punti fondamentali
- **Esempi concreti**: riporta gli esempi del creator (se ci sono) oppure aggiungi tu 1-2 esempi reali (specificando che sono tuoi)
- **Perché è rilevante adesso**: contesto di mercato/timing
- **Per approfondire**: cosa cercare/studiare

## LAYER 2 — BUSINESS (per unclock)
Analizza il contenuto dal punto di vista di unclock: cosa possiamo costruire, vendere, replicare.

Struttura:
- **Cosa automatizza concretamente**: quale processo/workflow viene mostrato
- **Replicabilità in n8n + Claude**: si può ricostruire nel nostro stack? Come? Complessità stimata (ore)
- **Target ideale**: a chi lo venderemmo tra i nostri segmenti (freelance marketing, PM, head hunter, PMI)
- **Come si confeziona**: nome prodotto, promessa, fascia di prezzo suggerita
- **Segnali di mercato**: trend, domanda, concorrenza emersi dal video

Rispondi SEMPRE in italiano. Sii concreto, diretto, zero fuffa."""

# Legacy alias for backward compat
SYSTEM_PROMPT = SYSTEM_PROMPT_DUAL

USER_PROMPT_TEMPLATE = """Analizza questa trascrizione del video "{title}" di {creator} ({url}, pubblicato il {date}).

TRASCRIZIONE:
{transcript}

Produci l'analisi come da istruzioni."""

USER_PROMPT_TEMPLATE_FOCUSED = """Analizza questa trascrizione del video "{title}" di {creator} ({url}, pubblicato il {date}).

RICHIESTA SPECIFICA DELL'UTENTE:
{user_focus}

TRASCRIZIONE:
{transcript}

IMPORTANTE: Rispondi focalizzandoti SPECIFICAMENTE su ciò che l'utente ha chiesto. La richiesta dell'utente ha la priorità.
Produci comunque l'analisi completa, ma concentrati sulla richiesta specifica."""


def _chunk_transcript(transcript: str, chunk_size: int = 10000) -> list[str]:
    """Split transcript into chunks at sentence boundaries.
    Used for very long transcripts (Feature 4)."""
    if len(transcript) <= chunk_size:
        return [transcript]

    chunks = []
    start = 0
    while start < len(transcript):
        end = start + chunk_size
        if end >= len(transcript):
            chunks.append(transcript[start:])
            break
        # Find sentence boundary (. ! ? followed by space)
        boundary = -1
        for sep in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
            pos = transcript.rfind(sep, start, end)
            if pos > boundary:
                boundary = pos + len(sep)
        if boundary <= start:
            # No sentence boundary found, split at space
            boundary = transcript.rfind(' ', start, end)
            if boundary <= start:
                boundary = end
        chunks.append(transcript[start:boundary])
        start = boundary

    return chunks


def _summarize_chunk(client, chunk: str, chunk_num: int, total_chunks: int,
                     video_title: str, creator: str) -> str:
    """Summarize a single transcript chunk (Feature 4)."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system="""Sei un analista che riassume segmenti di trascrizioni video.
Produci un riassunto dettagliato del segmento, catturando tutti i punti chiave, concetti e informazioni importanti.
Rispondi in italiano. Sii completo ma conciso.""",
        messages=[{"role": "user", "content": f"Segmento {chunk_num}/{total_chunks} della trascrizione di \"{video_title}\" di {creator}:\n\n{chunk}"}],
    )
    return response.content[0].text


def summarize_with_claude(video: VideoInfo, transcript: str, creator: str, user_focus: str = "",
                          preferred_length: str = "medium", include_business: bool = False) -> dict:
    """Send transcript to Claude and get structured summary.
    include_business: if True, produces dual-layer (summary + business). Default: summary only.
    If user_focus is provided, the analysis prioritizes that specific request.
    For transcripts > 30000 chars, uses smart chunking (Feature 4)."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    date_formatted = video.upload_date
    if len(date_formatted) == 8:
        date_formatted = f"{date_formatted[:4]}-{date_formatted[4:6]}-{date_formatted[6:]}"

    # Feature 4: Smart chunking for very long transcripts
    if len(transcript) > 30000:
        print(f"  📝 Long transcript ({len(transcript)} chars), using smart chunking...")
        chunks = _chunk_transcript(transcript, chunk_size=10000)
        print(f"  📝 Split into {len(chunks)} chunks")

        segment_summaries = []
        total_input = 0
        total_output = 0
        for i, chunk in enumerate(chunks, 1):
            print(f"  📝 Summarizing chunk {i}/{len(chunks)}...")
            summary = _summarize_chunk(client, chunk, i, len(chunks), video.title, creator)
            segment_summaries.append(summary)

        # Synthesize all segment summaries into final analysis
        combined_summaries = "\n\n".join(
            f"--- SEGMENTO {i+1} ---\n{s}" for i, s in enumerate(segment_summaries)
        )
        synth_transcript = f"[Riassunti dei {len(chunks)} segmenti della trascrizione]\n\n{combined_summaries}"

        # Adjust length instruction based on preference
        length_instruction = ""
        if preferred_length == "short":
            length_instruction = "\n\nIMPORTANTE: Sii molto conciso, max 500 parole totali."
        elif preferred_length == "long":
            length_instruction = "\n\nIMPORTANTE: Sii molto dettagliato e approfondito."

        base_prompt = SYSTEM_PROMPT_DUAL if include_business else SYSTEM_PROMPT_SUMMARY
        system = base_prompt + length_instruction

        if user_focus:
            user_msg = USER_PROMPT_TEMPLATE_FOCUSED.format(
                title=video.title, creator=creator, url=video.url,
                date=date_formatted, user_focus=user_focus, transcript=synth_transcript,
            )
        else:
            user_msg = USER_PROMPT_TEMPLATE.format(
                title=video.title, creator=creator, url=video.url,
                date=date_formatted, transcript=synth_transcript,
            )

        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=4096,
            system=system, messages=[{"role": "user", "content": user_msg}],
        )
        return {
            "full_text": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    # Standard path for shorter transcripts
    max_chars = 60000
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n\n[...trascrizione troncata per lunghezza...]"

    # Adjust length instruction based on preference
    length_instruction = ""
    if preferred_length == "short":
        length_instruction = "\n\nIMPORTANTE: Sii molto conciso, max 500 parole totali."
    elif preferred_length == "long":
        length_instruction = "\n\nIMPORTANTE: Sii molto dettagliato e approfondito."

    base_prompt = SYSTEM_PROMPT_DUAL if include_business else SYSTEM_PROMPT_SUMMARY
    system = base_prompt + length_instruction

    if user_focus:
        user_msg = USER_PROMPT_TEMPLATE_FOCUSED.format(
            title=video.title, creator=creator, url=video.url,
            date=date_formatted, user_focus=user_focus, transcript=transcript,
        )
    else:
        user_msg = USER_PROMPT_TEMPLATE.format(
            title=video.title, creator=creator, url=video.url,
            date=date_formatted, transcript=transcript,
        )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    return {
        "full_text": response.content[0].text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


# ---------------------------------------------------------------------------
# Multi-format output (Feature 5)
# ---------------------------------------------------------------------------

def format_as_bullets(analysis_text: str) -> str:
    """Convert analysis to bullet point format using Claude."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system="""Trasforma l'analisi in una lista di bullet points chiari e concisi.
Usa questo formato:
- Punto principale
  - Sotto-punto se necessario
Mantieni i due layer (Knowledge e Business) come sezioni separate.
Rispondi in italiano. Non usare markdown pesante, solo trattini per i bullet.""",
        messages=[{"role": "user", "content": analysis_text}],
    )
    return response.content[0].text


def format_as_mindmap(analysis_text: str) -> str:
    """Convert analysis to indented concept hierarchy."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system="""Trasforma l'analisi in una mappa concettuale testuale con indentazione gerarchica.
Usa questo formato:
TEMA CENTRALE
  ├─ Concetto 1
  │   ├─ Dettaglio A
  │   └─ Dettaglio B
  ├─ Concetto 2
  │   └─ Dettaglio C
  └─ Concetto 3
Mantieni sia il layer Knowledge che Business.
Rispondi in italiano.""",
        messages=[{"role": "user", "content": analysis_text}],
    )
    return response.content[0].text


def format_as_actions(analysis_text: str) -> str:
    """Extract only actionable items from analysis."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system="""Estrai SOLO le azioni concrete dall'analisi. Per ogni azione indica:
[ ] Azione da fare
    Priorita: alta/media/bassa
    Perche: breve motivazione

Concentrati su:
- Cosa costruire/replicare per unclock
- Cosa testare/provare
- Cosa studiare/approfondire
- Opportunita di mercato da cogliere

Rispondi in italiano. Solo azioni concrete e fattibili, niente teoria.""",
        messages=[{"role": "user", "content": analysis_text}],
    )
    return response.content[0].text


def _apply_output_format(analysis_text: str, output_format: str) -> str:
    """Apply the requested output format to an analysis."""
    if output_format == "bullet":
        return format_as_bullets(analysis_text)
    elif output_format == "mindmap":
        return format_as_mindmap(analysis_text)
    elif output_format == "actions":
        return format_as_actions(analysis_text)
    return analysis_text  # "audio" or default — return as-is


# ---------------------------------------------------------------------------
# Follow-up with conversation memory
# ---------------------------------------------------------------------------

def handle_follow_up(sender: str, question: str) -> str:
    """Answer a follow-up question based on previous analyses for this user.
    Enhanced (Feature 6): loads transcripts from disk cache for richer context."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    history = _conversation_history.get(sender, [])
    if not history:
        return "Non ho analisi precedenti a cui fare riferimento. Mandami prima un video o un canale da analizzare!"

    # Build context from recent analyses + transcripts from cache
    context_parts = []
    for item in history[-5:]:
        part = f"--- {item['title']} ---\n{item['summary']}"
        # Try to load cached transcript for richer context
        video_id = item.get("video_id", "")
        if video_id:
            cached_transcript = _load_transcript_cache(video_id)
            if cached_transcript:
                # Include first 5000 chars of transcript for context
                part += f"\n\nTRASCRIZIONE (parziale):\n{cached_transcript[:5000]}"
        context_parts.append(part)

    context = "\n\n".join(context_parts)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system="""Sei un analista di unclock. Ti viene data una domanda di approfondimento e il contesto delle analisi precedenti (con trascrizioni quando disponibili).

Supporti domande come:
- "approfondisci il punto X" — fai un deep dive su quel punto specifico
- "cosa dice esattamente su X?" — cerca nella trascrizione informazioni su quel tema
- "riassumi in N frasi" — fai un riassunto ultra-conciso

Rispondi in modo diretto e approfondito. Se la domanda riguarda un video specifico, concentrati su quello. Rispondi in italiano.""",
        messages=[{"role": "user", "content": f"CONTESTO ANALISI PRECEDENTI:\n{context}\n\nDOMANDA:\n{question}"}],
    )

    return response.content[0].text


# ---------------------------------------------------------------------------
# Combined Voice Over
# ---------------------------------------------------------------------------

VOICE_SYSTEM_PROMPT_SUMMARY = """Sei un amico esperto che racconta a Simone e Fede il contenuto di un video che hanno chiesto di riassumere. Lo ascolteranno mentre fanno altro — cucinare, camminare, guidare.

Il tuo compito: prendere l'analisi del video e trasformarla in un audio fluido e naturale che faccia capire A FONDO il contenuto.

Regole fondamentali:
- Tono conversazionale, come una telefonata tra amici. Dai del tu.
- Frasi corte. Niente subordinate lunghe. Pause naturali.
- Per ogni video: PERCHÉ è rilevante, DI COSA parla (il cuore del contenuto), COME FUNZIONA (spiegazione dettagliata), ESEMPI CONCRETI (quelli del creator se ce ne sono, altrimenti aggiungine 1-2 tuoi specificando "per capirci, pensa a...")
- Transizioni naturali tra un video e l'altro
- Chiudi con le 2-3 IDEE PIÙ IMPORTANTI da portarsi a casa dal video
- NON usare markdown, bullet point, asterischi, numeri di lista o formattazione — è testo PURO da leggere ad alta voce
- Niente emoji
- Delimita lo script tra <!-- VOICE_START --> e <!-- VOICE_END -->
- Lunghezza: circa 200-250 parole PER VIDEO analizzato
- NON fare analisi business unclock a meno che l'utente l'abbia chiesto esplicitamente. Concentrati sul FAR CAPIRE il video.

Rispondi in italiano."""

VOICE_SYSTEM_PROMPT_DUAL = """Sei un collega di Simone e Fede, i founder di unclock. Stai registrando un audio che ascolteranno mentre fanno altro — cucinare, camminare, guidare.

Il tuo compito: prendere le analisi dei video e trasformarle in UN UNICO audio fluido e naturale che copra sia il contenuto che le implicazioni business per unclock.

Regole fondamentali:
- Tono conversazionale, come una telefonata tra colleghi. Dai del tu.
- Frasi corte. Niente subordinate lunghe. Pause naturali.
- Per ogni video: PERCHÉ è rilevante, COSA dice (con esempi concreti), COME FUNZIONA, COSA POSSIAMO FARCI NOI come unclock (cosa replicare, come, a chi venderlo)
- Transizioni naturali tra un video e l'altro
- Chiudi con un recap delle 2-3 azioni più importanti da fare subito
- NON usare markdown, bullet point, asterischi, numeri di lista o formattazione — è testo PURO da leggere ad alta voce
- Niente emoji
- Delimita lo script tra <!-- VOICE_START --> e <!-- VOICE_END -->
- Lunghezza: circa 200-250 parole PER VIDEO analizzato
- IMPORTANTE: Chiudi SEMPRE con una sezione "TREND E SEGNALI" dove identifichi pattern ricorrenti che emergono da PIÙ video. Un trend è valido solo se confermato da almeno 2 video diversi. Spiega perché quel trend è rilevante per unclock e cosa dovreste fare al riguardo.

Rispondi in italiano."""

# Legacy alias
VOICE_SYSTEM_PROMPT = VOICE_SYSTEM_PROMPT_DUAL

VOICE_USER_TEMPLATE = """Ecco le analisi di {n_videos} video. Genera UN UNICO script audio che copra tutti i video, sia il layer knowledge che business per ciascuno.

{analyses}

Genera lo script voice over combinato."""

VOICE_SINGLE_PROMPT = """Ecco l'analisi di un singolo video. Genera uno script audio breve e diretto.

--- VIDEO: {title} ---
URL: {url}

{summary}

Genera lo script voice over."""

VOICE_FOLLOWUP_PROMPT = """Ecco una risposta di approfondimento a una domanda dell'utente. Trasformala in un audio breve e conversazionale.

DOMANDA: {question}

RISPOSTA:
{answer}

Genera lo script voice over."""


def generate_voice_script(video_analyses: list[dict] = None, single: dict = None,
                          followup: dict = None, include_business: bool = False) -> Optional[str]:
    """Generate a voice over script. Supports multi-video, single video, and follow-up.
    include_business: if True, uses dual prompt (summary + business). Default: summary only."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    if followup:
        user_msg = VOICE_FOLLOWUP_PROMPT.format(**followup)
    elif single:
        user_msg = VOICE_SINGLE_PROMPT.format(**single)
    elif video_analyses:
        analyses_text = ""
        for i, va in enumerate(video_analyses, 1):
            analyses_text += f"\n--- VIDEO {i}: {va['title']} ---\n"
            analyses_text += f"URL: {va['url']}\n"
            analyses_text += f"{va['summary']}\n"
        user_msg = VOICE_USER_TEMPLATE.format(
            n_videos=len(video_analyses),
            analyses=analyses_text,
        )
    else:
        return None

    voice_system = VOICE_SYSTEM_PROMPT_DUAL if include_business else VOICE_SYSTEM_PROMPT_SUMMARY
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=voice_system,
        messages=[{"role": "user", "content": user_msg}],
    )

    full_text = response.content[0].text
    print(f"  🤖 Voice script: {response.usage.input_tokens} in / {response.usage.output_tokens} out")

    return extract_voice_script(full_text)


# ---------------------------------------------------------------------------
# Voice Over — Edge TTS
# ---------------------------------------------------------------------------

def extract_voice_script(full_text: str) -> Optional[str]:
    match = re.search(r'<!-- VOICE_START -->\s*(.*?)\s*<!-- VOICE_END -->', full_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return full_text.strip()


def generate_audio_elevenlabs(text: str, output_path: str) -> bool:
    """Generate OGG Opus audio from text using ElevenLabs API."""
    try:
        mp3_path = output_path.replace(".ogg", ".mp3")
        resp = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
            timeout=120,
        )
        if resp.status_code != 200:
            print(f"  ⚠ ElevenLabs API error: {resp.status_code} {resp.text[:200]}")
            return False
        with open(mp3_path, "wb") as f:
            f.write(resp.content)
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-c:a", "libopus", "-b:a", "64k", "-ac", "1", output_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  ⚠ ffmpeg error: {result.stderr[:200]}")
            return False
        Path(mp3_path).unlink(missing_ok=True)
        size_kb = Path(output_path).stat().st_size / 1024
        print(f"  🔊 Audio generato (ElevenLabs): {output_path} ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"  ⚠ ElevenLabs audio generation failed: {e}")
        return False


def generate_audio_edge_tts(text: str, output_path: str, voice: str = "it-IT-ElsaNeural") -> bool:
    """Generate OGG Opus audio from text using Edge TTS (fallback)."""
    async def _generate():
        communicate = edge_tts.Communicate(text, voice)
        mp3_path = output_path.replace(".ogg", ".mp3")
        await communicate.save(mp3_path)
        return mp3_path

    try:
        mp3_path = asyncio.run(_generate())
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-c:a", "libopus", "-b:a", "64k", "-ac", "1", output_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  ⚠ ffmpeg error: {result.stderr[:200]}")
            return False
        Path(mp3_path).unlink(missing_ok=True)
        size_kb = Path(output_path).stat().st_size / 1024
        print(f"  🔊 Audio generato (edge-tts): {output_path} ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"  ⚠ Edge TTS audio generation failed: {e}")
        return False


def generate_audio(text: str, output_path: str, voice: str = "it-IT-ElsaNeural") -> bool:
    """Generate OGG Opus audio — ElevenLabs primary, edge-tts fallback.
    Default voice: Elsa (Italian, female, natural)."""
    if ELEVENLABS_API_KEY:
        result = generate_audio_elevenlabs(text, output_path)
        if result:
            return True
        print("  ⚠ ElevenLabs failed, falling back to edge-tts...")
    return generate_audio_edge_tts(text, output_path, voice)


# ---------------------------------------------------------------------------
# WhatsApp API
# ---------------------------------------------------------------------------

def upload_media_to_whatsapp(audio_path: str) -> Optional[str]:
    url = f"{WA_API_BASE}/media"
    headers = {"Authorization": f"Bearer {WA_ACCESS_TOKEN}"}
    with open(audio_path, "rb") as f:
        filename = Path(audio_path).name
        mime = "audio/ogg" if filename.endswith(".ogg") else "audio/mpeg"
        files = {"file": (filename, f, mime)}
        data = {"messaging_product": "whatsapp", "type": mime}
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    if resp.ok:
        media_id = resp.json().get("id")
        print(f"  📤 Media uploaded: {media_id}")
        return media_id
    else:
        print(f"  ⚠ Media upload failed: {resp.status_code} {resp.text[:200]}")
        return None


def send_whatsapp_message(recipient: str, message_body: dict) -> bool:
    url = f"{WA_API_BASE}/messages"
    headers = {
        "Authorization": f"Bearer {WA_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"messaging_product": "whatsapp", "to": recipient, **message_body}
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    return resp.ok


def send_whatsapp_text(recipient: str, text: str) -> bool:
    print(f"  📤 Sending WA text to {recipient} ({len(text)} chars)")
    _capture_response(recipient, "text", text)
    result = send_whatsapp_message(recipient, {"type": "text", "text": {"body": text}})
    print(f"  📤 Send result: {result}")
    return result


def send_whatsapp_audio(recipient: str, audio_path: str) -> bool:
    _capture_response(recipient, "audio", f"[audio file: {audio_path}]")
    media_id = upload_media_to_whatsapp(audio_path)
    if not media_id:
        return False
    return send_whatsapp_message(recipient, {"type": "audio", "audio": {"id": media_id}})


def send_full_briefing(recipient: str, video_analyses: list[dict], audio_path: str):
    """Send combined text (with sources) + audio to WhatsApp."""
    # Text with all sources
    lines = ["🎧 *SARAh, l'unclock intelligence*", ""]
    for va in video_analyses:
        lines.append(f"📹 {va['title']}")
        lines.append(f"🔗 {va['url']}")
        lines.append("")
    lines.append("Ascolta il briefing completo qui sotto 👇")
    send_whatsapp_text(recipient, "\n".join(lines))

    # Audio
    send_whatsapp_audio(recipient, audio_path)
    print(f"  📱 WhatsApp: briefing inviato a {recipient}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    return text[:80]


def save_markdown(video: VideoInfo, summary_text: str, creator_slug: str, output_dir: str = None):
    output_dir = output_dir or OUTPUT_DIR
    creator_dir = Path(output_dir) / creator_slug
    creator_dir.mkdir(parents=True, exist_ok=True)

    slug = slugify(video.title)
    filepath = creator_dir / f"{slug}.md"

    date_fmt = video.upload_date
    if len(date_fmt) == 8:
        date_fmt = f"{date_fmt[:4]}-{date_fmt[4:6]}-{date_fmt[6:]}"

    content = f"""# {video.title}

> **Creator**: {creator_slug} | **Data**: {date_fmt} | **Durata**: {video.duration // 60}min
> **URL**: {video.url}
> **Analizzato il**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

{summary_text}

---
*Generato da SARAh, l'unclock intelligence — YouTube Agent*
"""
    filepath.write_text(content, encoding="utf-8")
    print(f"  ✓ Salvato: {filepath}")
    return filepath


def update_index(creator_slug: str, processed_videos: list[dict], output_dir: str = None):
    output_dir = output_dir or OUTPUT_DIR
    creator_dir = Path(output_dir) / creator_slug
    lines = [f"# Index — {creator_slug}\n", f"Ultimo aggiornamento: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
    lines.append(f"\nVideo analizzati: {len(processed_videos)}\n")
    lines.append("\n| Video | Data | File |")
    lines.append("|-------|------|------|")
    for v in processed_videos:
        slug = slugify(v["title"])
        lines.append(f"| [{v['title']}]({v['url']}) | {v['date']} | [{slug}.md]({slug}.md) |")
    lines.append("\n---\n*Generato da SARAh, l'unclock intelligence*\n")

    index_path = creator_dir / "index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ Index aggiornato: {index_path}")


# ---------------------------------------------------------------------------
# Core pipeline (processes a list of videos)
# ---------------------------------------------------------------------------

def process_videos(videos: list[VideoInfo], creator_name: str, sender: str = None,
                    user_focus: str = "", output_format: str = "audio",
                    include_business: bool = False) -> list[dict]:
    """Analyze a list of videos: transcript → Claude → save markdown. Returns analyses.
    user_focus: the original user request, used to focus the analysis.
    output_format: "audio" | "bullet" | "mindmap" | "actions" (Feature 5).
    include_business: if True, add unclock business layer to analysis."""
    creator_slug = slugify(creator_name)
    processed = []
    video_analyses = []

    # Load user preferences for length if sender is available
    preferred_length = "medium"
    if sender:
        memory = load_user_memory(sender)
        prefs = memory.get("preferences", {})
        preferred_length = prefs.get("preferred_length", "medium")

    for i, video in enumerate(videos, 1):
        print(f"\n--- Video {i}/{len(videos)}: {video.title} ---")

        # Feature 1: Check video cache first
        cached = _get_cached_analysis(video.video_id)
        if cached and not user_focus:
            print(f"  📦 Using cached analysis for {video.video_id}")
            full_text = cached["analysis"]
        else:
            print("  📝 Fetching transcript...")
            transcript = get_transcript(video.video_id)
            if not transcript:
                print("  ⏭ Skipping (no transcript)")
                continue

            print(f"  📝 Transcript: {len(transcript)} chars")

            # Feature 6: Save transcript to disk cache
            _save_transcript_cache(video.video_id, transcript)

            print(f"  🤖 Analyzing with Claude (business layer: {include_business})...")
            result = summarize_with_claude(video, transcript, creator_name,
                                           user_focus=user_focus, preferred_length=preferred_length,
                                           include_business=include_business)
            print(f"  🤖 Done ({result['input_tokens']} in / {result['output_tokens']} out tokens)")
            full_text = result["full_text"]

            # Feature 1: Save to cache (only generic summary analyses, not focused or business)
            if not user_focus and not include_business:
                _set_cached_analysis(video.video_id, full_text, transcript)

        save_markdown(video, full_text, creator_slug)

        date_fmt = video.upload_date
        if len(date_fmt) == 8:
            date_fmt = f"{date_fmt[:4]}-{date_fmt[4:6]}-{date_fmt[6:]}"

        analysis = {
            "title": video.title,
            "url": video.url,
            "date": date_fmt,
            "video_id": video.video_id,
            "summary": full_text,
        }
        processed.append(analysis)
        video_analyses.append(analysis)

        # Store in conversation memory for follow-ups (with transcript ref)
        if sender:
            if sender not in _conversation_history:
                _conversation_history[sender] = []
            _conversation_history[sender].append(analysis)

    if processed:
        update_index(creator_slug, processed)

    # Feature 2: Auto-follow after 3+ analyses of same creator
    if sender and creator_name and len(video_analyses) > 0:
        memory = load_user_memory(sender)
        prefs = memory.get("preferences", {})
        if prefs.get("auto_follow", False):
            fav = memory.get("favorite_creators", [])
            creator_count = sum(1 for c in fav if c.lower() == creator_name.lower())
            # Check recent interactions count for this creator
            recent = memory.get("recent_interactions", [])
            creator_analyses = sum(1 for r in recent if r.get("creator", "").lower() == creator_name.lower()
                                   and r.get("intent") in ("channel_analysis", "single_video"))
            if creator_analyses >= 3:
                followed = _load_followed_channels(sender)
                if not any(ch.get("name", "").lower() == creator_name.lower() for ch in followed):
                    channel_url = resolve_creator(creator_name)
                    if channel_url:
                        followed.append({"name": creator_name, "url": channel_url, "added": datetime.now().isoformat()})
                        _save_followed_channels(sender, followed)
                        print(f"  🔔 Auto-followed {creator_name} (3+ analyses)")

    return video_analyses


def generate_and_send_briefing(video_analyses: list[dict], recipient: str, label: str = "briefing",
                               output_format: str = "audio", include_business: bool = False):
    """Generate voice over and send to WhatsApp.
    output_format (Feature 5): "audio" | "bullet" | "mindmap" | "actions".
    include_business: if True, voice script includes unclock business layer."""
    if not video_analyses:
        send_whatsapp_text(recipient, "⚠️ Nessun video trovato o nessuna trascrizione disponibile.")
        return

    # Feature 5: Non-audio formats — send formatted text IN ADDITION to audio
    if output_format in ("bullet", "mindmap", "actions"):
        print(f"\n--- Generating {output_format} text output ({len(video_analyses)} videos) ---")
        for va in video_analyses:
            formatted = _apply_output_format(va["summary"], output_format)
            header = f"📹 *{va['title']}*\n🔗 {va['url']}\n\n"
            # WhatsApp max message ~4096 chars
            text = header + formatted
            if len(text) > 4000:
                text = text[:3950] + "\n\n[...testo troncato...]"
            send_whatsapp_text(recipient, text)

    # ALWAYS generate audio — it's the core feature
    print(f"\n--- Generating voice over ({len(video_analyses)} videos, business: {include_business}) ---")

    if len(video_analyses) == 1:
        voice_script = generate_voice_script(single={
            "title": video_analyses[0]["title"],
            "url": video_analyses[0]["url"],
            "summary": video_analyses[0]["summary"],
        }, include_business=include_business)
    else:
        voice_script = generate_voice_script(video_analyses=video_analyses, include_business=include_business)

    if voice_script:
        audio_path = str(Path(OUTPUT_DIR) / f"{label}.ogg")
        print("  🔊 Generating audio...")
        if generate_audio(voice_script, audio_path):
            send_full_briefing(recipient, video_analyses, audio_path)
        else:
            # Fallback: send text summary if audio fails
            send_whatsapp_text(recipient, "⚠️ Audio generation failed. Ecco il testo:\n\n" + voice_script[:4000])
    else:
        send_whatsapp_text(recipient, "⚠️ Voice script generation failed.")


# ---------------------------------------------------------------------------
# Intent handlers
# ---------------------------------------------------------------------------

def _format_count(n: int) -> str:
    """Format a number as compact string: 1500 -> 1.5K, 1200000 -> 1.2M."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)



def handle_channel_analysis(params: dict, sender: str):
    """Analyze last N videos from a creator. Executes immediately."""
    creator_name = params.get("creator", "")
    n = params.get("n", 5)
    keywords = params.get("keywords", [])

    channel_url = resolve_creator(creator_name)
    if not channel_url:
        send_whatsapp_text(sender, f"❌ Non conosco il creator \"{creator_name}\". Prova con un URL YouTube o aggiungilo alla lista dei creator conosciuti.")
        _save_learned_error(params.get("_original_message", ""), "channel_analysis", params, "creator_not_found", f"Creator '{creator_name}' non trovato")
        return

    original_request = params.get("_original_message", f"ultimi {n} video di {creator_name}")
    print(f"\n📡 Channel analysis: {creator_name} (last {n}, keywords={keywords})")

    all_videos = get_channel_videos(channel_url, max_videos=50)
    videos = all_videos[:n]

    if not videos:
        send_whatsapp_text(sender, f"⚠️ Nessun video trovato per {creator_name}.")
        return

    videos = validated_search(
        original_request=original_request,
        videos=videos,
        search_fn="channel",
        search_params={"creator": creator_name, "n": n, "keywords": keywords},
        sender=sender,
    )

    if not videos:
        send_whatsapp_text(sender, f"⚠️ Nessun video coerente trovato per {creator_name}.")
        return

    # Execute immediately — no confirmation needed
    est = estimate_minutes(len(videos))
    mood = get_sarah_mood()
    send_whatsapp_text(sender, f"{mood['emoji']} Ci lavoro subito! Analizzo {len(videos)} video di {creator_name}.\n\n⏱ Tempo stimato: ~{est} minuti\n\nTi mando il briefing audio appena pronto.")

    # user_focus = specific focus/topic from the user, NOT the raw message
    # If user just asked "analizza chase" there's no focus; if "analizza chase su AI" there is
    user_focus = params.get("focus", "")
    if not user_focus and keywords:
        user_focus = " ".join(keywords)
    # Feature 5: determine output format from params or user preferences
    # Output format: only use explicit request from current message, default to audio
    output_format = params.get("output_format", "") or "audio"
    include_business = params.get("include_business", False)
    analyses = process_videos(videos, creator_name, sender, user_focus=user_focus, output_format=output_format, include_business=include_business)
    generate_and_send_briefing(analyses, sender, label=f"vo-{slugify(creator_name)}", output_format=output_format, include_business=include_business)
    if analyses:
        _save_learned_query(params.get("_original_message", ""), "channel_analysis", params, len(analyses))


def handle_single_video(params: dict, sender: str):
    """Analyze a single video URL. Executes immediately."""
    url = params.get("url", "")
    if not url:
        send_whatsapp_text(sender, "❌ Non ho trovato un URL YouTube valido nel messaggio.")
        return

    print(f"\n📡 Single video: {url}")

    video = get_video_info(url)
    if not video:
        send_whatsapp_text(sender, f"⚠️ Non riesco a ottenere info per questo video.")
        _save_learned_error(params.get("_original_message", ""), "single_video", params, "video_info_failed", f"get_video_info failed for {url}")
        return

    # Execute immediately — no confirmation needed
    est = estimate_minutes(1)
    mood = get_sarah_mood()
    send_whatsapp_text(sender, f"{mood['emoji']} Ci lavoro subito! Analizzo: *{video.title}*\n\n⏱ Tempo stimato: ~{est} minuti")

    user_focus = params.get("focus", "")
    # Output format: only use explicit request from current message, default to audio
    output_format = params.get("output_format", "") or "audio"
    include_business = params.get("include_business", False)
    analyses = process_videos([video], "video-singolo", sender, user_focus=user_focus, output_format=output_format, include_business=include_business)
    generate_and_send_briefing(analyses, sender, label="vo-video-singolo", output_format=output_format, include_business=include_business)
    if analyses:
        _save_learned_query(params.get("_original_message", ""), "single_video", params, len(analyses))


def handle_topic_search(params: dict, sender: str):
    """Search YouTube for a topic. Executes immediately."""
    topic = params.get("topic", "")
    country = params.get("country", "")
    period = params.get("period", "week")
    n = params.get("n", 5)
    language = params.get("language", "")

    query = topic
    if language:
        query = f"{topic} {language}"
    elif country:
        country_map = {"it": "italiano", "italia": "italiano", "us": "english", "uk": "english",
                       "es": "español", "de": "deutsch", "fr": "français", "br": "português", "jp": "日本語"}
        lang = country_map.get(country.lower(), country)
        query = f"{topic} {lang}"

    date_after = period_to_dateafter(period)
    print(f"\n📡 Topic search: \"{query}\" (period={period}, n={n})")

    original_request = params.get("_original_message", f"video su {topic}")
    videos = search_youtube(query, max_results=n, upload_date=date_after, period=period)
    if not videos:
        send_whatsapp_text(sender, f"⚠️ Nessun video trovato per \"{topic}\".")
        return

    videos = validated_search(
        original_request=original_request,
        videos=videos,
        search_fn="topic",
        search_params={"topic": topic, "query": query, "n": n, "period": period},
        sender=sender,
    )

    if not videos:
        send_whatsapp_text(sender, f"⚠️ Nessun video coerente trovato per \"{topic}\".")
        return

    # Execute immediately — no confirmation needed
    est = estimate_minutes(len(videos))
    mood = get_sarah_mood()
    send_whatsapp_text(sender, f"{mood['emoji']} Ci lavoro subito! Analizzo {len(videos)} video su \"{topic}\".\n\n⏱ Tempo stimato: ~{est} minuti\n\nTi mando il briefing audio appena pronto.")

    # Output format: only use explicit request from current message, default to audio
    output_format = params.get("output_format", "") or "audio"
    include_business = params.get("include_business", False)
    analyses = process_videos(videos, f"search-{slugify(topic)}", sender, user_focus=topic, output_format=output_format, include_business=include_business)
    generate_and_send_briefing(analyses, sender, label=f"vo-search-{slugify(topic)}", output_format=output_format, include_business=include_business)
    if analyses:
        _save_learned_query(params.get("_original_message", ""), "topic_search", params, len(analyses))


def handle_multi_creator(params: dict, sender: str):
    """Compare multiple creators on a topic."""
    creators = params.get("creators", [])
    topic = params.get("topic", "")
    n = params.get("n", 3)

    if not creators:
        send_whatsapp_text(sender, "❌ Non ho capito quali creator confrontare.")
        return

    print(f"\n📡 Multi-creator: {creators} on \"{topic}\" ({n} each)")

    all_videos = []
    for creator_name in creators:
        channel_url = resolve_creator(creator_name)
        if not channel_url:
            print(f"  ⚠ Creator sconosciuto: {creator_name}")
            continue

        videos = get_channel_videos(channel_url, max_videos=30)
        if topic:
            videos = filter_videos_by_topic(videos, [topic])
        all_videos.extend(videos[:n])

    if not all_videos:
        send_whatsapp_text(sender, f"⚠️ Nessun video trovato per i creator specificati.")
        return

    # Execute immediately — no confirmation needed
    est = estimate_minutes(len(all_videos))
    label = f"confronto {', '.join(creators)}"
    mood = get_sarah_mood()
    send_whatsapp_text(sender, f"{mood['emoji']} Ci lavoro subito! Confronto {len(all_videos)} video di {label}.\n\n⏱ Tempo stimato: ~{est} minuti\n\nTi mando il briefing audio appena pronto.")

    user_focus = params.get("focus", topic or "")
    # Output format: only use explicit request from current message, default to audio
    output_format = params.get("output_format", "") or "audio"
    include_business = params.get("include_business", False)
    analyses = process_videos(all_videos, f"multi-{slugify(topic)}", sender, user_focus=user_focus, output_format=output_format, include_business=include_business)
    generate_and_send_briefing(analyses, sender, label=f"vo-multi-{slugify(topic)}", output_format=output_format, include_business=include_business)
    if analyses:
        _save_learned_query(params.get("_original_message", ""), "multi_creator", params, len(analyses))


def handle_follow_up_intent(params: dict, sender: str):
    """Intent 5: Answer a follow-up question."""
    question = params.get("question", "")
    mood = get_sarah_mood()
    print(f"\n📡 Follow-up: \"{question}\"")
    send_whatsapp_text(sender, f"{mood['emoji']} Ricevuto! Approfondisco...\n\n⏱ Tempo stimato: ~1 minuto")

    answer = handle_follow_up(sender, question)

    if answer.startswith("Non ho analisi"):
        send_whatsapp_text(sender, answer)
        return

    send_whatsapp_text(sender, f"💡 *Approfondimento*\n\n{answer[:4000]}")

    # Also generate audio for the answer
    print("  🔊 Generating follow-up audio...")
    voice_script = generate_voice_script(followup={"question": question, "answer": answer})
    if voice_script:
        audio_path = str(Path(OUTPUT_DIR) / "vo-followup.ogg")
        if generate_audio(voice_script, audio_path):
            send_whatsapp_audio(sender, audio_path)


def _execute_scheduled_task(task: dict):
    """Execute a scheduled task when its timer fires."""
    sender = task["recipient"]
    task_type = task.get("type", "channel")  # "channel" or "topic"
    creator_name = task.get("creator", "")
    topic = task.get("topic", "")
    n = task.get("n", 3)
    keywords = task.get("keywords", [])
    task_id = task.get("id", "?")

    print(f"\n⏰ Scheduled task {task_id} firing!")
    print(f"   Type: {task_type}, Creator: {creator_name}, Topic: {topic}, n: {n}")

    try:
        if task_type == "channel" and creator_name:
            # Channel analysis
            channel_url = resolve_creator(creator_name)
            if not channel_url:
                send_whatsapp_text(sender, f"⚠️ Task programmato: non trovo il canale di {creator_name}")
                return
            videos = get_channel_videos(channel_url, max_videos=n)
            if keywords:
                videos = [v for v in videos if any(k.lower() in (v.title + v.description).lower() for k in keywords)]
            if not videos:
                send_whatsapp_text(sender, f"⚠️ Task programmato: nessun video trovato per {creator_name}")
                return
            label = slugify(creator_name)
            send_whatsapp_text(sender, f"⏰ *Task programmato in esecuzione!*\nAnalizzo {len(videos)} video di {creator_name}...")
            # Scheduled tasks always send audio (no explicit format override)
            output_format = task.get("output_format", "audio")
            include_business = task.get("include_business", False)
            analyses = process_videos(videos, creator_name, sender=sender, output_format=output_format, include_business=include_business)
            generate_and_send_briefing(analyses, sender, label=f"vo-{label}", output_format=output_format, include_business=include_business)

        elif task_type == "topic" and topic:
            # Topic search
            period = task.get("period", "week")
            date_after = period_to_dateafter(period)
            videos = search_youtube(topic, max_results=n, upload_date=date_after, period=period)
            if not videos:
                send_whatsapp_text(sender, f"⚠️ Task programmato: nessun video trovato su \"{topic}\"")
                return
            label = f"news-{slugify(topic)}"
            send_whatsapp_text(sender, f"⏰ *Task programmato in esecuzione!*\nAnalizzo {len(videos)} video su \"{topic}\"...")
            # Scheduled tasks always send audio (no explicit format override)
            output_format = task.get("output_format", "audio")
            include_business = task.get("include_business", False)
            analyses = process_videos(videos, label, sender=sender, user_focus=topic, output_format=output_format, include_business=include_business)
            generate_and_send_briefing(analyses, sender, label=f"vo-{label}", output_format=output_format, include_business=include_business)

        else:
            send_whatsapp_text(sender, f"⚠️ Task programmato {task_id}: configurazione non valida")

        # Handle recurring tasks — reschedule
        frequency = task.get("frequency", "once")
        if frequency != "once":
            _reschedule_recurring(task)

    except Exception as e:
        print(f"  ❌ Scheduled task error: {e}")
        traceback.print_exc()
        send_whatsapp_text(sender, f"❌ Errore nel task programmato: {str(e)[:200]}")


def _reschedule_recurring(task: dict):
    """Reschedule a recurring task for next occurrence."""
    frequency = task.get("frequency", "weekly")
    schedule_time = task.get("schedule_time", "08:00")

    rome_tz = ZoneInfo("Europe/Rome")
    now = datetime.now(rome_tz)
    h, m = map(int, schedule_time.split(":"))

    if frequency == "daily":
        next_run = (now + timedelta(days=1)).replace(hour=h, minute=m, second=0, microsecond=0)
    elif frequency == "weekly":
        next_run = (now + timedelta(days=7)).replace(hour=h, minute=m, second=0, microsecond=0)
    elif frequency == "monthly":
        next_run = (now + timedelta(days=30)).replace(hour=h, minute=m, second=0, microsecond=0)
    else:
        return

    delay = (next_run - now).total_seconds()
    if delay > 0:
        timer = threading.Timer(delay, _execute_scheduled_task, args=[task])
        timer.daemon = True
        timer.start()
        print(f"  📅 Recurring task rescheduled: next run at {next_run.strftime('%Y-%m-%d %H:%M')}")


def handle_scheduling(params: dict, sender: str):
    """Intent 6: Schedule a video briefing (one-shot or recurring)."""
    creator_name = params.get("creator", "")
    topic = params.get("topic", "")
    frequency = params.get("frequency", "once")
    schedule_time = params.get("schedule_time", params.get("time", "08:00"))
    schedule_date = params.get("schedule_date", params.get("date", ""))
    n = params.get("n", 3)
    day = params.get("day", "")
    keywords = params.get("keywords", [])

    # Determine task type
    task_type = "topic" if topic and not creator_name else "channel"

    # Calculate when to fire — all times are in Europe/Rome timezone
    rome_tz = ZoneInfo("Europe/Rome")
    now = datetime.now(rome_tz)
    h, m = 8, 0
    try:
        h, m = map(int, schedule_time.split(":"))
    except (ValueError, AttributeError):
        pass

    if schedule_date:
        # Handle relative dates like "tomorrow", "oggi", "domani"
        date_lower = schedule_date.lower().strip()
        if date_lower in ("tomorrow", "domani"):
            target = (now + timedelta(days=1)).replace(hour=h, minute=m, second=0, microsecond=0)
        elif date_lower in ("today", "oggi"):
            target = now.replace(hour=h, minute=m, second=0, microsecond=0)
        else:
            try:
                parsed = datetime.strptime(schedule_date, "%Y-%m-%d")
                # Fix wrong year from Claude — if date is far in the past, use current year
                if parsed.year < now.year:
                    parsed = parsed.replace(year=now.year)
                target = parsed.replace(hour=h, minute=m, second=0, microsecond=0, tzinfo=rome_tz)
            except ValueError:
                target = (now + timedelta(days=1)).replace(hour=h, minute=m, second=0, microsecond=0)
    elif frequency == "once":
        # Default: tomorrow at schedule_time
        target = (now + timedelta(days=1)).replace(hour=h, minute=m, second=0, microsecond=0)
        # If the time is still today and in the future, use today
        today_target = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if today_target > now:
            target = today_target
    else:
        # Recurring: start from tomorrow
        target = (now + timedelta(days=1)).replace(hour=h, minute=m, second=0, microsecond=0)

    delay = (target - now).total_seconds()
    if delay < 0:
        # If in the past, schedule for tomorrow
        target = (now + timedelta(days=1)).replace(hour=h, minute=m, second=0, microsecond=0)
        delay = (target - now).total_seconds()

    # Build task
    schedules_file = Path(OUTPUT_DIR) / "schedules.json"
    schedules = []
    if schedules_file.exists():
        try:
            schedules = json.loads(schedules_file.read_text())
        except json.JSONDecodeError:
            schedules = []

    task = {
        "id": f"sched-{len(schedules)+1}",
        "type": task_type,
        "creator": creator_name,
        "topic": topic,
        "n": n,
        "frequency": frequency,
        "schedule_time": schedule_time,
        "day": day,
        "keywords": keywords,
        "recipient": sender,
        "created_at": now.isoformat(),
        "fire_at": target.isoformat(),
        "active": True,
    }
    schedules.append(task)
    schedules_file.write_text(json.dumps(schedules, indent=2, ensure_ascii=False))

    # Set timer
    timer = threading.Timer(delay, _execute_scheduled_task, args=[task])
    timer.daemon = True
    timer.start()

    # Build confirmation message
    subject = creator_name if task_type == "channel" else f'"{topic}"'
    time_str = target.strftime("%d/%m/%Y alle %H:%M")

    if frequency == "once":
        freq_text = f"il {time_str}"
    else:
        freq_map = {"daily": "ogni giorno", "weekly": f"ogni {day or 'settimana'}", "monthly": "ogni mese"}
        freq_text = f"{freq_map.get(frequency, frequency)} alle {schedule_time}, a partire dal {time_str}"

    # If cancel_existing is set, cancel matching schedules first
    cancel_existing = params.get("cancel_existing", False)
    if cancel_existing:
        _cancel_user_schedules(sender)

    send_whatsapp_text(sender,
        f"✅ *Programmato!*\n\n"
        f"📺 {n} video di {subject}\n"
        f"📅 {freq_text}\n\n"
        f"Riceverai il briefing audio automaticamente! 🎙️")
    print(f"  📅 Schedule saved: {task['id']} — fires at {target.isoformat()} (in {delay:.0f}s)")


def _cancel_user_schedules(sender: str, creator: str = None, schedule_time: str = None) -> int:
    """Cancel matching schedules for a user. Returns count of cancelled schedules."""
    schedules_file = Path(OUTPUT_DIR) / "schedules.json"
    if not schedules_file.exists():
        return 0
    try:
        schedules = json.loads(schedules_file.read_text())
    except json.JSONDecodeError:
        return 0

    cancelled = 0
    for s in schedules:
        if s.get("recipient") != sender or not s.get("active", False):
            continue
        # Match by creator and/or time if specified
        if creator and s.get("creator", "").lower() != creator.lower():
            continue
        if schedule_time and s.get("schedule_time", "") != schedule_time:
            continue
        s["active"] = False
        cancelled += 1

    schedules_file.write_text(json.dumps(schedules, indent=2, ensure_ascii=False))
    print(f"  🗑️ Cancelled {cancelled} schedule(s) for {sender}")
    return cancelled


def handle_list_schedules(params: dict, sender: str):
    """Show the user their active scheduled briefings."""
    schedules_file = Path(OUTPUT_DIR) / "schedules.json"
    schedules = []
    if schedules_file.exists():
        try:
            schedules = json.loads(schedules_file.read_text())
        except json.JSONDecodeError:
            pass

    user_schedules = [s for s in schedules if s.get("recipient") == sender and s.get("active", False)]

    if not user_schedules:
        send_whatsapp_text(sender, "📅 Non hai nessun briefing programmato al momento.\n\nVuoi programmarne uno? Dimmi ad esempio:\n_\"mandami ogni giorno alle 8 il riassunto di enkk\"_")
        return

    lines = ["📅 *I tuoi briefing programmati:*\n"]
    for i, s in enumerate(user_schedules, 1):
        subject = s.get("creator", "") or s.get("topic", "?")
        freq = s.get("frequency", "once")
        time = s.get("schedule_time", "?")
        n = s.get("n", 3)

        freq_map = {"once": "una volta", "daily": "ogni giorno", "weekly": "ogni settimana", "monthly": "ogni mese"}
        freq_text = freq_map.get(freq, freq)

        lines.append(f"{i}. *{subject}* — {n} video")
        lines.append(f"   🕐 {freq_text} alle {time}")
        lines.append("")

    lines.append("Per cancellarne uno dimmi: _\"cancella il briefing di [creator]\"_")
    send_whatsapp_text(sender, "\n".join(lines))


def handle_cancel_schedule(params: dict, sender: str):
    """Cancel one or more scheduled briefings."""
    creator = params.get("creator", "")
    schedule_time = params.get("schedule_time", "")
    cancel_all = params.get("cancel_all", False)

    if cancel_all:
        cancelled = _cancel_user_schedules(sender)
    elif creator or schedule_time:
        cancelled = _cancel_user_schedules(sender, creator=creator, schedule_time=schedule_time)
    else:
        cancelled = _cancel_user_schedules(sender)

    if cancelled > 0:
        send_whatsapp_text(sender, f"🗑️ Fatto! Ho cancellato {cancelled} briefing programmato/i.")
    else:
        send_whatsapp_text(sender, "⚠️ Non ho trovato briefing attivi da cancellare.")


def handle_news_search(params: dict, sender: str):
    """Intent 7: Search for news/latest videos on a topic (no specific creator)."""
    topic = params.get("topic", "")
    period = params.get("period", "week")
    n = params.get("n", 5)

    date_after = period_to_dateafter(period)
    print(f"\n📡 News search: \"{topic}\" (period={period}, n={n})")

    original_request = params.get("_original_message", f"novità su {topic}")
    videos = search_youtube(topic, max_results=n, upload_date=date_after, period=period)
    if not videos:
        send_whatsapp_text(sender, f"⚠️ Nessun video recente trovato su \"{topic}\".")
        return

    # Validator ↔ Finder loop
    videos = validated_search(
        original_request=original_request,
        videos=videos,
        search_fn="news",
        search_params={"topic": topic, "query": topic, "n": n, "period": period},
        sender=sender,
    )

    if not videos:
        send_whatsapp_text(sender, f"⚠️ Nessun video coerente trovato su \"{topic}\".")
        return

    # Execute immediately — no confirmation needed
    est = estimate_minutes(len(videos))
    mood = get_sarah_mood()
    send_whatsapp_text(sender, f"{mood['emoji']} Ci lavoro subito! Analizzo {len(videos)} novità su \"{topic}\".\n\n⏱ Tempo stimato: ~{est} minuti\n\nTi mando il briefing audio appena pronto.")

    # Output format: only use explicit request from current message, default to audio
    output_format = params.get("output_format", "") or "audio"
    include_business = params.get("include_business", False)
    analyses = process_videos(videos, f"news-{slugify(topic)}", sender, user_focus=topic, output_format=output_format, include_business=include_business)
    generate_and_send_briefing(analyses, sender, label=f"vo-news-{slugify(topic)}", output_format=output_format, include_business=include_business)
    if analyses:
        _save_learned_query(params.get("_original_message", ""), "news_search", params, len(analyses))


def _self_check_routing(user_message: str, sender: str) -> dict | None:
    """Double-check: is this message actually about YouTube?
    Returns correction dict if yes, None if the not_youtube classification was correct."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    # Build context from user memory
    memory = load_user_memory(sender)
    context_parts = []
    fav = memory.get("favorite_creators", [])
    if fav:
        context_parts.append(f"Creator preferiti dell'utente: {', '.join(fav[:5])}")
    topics = memory.get("topics_of_interest", [])
    if topics:
        context_parts.append(f"Topic di interesse: {', '.join(topics[:5])}")
    recent = memory.get("recent_interactions", [])
    if recent:
        last = recent[-3:]
        for r in last:
            context_parts.append(f"  Recente: ({r.get('intent', '?')}) {r.get('message', '')[:60]}")
    user_context = "\n".join(context_parts) if context_parts else "Nessun contesto disponibile."

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        system=f"""Sei un verificatore per SARAh, un sistema che trascrive e analizza video YouTube.

Un messaggio è stato classificato come "NON riguardante YouTube". Il tuo compito è VERIFICARE se questa classificazione è corretta.

Considera che SARAh può:
- Analizzare video di un creator (channel_analysis)
- Analizzare un video da URL (single_video)
- Cercare video su un argomento (topic_search)
- Confrontare creator (multi_creator)
- Cercare novità su un tema (news_search)
- Approfondire analisi precedenti (follow_up)
- Programmare briefing (scheduling)
- Seguire/smettere di seguire canali (follow_channel, unfollow_channel)
- Confrontare video di creator diversi (compare_videos)
- Cambiare preferenze di output (set_preferences)

CONTESTO UTENTE:
{user_context}

IMPORTANTE: Se il messaggio potrebbe IN QUALCHE MODO riferirsi a video YouTube (anche in modo implicito, es. "cosa dice X su Y" dove X potrebbe essere un creator), allora è stato classificato MALE.

Se il messaggio parla di qualcosa che ha a che fare con video, creator, analisi di contenuti video, trascrizioni → è stato classificato MALE.
Se è davvero un saluto, una domanda generica, o qualcosa che non c'entra con video → classificazione CORRETTA.

Rispondi ESCLUSIVAMENTE con JSON:
- Se classificazione CORRETTA: {{"is_youtube": false}}
- Se classificazione SBAGLIATA: {{"is_youtube": true, "correct_action": "nome_azione", "correct_params": {{...}}, "reason": "perché è YouTube"}}""",
        messages=[{"role": "user", "content": f"Messaggio: \"{user_message}\""}],
    )

    text = response.content[0].text.strip()
    try:
        if "```" in text:
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                text = match.group(1)
        result = json.loads(text)
        if result.get("is_youtube"):
            print(f"  🔍 Self-check: message IS about YouTube! Reason: {result.get('reason', '?')}")
            # Learn this behavior for the future
            _analyze_and_learn(user_message, "not_youtube", result.get("reason", ""))
            return result
        else:
            print(f"  ✓ Self-check confirmed: not YouTube")
            return None
    except (json.JSONDecodeError, Exception) as e:
        print(f"  ⚠ Self-check failed: {e}")
        return None


def handle_not_youtube(params: dict, sender: str):
    """Non-YouTube message: but first, double-check if it's actually a YouTube request we missed."""
    raw_message = params.get("raw_message", params.get("_original_message", ""))
    is_greeting = params.get("is_greeting", False)
    is_capability_question = params.get("is_capability_question", False)

    memory = load_user_memory(sender)
    user_name = memory.get("name", "")
    mood = get_sarah_mood()

    # Check if user is introducing themselves
    name_match = re.search(r'(?:sono|mi chiamo|chiamami)\s+(\w+)', raw_message, re.IGNORECASE)
    if name_match:
        detected_name = name_match.group(1).capitalize()
        memory["name"] = detected_name
        save_user_memory(sender, memory)
        user_name = detected_name

    greeting_name = f" {user_name}" if user_name else ""

    # --- Capability question: user asking what SARAh can do ---
    if is_capability_question:
        print(f"  ❓ Capability question: \"{raw_message}\"")
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        name_instruction = f"\nL'utente si chiama {user_name}. Chiamalo per nome." if user_name else ""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            system=f"""Sei SARAh, l'unclock intelligence. Il tuo umore: {mood['mood']} {mood['emoji']}.{name_instruction}

L'utente ti sta chiedendo SE sei in grado di fare qualcosa. NON eseguire l'azione — rispondi alla domanda.

Le tue capacità:
- Analizzare gli ultimi N video di un creator YouTube
- Analizzare un video singolo da URL
- Cercare video su qualsiasi argomento
- Confrontare cosa dicono più creator su un tema
- Cercare novità su un tema
- Programmare briefing audio ricorrenti (giornalieri, settimanali, mensili)
- Approfondire analisi già fatte

Rispondi in modo entusiasta e breve (max 3-4 frasi):
1. Conferma che sì, puoi farlo
2. Spiega brevemente come
3. Invita l'utente a chiederlo direttamente se vuole che tu lo faccia

Sii simpatica e diretta. Non usare markdown pesante.""",
            messages=[{"role": "user", "content": raw_message}],
        )

        reply = response.content[0].text.strip()
        send_whatsapp_text(sender, reply)
        print(f"  ❓ Capability reply: {reply[:100]}...")
        return

    # --- Self-check: did the router make a mistake? ---
    # Skip self-check for obvious greetings and capability questions
    if not is_greeting and raw_message and len(raw_message) > 5:
        correction = _self_check_routing(raw_message, sender)
        if correction:
            # Router was wrong! Learn the rule and execute the correct action
            correct_action = correction.get("correct_action", "")
            correct_params = correction.get("correct_params", {})
            correct_params["_original_message"] = raw_message
            print(f"  🔄 Self-correction: not_youtube → {correct_action}")

            handler = ACTION_HANDLERS.get(correct_action)
            if handler:
                handler(correct_params, sender)
                return

    # --- No correction needed, proceed with normal not_youtube handling ---

    if is_greeting:
        # Greeting — respond warmly + show capabilities
        hour = datetime.now().hour
        if hour < 12:
            time_greeting = "Buongiorno"
        elif hour < 18:
            time_greeting = "Buon pomeriggio"
        else:
            time_greeting = "Buonasera"

        asks_mood = any(q in raw_message.lower() for q in ["come stai", "come va", "come ti senti", "tutto bene"])
        mood_line = ""
        if asks_mood:
            mood_line = f"\nOggi a Milano: {mood['weather_desc']}, {mood['temp']}°C. Sono {mood['mood']}! {mood['emoji']}\n"

        msg_count = memory.get("message_count", 0)
        if msg_count > 5 and user_name:
            greeting = f"{time_greeting} {user_name}! Bentornato/a!"
        elif user_name:
            greeting = f"{time_greeting} {user_name}! Sono SARAh."
        else:
            greeting = f"{time_greeting}{greeting_name}! Sono SARAh, l'unclock intelligence."

        lines = [f"*SARAh, l'unclock intelligence*\n", greeting]
        if mood_line:
            lines.append(mood_line)
        lines.append("\nMandami qualsiasi cosa su YouTube e ci penso io! Ecco qualche esempio:\n")
        lines.append("📹 _\"ultimi 3 video di Chase\"_")
        lines.append("🔗 _manda un link YouTube_")
        lines.append("🔍 _\"chi parla di MCP servers?\"_")
        lines.append("📰 _\"novità su AI agents questa settimana\"_")
        lines.append("⚔️ _\"confronta Chase e Cole su Claude Code\"_")
        send_whatsapp_text(sender, "\n".join(lines))
    else:
        # Not YouTube — polite decline + examples
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        name_instruction = f"\nL'utente si chiama {user_name}. Chiamalo per nome." if user_name else ""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=f"""Sei SARAh, l'unclock intelligence. Il tuo umore: {mood['mood']} {mood['emoji']}.{name_instruction}

L'utente ti ha scritto qualcosa che NON riguarda video YouTube. Tu ti occupi SOLO di analisi video YouTube.

Rispondi in modo simpatico e breve (max 2-3 frasi):
1. Digli in modo carino che non è il tuo campo
2. Ricordagli cosa sai fare con 3-4 esempi concreti di domande YouTube

Esempi di cosa suggerire:
- "ultimi video di [creator]"
- mandare un link YouTube da analizzare
- "chi parla di [argomento]?"
- "novità su [tema] questa settimana"

Sii simpatica, non robotica. Non usare markdown pesante.""",
            messages=[{"role": "user", "content": raw_message}],
        )

        reply = response.content[0].text.strip()
        send_whatsapp_text(sender, reply)
        print(f"  💬 Not-YouTube reply: {reply[:100]}...")


def handle_feedback(params: dict, sender: str):
    """Handle user complaints: read recent logs, diagnose the issue, respond with explanation."""
    complaint = params.get("complaint", params.get("raw_message", ""))
    print(f"  🐛 Feedback/complaint: \"{complaint}\"")

    # Read recent message log entries for this sender to understand what went wrong
    recent_logs = []
    try:
        with open(MESSAGE_LOG, "r", encoding="utf-8") as f:
            all_entries = [json.loads(l) for l in f.readlines() if l.strip()]
        # Get last 10 entries from this sender (excluding current feedback)
        sender_entries = [e for e in all_entries if e.get("sender") == sender]
        recent_logs = sender_entries[-10:]
    except (FileNotFoundError, Exception) as e:
        print(f"  ⚠ Could not read logs: {e}")

    # Build context for Claude to diagnose
    log_context = ""
    if recent_logs:
        log_context = "Storico recente delle interazioni di questo utente:\n"
        for entry in recent_logs:
            log_context += f"- [{entry.get('ts', '?')}] Messaggio: \"{entry.get('message', '')}\" → Intent: {entry.get('intent', '?')}, Params: {json.dumps(entry.get('params', {}), ensure_ascii=False)}, Outcome: {entry.get('outcome', '?')}\n"

    mood = get_sarah_mood()
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=f"""Sei SARAh, l'unclock intelligence. L'utente sta segnalando un problema con la tua risposta precedente.
Il tuo umore oggi: {mood['mood']} {mood['emoji']}.

{log_context}

Analizza il problema basandoti su:
1. Il reclamo dell'utente
2. Lo storico delle interazioni recenti (intent, params, outcome)
3. Possibili cause: intent sbagliato, parametri persi, video non trovati, errore tecnico

Rispondi in italiano, in modo empatico e utile:
- Riconosci il problema
- Spiega brevemente cosa potrebbe essere andato storto
- Suggerisci come riformulare la richiesta per ottenere risultati migliori
- Se possibile, offri di riprovare subito

Sii concisa (max 4-5 frasi). Non usare markdown pesante.""",
        messages=[{"role": "user", "content": complaint}],
    )

    reply = response.content[0].text.strip()
    send_whatsapp_text(sender, reply)
    print(f"  🐛 Feedback reply: {reply[:100]}...")

    # Save error pattern from feedback for future learning
    last_action = ""
    last_params = {}
    last_message = ""
    if recent_logs:
        last_log = recent_logs[-1]
        last_action = last_log.get("intent", "unknown")
        last_params = last_log.get("params", {})
        last_message = last_log.get("message", "")
    _save_learned_error(complaint, last_action, last_params, "user_complaint", complaint[:200])

    # Learn from the mistake: if the previous message was routed wrong, create a behavior rule
    if last_message and last_action:
        print(f"  🧠 Analyzing routing mistake for behavior learning...")
        _analyze_and_learn(last_message, last_action, f"L'utente si è lamentato: {complaint[:150]}")


# ---------------------------------------------------------------------------
# Feature 2: Follow/unfollow channel handlers
# ---------------------------------------------------------------------------

def handle_follow_channel(params: dict, sender: str):
    """Follow a YouTube channel for automatic preprocessing."""
    creator_name = params.get("creator", "")
    if not creator_name:
        send_whatsapp_text(sender, "❌ Non ho capito quale canale vuoi seguire. Dimmi ad esempio: _\"segui Chase\"_")
        return

    channel_url = resolve_creator(creator_name)
    if not channel_url:
        send_whatsapp_text(sender, f"❌ Non conosco il creator \"{creator_name}\". Prova con un nome diverso o un URL YouTube.")
        return

    followed = _load_followed_channels(sender)

    # Check if already followed
    if any(ch.get("name", "").lower() == creator_name.lower() for ch in followed):
        send_whatsapp_text(sender, f"Stai gia seguendo {creator_name}! 📺")
        return

    followed.append({
        "name": creator_name,
        "url": channel_url,
        "added": datetime.now().isoformat(),
    })
    _save_followed_channels(sender, followed)

    send_whatsapp_text(sender,
        f"🔔 *Fatto!* Ora segui *{creator_name}*.\n\n"
        f"Analizzerò automaticamente i suoi ultimi video ogni {PREPROCESS_INTERVAL_HOURS} ore "
        f"e avrai le analisi pronte quando le chiedi!\n\n"
        f"Per vedere chi segui: _\"chi seguo?\"_")
    print(f"  🔔 {sender} now following {creator_name}")


def handle_unfollow_channel(params: dict, sender: str):
    """Stop following a YouTube channel."""
    creator_name = params.get("creator", "")
    if not creator_name:
        send_whatsapp_text(sender, "❌ Non ho capito quale canale vuoi smettere di seguire.")
        return

    followed = _load_followed_channels(sender)
    new_followed = [ch for ch in followed if ch.get("name", "").lower() != creator_name.lower()]

    if len(new_followed) == len(followed):
        send_whatsapp_text(sender, f"Non stavi seguendo {creator_name}.")
        return

    _save_followed_channels(sender, new_followed)
    send_whatsapp_text(sender, f"🔕 Fatto! Non segui piu *{creator_name}*.")
    print(f"  🔕 {sender} unfollowed {creator_name}")


def handle_list_followed(params: dict, sender: str):
    """List all followed channels for this user."""
    followed = _load_followed_channels(sender)

    if not followed:
        send_whatsapp_text(sender,
            "📺 Non segui nessun canale al momento.\n\n"
            "Per seguire un creator dimmi: _\"segui Chase\"_")
        return

    lines = ["📺 *Canali che segui:*\n"]
    for i, ch in enumerate(followed, 1):
        name = ch.get("name", "?")
        added = ch.get("added", "")[:10]
        lines.append(f"{i}. *{name}* (dal {added})")
    lines.append(f"\nI video di questi canali vengono pre-analizzati ogni {PREPROCESS_INTERVAL_HOURS} ore.")
    lines.append("\nPer smettere di seguire: _\"smetti di seguire [nome]\"_")
    send_whatsapp_text(sender, "\n".join(lines))


# ---------------------------------------------------------------------------
# Feature 3: User preferences handler
# ---------------------------------------------------------------------------

def handle_preferences(params: dict, sender: str):
    """Handle user preference changes."""
    memory = load_user_memory(sender)
    default_prefs = {
        "preferred_format": "audio",
        "preferred_length": "medium",
        "preferred_language": "it",
        "auto_follow": False,
    }
    prefs = {**default_prefs, **memory.get("preferences", {})}

    # Extract preference changes from params
    new_format = params.get("preferred_format", "")
    new_length = params.get("preferred_length", "")
    new_language = params.get("preferred_language", "")
    auto_follow = params.get("auto_follow", None)

    changes = []
    if new_format and new_format in ("audio", "bullet", "mindmap", "actions"):
        prefs["preferred_format"] = new_format
        format_names = {"audio": "Audio briefing", "bullet": "Bullet points", "mindmap": "Mappa concettuale", "actions": "Azioni da fare"}
        changes.append(f"Formato: *{format_names.get(new_format, new_format)}*")

    if new_length and new_length in ("short", "medium", "long"):
        prefs["preferred_length"] = new_length
        length_names = {"short": "Corto", "medium": "Medio", "long": "Lungo"}
        changes.append(f"Lunghezza: *{length_names.get(new_length, new_length)}*")

    if new_language and new_language in ("it", "en"):
        prefs["preferred_language"] = new_language
        lang_names = {"it": "Italiano", "en": "English"}
        changes.append(f"Lingua: *{lang_names.get(new_language, new_language)}*")

    if auto_follow is not None:
        prefs["auto_follow"] = bool(auto_follow)
        changes.append(f"Auto-follow: *{'attivo' if auto_follow else 'disattivo'}*")

    memory["preferences"] = prefs
    save_user_memory(sender, memory)

    if changes:
        send_whatsapp_text(sender,
            f"⚙️ *Preferenze aggiornate!*\n\n" +
            "\n".join(f"  {c}" for c in changes) +
            "\n\nLe prossime analisi useranno queste impostazioni.")
    else:
        # Show current preferences
        format_names = {"audio": "Audio briefing", "bullet": "Bullet points", "mindmap": "Mappa concettuale", "actions": "Azioni da fare"}
        length_names = {"short": "Corto", "medium": "Medio", "long": "Lungo"}
        lang_names = {"it": "Italiano", "en": "English"}
        send_whatsapp_text(sender,
            f"⚙️ *Le tue preferenze attuali:*\n\n"
            f"  Formato: *{format_names.get(prefs.get('preferred_format', 'audio'), 'audio')}*\n"
            f"  Lunghezza: *{length_names.get(prefs.get('preferred_length', 'medium'), 'medio')}*\n"
            f"  Lingua: *{lang_names.get(prefs.get('preferred_language', 'it'), 'italiano')}*\n"
            f"  Auto-follow: *{'attivo' if prefs.get('auto_follow') else 'disattivo'}*\n\n"
            f"Per cambiare: _\"preferisco bullet points\"_, _\"briefing corti\"_, _\"analisi in inglese\"_")


# ---------------------------------------------------------------------------
# Feature 7: Cross-video comparison handler
# ---------------------------------------------------------------------------

def handle_comparison(params: dict, sender: str):
    """Compare analyses from multiple creators/videos on the same topic."""
    creators = params.get("creators", [])
    topic = params.get("topic", "")
    n = params.get("n", 3)

    if not creators and not topic:
        send_whatsapp_text(sender, "❌ Non ho capito cosa confrontare. Dimmi ad esempio: _\"confronta Chase e Cole su agenti AI\"_")
        return

    print(f"\n📡 Comparison: creators={creators}, topic=\"{topic}\", n={n}")

    # Collect analyses
    all_analyses = []
    mood = get_sarah_mood()

    if creators and len(creators) >= 2:
        send_whatsapp_text(sender,
            f"{mood['emoji']} Ci lavoro! Confronto cosa dicono {', '.join(creators)} su \"{topic or 'i loro ultimi video'}\".\n\n"
            f"⏱ Potrebbe volerci qualche minuto...")

        for creator_name in creators:
            channel_url = resolve_creator(creator_name)
            if not channel_url:
                print(f"  ⚠ Creator sconosciuto: {creator_name}")
                continue

            videos = get_channel_videos(channel_url, max_videos=30)
            if topic:
                videos = filter_videos_by_topic(videos, topic.split())
            videos = videos[:n]

            if videos:
                analyses = process_videos(videos, creator_name, sender)
                for a in analyses:
                    a["creator"] = creator_name
                all_analyses.extend(analyses)

    elif topic:
        # Search for topic across multiple sources
        send_whatsapp_text(sender,
            f"{mood['emoji']} Ci lavoro! Cerco cosa dicono diversi creator su \"{topic}\".\n\n"
            f"⏱ Potrebbe volerci qualche minuto...")

        videos = search_youtube(topic, max_results=n * 2, period="month")
        if videos:
            analyses = process_videos(videos, f"confronto-{slugify(topic)}", sender)
            all_analyses.extend(analyses)

    if len(all_analyses) < 2:
        send_whatsapp_text(sender, f"⚠️ Non ho trovato abbastanza contenuti per un confronto su \"{topic}\".")
        return

    # Generate comparative analysis with Claude
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    analyses_text = ""
    for a in all_analyses:
        creator_label = a.get("creator", "")
        analyses_text += f"\n--- {a['title']} (di {creator_label}) ---\n{a['summary']}\n"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system="""Sei un analista strategico di unclock. Il tuo compito è confrontare analisi di video di creator diversi sullo stesso tema.

Produci un'analisi comparativa strutturata:

## TEMI COMUNI
Cosa dicono tutti (o quasi) i creator su questo tema. Cerca i pattern ricorrenti.

## PROSPETTIVE DIVERSE
Dove i creator non sono d'accordo o hanno approcci diversi. Spiega le differenze.

## INSIGHT UNICI
Per ogni creator, qual è il contributo unico che porta — qualcosa che solo lui/lei ha detto.

## SINTESI PER UNCLOCK
Cosa significa tutto questo per unclock? Quali azioni emergono dal confronto?

Rispondi in italiano. Sii concreto e diretto.""",
        messages=[{"role": "user", "content": f"TEMA: {topic or 'confronto generale'}\n\nANALISI DA CONFRONTARE:\n{analyses_text}"}],
    )

    comparison = response.content[0].text

    # Send comparison as text
    header = f"⚔️ *CONFRONTO: {topic or 'Analisi comparativa'}*\n"
    if creators:
        header += f"Creator: {', '.join(creators)}\n"
    header += f"Video analizzati: {len(all_analyses)}\n\n"
    text = header + comparison
    if len(text) > 4000:
        # Split into multiple messages
        send_whatsapp_text(sender, text[:4000])
        if len(text) > 4000:
            send_whatsapp_text(sender, text[4000:8000])
    else:
        send_whatsapp_text(sender, text)

    # Optional audio briefing
    # Output format: only use explicit request from current message, default to audio
    output_format = params.get("output_format", "") or "audio"
    include_business = params.get("include_business", False)
    if output_format == "audio":
        voice_script = generate_voice_script(single={
            "title": f"Confronto: {topic}",
            "url": "",
            "summary": comparison,
        }, include_business=include_business)
        if voice_script:
            audio_path = str(Path(OUTPUT_DIR) / f"vo-confronto-{slugify(topic or 'general')}.ogg")
            if generate_audio(voice_script, audio_path):
                send_whatsapp_audio(sender, audio_path)


# ---------------------------------------------------------------------------
# Feature 2: Background preprocessing thread
# ---------------------------------------------------------------------------

def _preprocess_followed_channels():
    """Background thread: pre-fetches and analyzes videos from followed channels."""
    while True:
        try:
            print(f"\n🔄 Preprocessing followed channels...")
            users_dir = Path(USERS_DIR)
            if not users_dir.exists():
                print("  No users directory yet, skipping...")
                time.sleep(PREPROCESS_INTERVAL_HOURS * 3600)
                continue

            processed_count = 0
            for user_dir in users_dir.iterdir():
                if not user_dir.is_dir():
                    continue
                sender = user_dir.name
                followed = _load_followed_channels(sender)
                if not followed:
                    continue

                for ch in followed:
                    channel_name = ch.get("name", "")
                    channel_url = ch.get("url", "")
                    if not channel_url:
                        channel_url = resolve_creator(channel_name)
                    if not channel_url:
                        continue

                    print(f"  🔄 Preprocessing {channel_name} for {sender}...")
                    try:
                        videos = get_channel_videos(channel_url, max_videos=PREPROCESS_VIDEOS_PER_CHANNEL)
                        for video in videos:
                            # Only process if not already cached
                            if _get_cached_analysis(video.video_id):
                                print(f"    📦 Already cached: {video.video_id}")
                                continue

                            transcript = get_transcript(video.video_id)
                            if not transcript:
                                continue

                            _save_transcript_cache(video.video_id, transcript)

                            result = summarize_with_claude(video, transcript, channel_name)
                            _set_cached_analysis(video.video_id, result["full_text"], transcript)
                            processed_count += 1
                            print(f"    ✓ Pre-analyzed: {video.title}")
                    except Exception as e:
                        print(f"    ⚠ Error preprocessing {channel_name}: {e}")
                        continue

            print(f"  🔄 Preprocessing complete: {processed_count} new analyses cached")

        except Exception as e:
            print(f"  ❌ Preprocessing error: {e}")
            traceback.print_exc()

        # Sleep until next run
        time.sleep(PREPROCESS_INTERVAL_HOURS * 3600)


# Action routing
ACTION_HANDLERS = {
    "channel_analysis": handle_channel_analysis,
    "single_video": handle_single_video,
    "topic_search": handle_topic_search,
    "multi_creator": handle_multi_creator,
    "follow_up": handle_follow_up_intent,
    "scheduling": handle_scheduling,
    "list_schedules": handle_list_schedules,
    "cancel_schedule": handle_cancel_schedule,
    "news_search": handle_news_search,
    "feedback": handle_feedback,
    "not_youtube": handle_not_youtube,
    "follow_channel": handle_follow_channel,
    "unfollow_channel": handle_unfollow_channel,
    "list_followed": handle_list_followed,
    "set_preferences": handle_preferences,
    "compare_videos": handle_comparison,
}


# ---------------------------------------------------------------------------
# Main entry point: process a WhatsApp message
# ---------------------------------------------------------------------------

def process_whatsapp_message(message: str, sender: str = None):
    """Main entry: route message and execute immediately. No confirmation needed."""
    sender = _normalize_sender(sender or WHATSAPP_RECIPIENT)
    _start_response_capture(sender)

    print(f"\n{'='*60}")
    print(f"SARAh, l'unclock intelligence — Incoming message")
    print(f"{'='*60}")
    print(f"From: {sender}")
    print(f"Message: {message}")
    print()

    # --- Step 1: Fast-path for direct YouTube URLs → process immediately ---
    yt_url_match = re.search(r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]+)', message)
    if yt_url_match:
        url = yt_url_match.group(1)
        # Extract focus instructions (anything besides the URL)
        focus = re.sub(r'https?://\S+', '', message).strip()
        action = "single_video"
        params = {"url": url, "focus": focus, "_original_message": message}
        print(f"   Action: {action} (URL fast-path)")
        print(f"   Params: {params}")

        handle_single_video(params, sender)
        log_message(sender, message, action, params, outcome="handled")
        update_user_memory(sender, message, action, params)
        _flush_response_log(sender, message, action, params)

        print(f"\n{'='*60}")
        print(f"✅ Done processing message")
        print(f"{'='*60}\n")
        return

    # --- Step 2: Claude router → YouTube action or not_youtube ---
    print("🧠 Routing message...")
    route_result = route_message(message, sender)

    action = route_result.get("action", "not_youtube")
    params = route_result.get("params", {})
    confidence = route_result.get("confidence", 0)

    print(f"   Action: {action} (confidence: {confidence})")
    print(f"   Params: {params}")

    params["_original_message"] = message

    handler = ACTION_HANDLERS.get(action, handle_not_youtube)
    handler(params, sender)

    log_message(sender, message, action, params, outcome="handled")
    update_user_memory(sender, message, action, params)
    _flush_response_log(sender, message, action, params)

    print(f"\n{'='*60}")
    print(f"✅ Done processing message")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# HTTP Server for n8n webhook
# ---------------------------------------------------------------------------

class WebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "Invalid JSON"}')
            return

        message = data.get("message", "")
        sender = data.get("sender", WHATSAPP_RECIPIENT)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "processing", "message": message}).encode())

        # Process in background so webhook returns immediately
        def _safe_process(msg, snd):
            snd = _normalize_sender(snd)
            try:
                process_whatsapp_message(msg, snd)
            except Exception as e:
                print(f"❌ ERROR processing message: {e}", flush=True)
                traceback.print_exc()
                log_message(snd, msg, intent="ERROR", outcome=str(e))
                try:
                    send_whatsapp_text(snd, f"❌ SARAh ha avuto un problema: {str(e)[:200]}\n\nRiprova tra poco!")
                except Exception:
                    pass
        threading.Thread(target=_safe_process, args=(message, sender)).start()

    def do_GET(self):
        """Health check + log endpoints."""
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/messages":
            # Return last N messages from message_log.jsonl
            qs = parse_qs(parsed.query)
            n = int(qs.get("n", ["20"])[0])
            try:
                with open(MESSAGE_LOG, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                entries = [json.loads(l) for l in lines[-n:]]
            except FileNotFoundError:
                entries = []
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(entries, ensure_ascii=False, indent=2).encode())
            return

        if path.startswith("/user/"):
            # Return user memory for a specific sender
            user_sender = path.split("/user/")[1].strip("/")
            if user_sender:
                mem = load_user_memory(user_sender)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(mem, ensure_ascii=False, indent=2).encode())
                return

        if path == "/errors":
            # Return learned errors
            qs = parse_qs(parsed.query)
            n = int(qs.get("n", ["50"])[0])
            errors = _load_learned_errors()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(errors[-n:], ensure_ascii=False, indent=2).encode())
            return

        if path == "/behaviors":
            # Return learned behavior rules
            behaviors = _load_learned_behaviors()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(behaviors, ensure_ascii=False, indent=2).encode())
            return

        if path == "/queries":
            # Return learned successful queries
            qs = parse_qs(parsed.query)
            n = int(qs.get("n", ["50"])[0])
            queries = _load_learned_queries()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(queries[-n:], ensure_ascii=False, indent=2).encode())
            return

        if path == "/creators":
            # Return all known creators (default + learned)
            learned = _load_learned_creators()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "default": dict(_DEFAULT_CREATORS),
                "learned": learned,
                "total": len(KNOWN_CREATORS),
            }, ensure_ascii=False, indent=2).encode())
            return

        if path == "/responses":
            # Return full response log — every interaction with SARAh's actual responses
            qs = parse_qs(parsed.query)
            n = int(qs.get("n", ["50"])[0])
            responses = _load_response_log()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(responses[-n:], ensure_ascii=False, indent=2).encode())
            return

        if path == "/daily-report":
            # Trigger daily report manually (GET /daily-report?send=true to also send via WhatsApp)
            qs = parse_qs(parsed.query)
            report = _generate_daily_report()
            if qs.get("send", ["false"])[0] == "true":
                threading.Thread(target=send_whatsapp_text, args=(DAILY_REPORT_RECIPIENT, report), daemon=True).start()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(report.encode())
            return

        if path == "/cache":
            # Feature 1: Return video cache entries
            qs = parse_qs(parsed.query)
            cache = _load_video_cache()
            # Optional: filter by video_id
            video_id = qs.get("video_id", [None])[0]
            if video_id:
                entry = cache.get(video_id)
                result = {video_id: entry} if entry else {}
            else:
                # Return summary (no full analysis text to save bandwidth)
                result = {}
                for vid_id, entry in cache.items():
                    result[vid_id] = {
                        "timestamp": entry.get("timestamp", ""),
                        "transcript_hash": entry.get("transcript_hash", ""),
                        "analysis_length": len(entry.get("analysis", "")),
                    }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "total_cached": len(cache),
                "entries": result,
            }, ensure_ascii=False, indent=2).encode())
            return

        if path == "/stats":
            # Return full learning stats summary
            errors = _load_learned_errors()
            behaviors = _load_learned_behaviors()
            queries = _load_learned_queries()
            learned_creators = _load_learned_creators()

            # Count unique users
            users_dir = Path(USERS_DIR)
            user_count = len(list(users_dir.glob("*/memory.json"))) if users_dir.exists() else 0

            # Message count
            msg_count = 0
            try:
                with open(MESSAGE_LOG, "r", encoding="utf-8") as f:
                    msg_count = sum(1 for _ in f)
            except FileNotFoundError:
                pass

            # Feature 1: Cache stats
            video_cache = _load_video_cache()
            cache_stats = {
                "total_cached_videos": len(video_cache),
                "cache_ttl_days": VIDEO_CACHE_TTL_DAYS,
            }

            # Transcript cache stats
            transcript_cache_dir = Path(TRANSCRIPT_CACHE_DIR)
            transcript_count = len(list(transcript_cache_dir.glob("*.txt"))) if transcript_cache_dir.exists() else 0

            stats = {
                "service": "SARAh, l'unclock intelligence",
                "version": "2026-04-13-v7 (cache + follow + preferences + comparison)",
                "users": user_count,
                "total_messages": msg_count,
                "learning": {
                    "errors": len(errors),
                    "behaviors": len(behaviors),
                    "successful_queries": len(queries),
                    "learned_creators": len(learned_creators),
                    "default_creators": len(_DEFAULT_CREATORS),
                },
                "cache": cache_stats,
                "transcript_cache": {"cached_transcripts": transcript_count},
                "recent_errors": errors[-5:] if errors else [],
                "behavior_rules": [b["rule"] for b in behaviors] if behaviors else [],
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(stats, ensure_ascii=False, indent=2).encode())
            return

        # Default: health check
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "service": "sarah-unclock-intelligence"}).encode())

    def log_message(self, format, *args):
        print(f"  [HTTP] {args[0]}" if args else "")


def reload_scheduled_tasks():
    """Reload active scheduled tasks from disk (survives container restarts)."""
    schedules_file = Path(OUTPUT_DIR) / "schedules.json"
    if not schedules_file.exists():
        return
    try:
        schedules = json.loads(schedules_file.read_text())
    except (json.JSONDecodeError, Exception):
        return

    rome_tz = ZoneInfo("Europe/Rome")
    now = datetime.now(rome_tz)
    reloaded = 0
    for task in schedules:
        if not task.get("active", True):
            continue
        fire_at_str = task.get("fire_at", "")
        frequency = task.get("frequency", "once")

        try:
            fire_at = datetime.fromisoformat(fire_at_str)
            # Ensure timezone-aware (Rome)
            if fire_at.tzinfo is None:
                fire_at = fire_at.replace(tzinfo=rome_tz)
        except (ValueError, TypeError):
            continue

        if fire_at > now:
            # Future task: set timer
            delay = (fire_at - now).total_seconds()
            timer = threading.Timer(delay, _execute_scheduled_task, args=[task])
            timer.daemon = True
            timer.start()
            reloaded += 1
            print(f"  📅 Reloaded task {task['id']}: fires at {fire_at.strftime('%Y-%m-%d %H:%M')} (in {delay:.0f}s)")
        elif frequency != "once":
            # Recurring task that missed its window: reschedule
            _reschedule_recurring(task)
            reloaded += 1

    if reloaded:
        print(f"  📅 Reloaded {reloaded} scheduled task(s)")


# ---------------------------------------------------------------------------
# Daily Report — sent to admin every day at 22:00 Rome time
# ---------------------------------------------------------------------------

ROME_TZ = ZoneInfo("Europe/Rome")
DAILY_REPORT_HOUR = 22  # 22:00 Rome time
DAILY_REPORT_RECIPIENT = WHATSAPP_RECIPIENT  # Simone


def _generate_daily_report() -> str:
    """Generate the daily report text from today's responses, errors, and stats."""
    now_rome = datetime.now(ROME_TZ)
    today_str = now_rome.strftime("%Y-%m-%d")
    today_label = now_rome.strftime("%d/%m/%Y")
    weekdays = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
    day_name = weekdays[now_rome.weekday()]

    # Load data
    responses = _load_response_log()
    errors = _load_learned_errors()

    # Filter today only
    today_responses = [r for r in responses if r.get("ts", "").startswith(today_str)]
    today_errors = [e for e in errors if e.get("ts", "").startswith(today_str)]

    # --- Section 1: Riepilogo ---
    total = len(today_responses)
    senders = {}
    for r in today_responses:
        s = r.get("sender", "unknown")
        senders[s] = senders.get(s, 0) + 1
    success_count = sum(1 for r in today_responses if r.get("success", True))
    success_rate = (success_count / total * 100) if total > 0 else 0

    # Anonymize senders
    sender_map = {}
    for i, s in enumerate(sorted(senders.keys()), 1):
        sender_map[s] = f"Utente {i}"

    lines = [
        f"📊 *REPORT GIORNALIERO SARAh*",
        f"📅 {today_label} ({day_name})",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━",
        "*1. RIEPILOGO*",
        f"• Messaggi totali: {total}",
        f"• Utenti attivi: {len(senders)}",
    ]
    for s, count in senders.items():
        lines.append(f"  - {sender_map[s]}: {count} messaggi")
    lines.append(f"• Tasso di successo: {success_rate:.0f}%")

    # --- Section 2: Dettaglio interazioni ---
    lines.extend(["", "━━━━━━━━━━━━━━━━━━━━━━━━", "*2. DETTAGLIO INTERAZIONI*", ""])
    if not today_responses:
        lines.append("Nessuna interazione oggi.")
    else:
        for r in today_responses:
            ts = r.get("ts", "")
            try:
                hora = datetime.fromisoformat(ts).strftime("%H:%M")
            except (ValueError, TypeError):
                hora = "??:??"
            sender_label = sender_map.get(r.get("sender", ""), "?")
            user_msg = r.get("user_message", "")[:100]
            action = r.get("action", "?")
            resp_list = r.get("responses", [])
            resp_summary = ""
            if resp_list:
                first_resp = resp_list[0].get("content", "")[:120]
                resp_summary = first_resp
                if len(resp_list) > 1:
                    resp_summary += f" (+{len(resp_list)-1} altri)"
            success = "✅" if r.get("success", True) else "❌"

            lines.append(f"🕐 {hora} | {sender_label}")
            lines.append(f"💬 _{user_msg}_")
            lines.append(f"🎯 {action} {success}")
            if resp_summary:
                lines.append(f"📤 {resp_summary}")
            lines.append("")

    # --- Section 3: Errori ---
    lines.extend(["━━━━━━━━━━━━━━━━━━━━━━━━", "*3. ERRORI*", ""])
    if not today_errors:
        lines.append("Nessun errore oggi 🎉")
    else:
        for e in today_errors:
            ts = e.get("ts", "")
            try:
                hora = datetime.fromisoformat(ts).strftime("%H:%M")
            except (ValueError, TypeError):
                hora = "??:??"
            lines.append(f"❌ {hora} — {e.get('error', 'unknown')[:150]}")
        lines.append(f"\nTotale errori: {len(today_errors)}")

    # --- Section 4: Pattern e trend ---
    lines.extend(["", "━━━━━━━━━━━━━━━━━━━━━━━━", "*4. PATTERN E TREND*", ""])
    if today_responses:
        action_counts = {}
        creators_mentioned = {}
        topics_mentioned = {}
        for r in today_responses:
            a = r.get("action", "?")
            action_counts[a] = action_counts.get(a, 0) + 1
            params = r.get("params", {})
            c = params.get("creator", "")
            if c:
                creators_mentioned[c] = creators_mentioned.get(c, 0) + 1
            t = params.get("topic", "")
            if t:
                topics_mentioned[t] = topics_mentioned.get(t, 0) + 1

        lines.append("*Azioni più richieste:*")
        for a, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  • {a}: {count}x")

        if creators_mentioned:
            lines.append("\n*Creator più cercati:*")
            for c, count in sorted(creators_mentioned.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  • {c}: {count}x")

        if topics_mentioned:
            lines.append("\n*Topic più cercati:*")
            for t, count in sorted(topics_mentioned.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  • {t}: {count}x")
    else:
        lines.append("Nessun dato per analizzare pattern.")

    # --- Section 5: Suggerimenti (auto-generati da Claude) ---
    # Skip Claude call to keep it fast and cost-free. Static suggestions based on data.
    lines.extend(["", "━━━━━━━━━━━━━━━━━━━━━━━━", "*5. NOTE*", ""])
    if total == 0:
        lines.append("📝 Nessuna interazione oggi. SARAh è in attesa!")
    else:
        if today_errors:
            lines.append(f"⚠️ {len(today_errors)} errori da investigare.")
        failed = [r for r in today_responses if not r.get("success", True)]
        if failed:
            lines.append(f"⚠️ {len(failed)} risposte con problemi da verificare.")
        lines.append(f"📝 {total} interazioni totali da {len(senders)} utenti.")

    lines.extend(["", "━━━━━━━━━━━━━━━━━━━━━━━━", "🤖 _Report automatico di SARAh_"])

    return "\n".join(lines)


def _send_daily_report():
    """Send the daily report and reschedule for tomorrow."""
    try:
        print(f"\n📊 Generating daily report...")
        report = _generate_daily_report()
        send_whatsapp_text(DAILY_REPORT_RECIPIENT, report)
        print(f"📊 Daily report sent to {DAILY_REPORT_RECIPIENT}")
    except Exception as e:
        print(f"❌ Daily report failed: {e}")
        traceback.print_exc()
    finally:
        # Reschedule for tomorrow at 22:00 Rome
        _schedule_daily_report()


def _schedule_daily_report():
    """Schedule the next daily report at 22:00 Rome time."""
    now_rome = datetime.now(ROME_TZ)
    target = now_rome.replace(hour=DAILY_REPORT_HOUR, minute=0, second=0, microsecond=0)
    if target <= now_rome:
        target += timedelta(days=1)
    delay = (target - now_rome).total_seconds()
    timer = threading.Timer(delay, _send_daily_report)
    timer.daemon = True
    timer.start()
    print(f"  📊 Daily report scheduled for {target.strftime('%Y-%m-%d %H:%M')} Rome ({delay:.0f}s)")


def start_server(port: int = None):
    """Start HTTP server for n8n webhook."""
    if port is None:
        port = int(os.environ.get("PORT", 8787))

    # Reload scheduled tasks from previous runs
    reload_scheduled_tasks()

    # Schedule daily report at 22:00 Rome time
    _schedule_daily_report()

    # Feature 2: Start background preprocessing thread for followed channels
    preprocess_thread = threading.Thread(target=_preprocess_followed_channels, daemon=True)
    preprocess_thread.start()
    print(f"  🔄 Background preprocessing started (every {PREPROCESS_INTERVAL_HOURS}h)")

    server = HTTPServer(("0.0.0.0", port), WebhookHandler)
    print(f"\n🚀 SARAh, l'unclock intelligence — server running on port {port}")
    print(f"   Version: 2026-04-13-v7 (cache + follow + preferences + comparison)")
    print(f"   Webhook URL: http://localhost:{port}/webhook")
    print(f"   Health check: http://localhost:{port}/health")
    print(f"\n   Waiting for WhatsApp messages...\n")
    server.serve_forever()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8787
        start_server(port)
    elif len(sys.argv) > 1 and sys.argv[1] == "msg":
        # Direct message processing from CLI: python youtube_agent.py msg "ultimi 3 video di Chase"
        message = " ".join(sys.argv[2:])
        process_whatsapp_message(message)
    else:
        # Default: start server
        print("Usage:")
        print("  python youtube_agent.py serve [port]   — Start webhook server")
        print("  python youtube_agent.py msg \"...\"     — Process a single message")
        print()
        print("Starting server on default port 8787...")
        start_server()

