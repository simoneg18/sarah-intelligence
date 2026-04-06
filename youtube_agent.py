#!/usr/bin/env python3
"""
SARAh, l'unclock intelligence — YouTube Agent
Receives commands via WhatsApp, parses intent, executes YouTube research,
and sends back voice briefings + source links.
SARAh è meteoropatica: il suo umore dipende dal meteo di Milano.
"""

import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
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
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "nPczCjzI2devNBz1zQrb")  # Brian - Deep, Resonant and Comforting

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
            "preferences": {},  # user-specific prefs
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

8. feedback — L'utente segnala un PROBLEMA o ERRORE di SARAh.
   Params: complaint (descrizione), raw_message (messaggio originale)

9. not_youtube — Il messaggio NON riguarda video YouTube. Include saluti puri, domande generiche, conversazione.
   Params: raw_message (messaggio originale), is_greeting (boolean — true se è un saluto/presentazione)

REGOLE:
- Se il messaggio riguarda video, creator, trascrizioni, riassunti video, analisi video, YouTube → scegli l'azione YouTube appropriata (1-7).
- Se l'utente si LAMENTA di qualcosa che SARAh ha fatto male → feedback.
- TUTTO il resto (saluti, domande non-YouTube, conversazione) → not_youtube.
- In caso di dubbio, se c'è QUALSIASI riferimento a video/YouTube → trattalo come richiesta YouTube.
- "topic_search" e "news_search" sono simili: usa news_search quando l'utente dice "novità"/"news"/"ultimi", topic_search per il resto.

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

    # Recent interactions — help disambiguate short messages
    recent = memory.get("recent_interactions", [])
    if recent:
        last_3 = recent[-3:]
        history = []
        for r in last_3:
            msg = r.get("message", "")[:80]
            action = r.get("intent", "?")
            history.append(f"  ({action}) {msg}")
        parts.append("Ultime interazioni:\n" + "\n".join(history))

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

    if not parts:
        return ""
    return "\n\nCONTESTO APPRESO (usa per decisioni migliori, NON menzionare all'utente):\n" + "\n".join(parts)


def route_message(message: str, sender: str = None) -> dict:
    """Use Claude to route a WhatsApp message: YouTube action or not_youtube.
    Injects user context and learned patterns for smarter routing."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    # Build dynamic context from learning
    learning_context = _build_learning_context(sender) if sender else ""
    system_prompt = ROUTER_SYSTEM_PROMPT + learning_context

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

SYSTEM_PROMPT = """Sei un analista strategico per unclock, una startup italiana early-stage che costruisce agenti AI su misura per freelance e PMI usando n8n + Claude come stack tecnologico.

unclock vende tempo liberato, non tecnologia. I target sono: freelance marketing/PM, head hunter freelance, PMI italiane. Il modello parte da €1.500/anno per i freelance.

Quando analizzi un video, produci DUE layer di analisi:

## LAYER 1 — KNOWLEDGE (per imparare)
Spiega in modo chiaro e pratico cosa viene trattato nel video. Scrivi per qualcuno che vuole capire a fondo la tecnologia/concetto, non per un principiante assoluto ma nemmeno per un esperto.

Struttura:
- **Cos'è e come funziona**: spiegazione chiara del concetto/tool/tecnica principale
- **Concetti chiave**: i 3-5 punti fondamentali spiegati
- **Perché è rilevante adesso**: contesto di mercato/timing
- **Approfondimenti**: se qualcuno volesse andare più a fondo, cosa dovrebbe cercare/studiare

## LAYER 2 — BUSINESS (per unclock)
Analizza il contenuto dal punto di vista di unclock: cosa possiamo costruire, vendere, replicare.

Struttura:
- **Cosa automatizza concretamente**: quale processo/workflow viene mostrato
- **Replicabilità in n8n + Claude**: si può ricostruire nel nostro stack? Come? Complessità stimata (ore)
- **Target ideale**: a chi lo venderemmo tra i nostri segmenti (freelance marketing, PM, head hunter, PMI)
- **Come si confeziona**: nome prodotto, promessa, fascia di prezzo suggerita
- **Segnali di mercato**: trend, domanda, concorrenza emersi dal video

Rispondi SEMPRE in italiano. Sii concreto, diretto, zero fuffa."""

USER_PROMPT_TEMPLATE = """Analizza questa trascrizione del video "{title}" di {creator} ({url}, pubblicato il {date}).

TRASCRIZIONE:
{transcript}

Produci l'analisi dual-layer come da istruzioni."""

USER_PROMPT_TEMPLATE_FOCUSED = """Analizza questa trascrizione del video "{title}" di {creator} ({url}, pubblicato il {date}).

RICHIESTA SPECIFICA DELL'UTENTE:
{user_focus}

TRASCRIZIONE:
{transcript}

IMPORTANTE: Rispondi focalizzandoti SPECIFICAMENTE su ciò che l'utente ha chiesto. La richiesta dell'utente ha la priorità.
Produci comunque l'analisi dual-layer, ma concentrati sulla richiesta specifica."""


def summarize_with_claude(video: VideoInfo, transcript: str, creator: str, user_focus: str = "") -> dict:
    """Send transcript to Claude and get structured dual-layer summary.
    If user_focus is provided, the analysis prioritizes that specific request."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    max_chars = 60000
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n\n[...trascrizione troncata per lunghezza...]"

    date_formatted = video.upload_date
    if len(date_formatted) == 8:
        date_formatted = f"{date_formatted[:4]}-{date_formatted[4:6]}-{date_formatted[6:]}"

    if user_focus:
        user_msg = USER_PROMPT_TEMPLATE_FOCUSED.format(
            title=video.title,
            creator=creator,
            url=video.url,
            date=date_formatted,
            user_focus=user_focus,
            transcript=transcript,
        )
    else:
        user_msg = USER_PROMPT_TEMPLATE.format(
            title=video.title,
            creator=creator,
            url=video.url,
            date=date_formatted,
            transcript=transcript,
        )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    return {
        "full_text": response.content[0].text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


# ---------------------------------------------------------------------------
# Follow-up with conversation memory
# ---------------------------------------------------------------------------

def handle_follow_up(sender: str, question: str) -> str:
    """Answer a follow-up question based on previous analyses for this user."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    history = _conversation_history.get(sender, [])
    if not history:
        return "Non ho analisi precedenti a cui fare riferimento. Mandami prima un video o un canale da analizzare!"

    # Build context from recent analyses
    context = "\n\n".join([
        f"--- {item['title']} ---\n{item['summary']}"
        for item in history[-5:]  # last 5 analyses
    ])

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system="""Sei un analista di unclock. Ti viene data una domanda di approfondimento e il contesto delle analisi precedenti.
Rispondi in modo diretto e approfondito. Se la domanda riguarda un video specifico, concentrati su quello. Rispondi in italiano.""",
        messages=[{"role": "user", "content": f"CONTESTO ANALISI PRECEDENTI:\n{context}\n\nDOMANDA:\n{question}"}],
    )

    return response.content[0].text


# ---------------------------------------------------------------------------
# Combined Voice Over
# ---------------------------------------------------------------------------

VOICE_SYSTEM_PROMPT = """Sei un collega di Simone e Fede, i founder di unclock. Stai registrando un audio che ascolteranno mentre fanno altro — cucinare, camminare, guidare.

Il tuo compito: prendere le analisi di più video e trasformarle in UN UNICO audio fluido e naturale.

Regole fondamentali:
- Tono conversazionale, come una telefonata tra colleghi. Dai del tu.
- Frasi corte. Niente subordinate lunghe. Pause naturali.
- Per ogni video: spiega prima PERCHÉ è rilevante, poi COSA dice, poi COSA POSSIAMO FARCI NOI come unclock
- Transizioni naturali tra un video e l'altro ("Passiamo al secondo video..." / "L'altro contenuto interessante è...")
- Chiudi con un recap delle 2-3 azioni più importanti da fare subito
- NON usare markdown, bullet point, asterischi, numeri di lista o formattazione — è testo PURO da leggere ad alta voce
- Niente emoji
- Delimita lo script tra <!-- VOICE_START --> e <!-- VOICE_END -->
- Lunghezza: circa 200-250 parole PER VIDEO analizzato
- IMPORTANTE: Chiudi SEMPRE con una sezione "TREND E SEGNALI" dove identifichi pattern ricorrenti che emergono da PIÙ video. Un trend è valido solo se confermato da almeno 2 video diversi. Spiega perché quel trend è rilevante per unclock e cosa dovreste fare al riguardo.

Rispondi in italiano."""

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
                          followup: dict = None) -> Optional[str]:
    """Generate a voice over script. Supports multi-video, single video, and follow-up."""
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

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=VOICE_SYSTEM_PROMPT,
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


def generate_audio_edge_tts(text: str, output_path: str, voice: str = "it-IT-DiegoNeural") -> bool:
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


def generate_audio(text: str, output_path: str, voice: str = "it-IT-DiegoNeural") -> bool:
    """Generate OGG Opus audio — ElevenLabs primary, edge-tts fallback."""
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
    result = send_whatsapp_message(recipient, {"type": "text", "text": {"body": text}})
    print(f"  📤 Send result: {result}")
    return result


def send_whatsapp_audio(recipient: str, audio_path: str) -> bool:
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

def process_videos(videos: list[VideoInfo], creator_name: str, sender: str = None, user_focus: str = "") -> list[dict]:
    """Analyze a list of videos: transcript → Claude → save markdown. Returns analyses.
    user_focus: the original user request, used to focus the analysis."""
    creator_slug = slugify(creator_name)
    processed = []
    video_analyses = []

    for i, video in enumerate(videos, 1):
        print(f"\n--- Video {i}/{len(videos)}: {video.title} ---")

        print("  📝 Fetching transcript...")
        transcript = get_transcript(video.video_id)
        if not transcript:
            print("  ⏭ Skipping (no transcript)")
            continue

        print(f"  📝 Transcript: {len(transcript)} chars")
        print("  🤖 Analyzing with Claude...")
        result = summarize_with_claude(video, transcript, creator_name, user_focus=user_focus)
        print(f"  🤖 Done ({result['input_tokens']} in / {result['output_tokens']} out tokens)")

        save_markdown(video, result["full_text"], creator_slug)

        date_fmt = video.upload_date
        if len(date_fmt) == 8:
            date_fmt = f"{date_fmt[:4]}-{date_fmt[4:6]}-{date_fmt[6:]}"

        analysis = {
            "title": video.title,
            "url": video.url,
            "date": date_fmt,
            "video_id": video.video_id,
            "summary": result["full_text"],
        }
        processed.append(analysis)
        video_analyses.append(analysis)

        # Store in conversation memory for follow-ups
        if sender:
            if sender not in _conversation_history:
                _conversation_history[sender] = []
            _conversation_history[sender].append(analysis)

    if processed:
        update_index(creator_slug, processed)

    return video_analyses


def generate_and_send_briefing(video_analyses: list[dict], recipient: str, label: str = "briefing"):
    """Generate voice over and send to WhatsApp."""
    if not video_analyses:
        send_whatsapp_text(recipient, "⚠️ Nessun video trovato o nessuna trascrizione disponibile.")
        return

    print(f"\n--- Generating voice over ({len(video_analyses)} videos) ---")

    if len(video_analyses) == 1:
        voice_script = generate_voice_script(single={
            "title": video_analyses[0]["title"],
            "url": video_analyses[0]["url"],
            "summary": video_analyses[0]["summary"],
        })
    else:
        voice_script = generate_voice_script(video_analyses=video_analyses)

    if voice_script:
        audio_path = str(Path(OUTPUT_DIR) / f"{label}.ogg")
        print("  🔊 Generating audio...")
        if generate_audio(voice_script, audio_path):
            send_full_briefing(recipient, video_analyses, audio_path)
        else:
            # Fallback: send text summary
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

    user_focus = params.get("_original_message", "")
    analyses = process_videos(videos, creator_name, sender, user_focus=user_focus)
    generate_and_send_briefing(analyses, sender, label=f"vo-{slugify(creator_name)}")
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

    user_focus = params.get("focus", params.get("_original_message", ""))
    analyses = process_videos([video], "video-singolo", sender, user_focus=user_focus)
    generate_and_send_briefing(analyses, sender, label="vo-video-singolo")
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

    analyses = process_videos(videos, f"search-{slugify(topic)}", sender, user_focus=original_request)
    generate_and_send_briefing(analyses, sender, label=f"vo-search-{slugify(topic)}")
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

    user_focus = params.get("_original_message", "")
    analyses = process_videos(all_videos, f"multi-{slugify(topic)}", sender, user_focus=user_focus)
    generate_and_send_briefing(analyses, sender, label=f"vo-multi-{slugify(topic)}")
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
            videos = get_channel_videos(channel_url, max_results=n)
            if keywords:
                videos = [v for v in videos if any(k.lower() in (v.title + v.description).lower() for k in keywords)]
            if not videos:
                send_whatsapp_text(sender, f"⚠️ Task programmato: nessun video trovato per {creator_name}")
                return
            label = slugify(creator_name)
            send_whatsapp_text(sender, f"⏰ *Task programmato in esecuzione!*\nAnalizzo {len(videos)} video di {creator_name}...")
            process_videos(videos, label=label, creator=creator_name, sender=sender)

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
            process_videos(videos, label=label, creator=label, sender=sender)

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

    now = datetime.now()
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

    # Calculate when to fire
    now = datetime.now()
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
                target = datetime.strptime(schedule_date, "%Y-%m-%d").replace(hour=h, minute=m, second=0)
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

    send_whatsapp_text(sender,
        f"✅ *Programmato!*\n\n"
        f"📺 {n} video di {subject}\n"
        f"📅 {freq_text}\n\n"
        f"Riceverai il briefing audio automaticamente! 🎙️")
    print(f"  📅 Schedule saved: {task['id']} — fires at {target.isoformat()} (in {delay:.0f}s)")


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

    analyses = process_videos(videos, f"news-{slugify(topic)}", sender, user_focus=original_request)
    generate_and_send_briefing(analyses, sender, label=f"vo-news-{slugify(topic)}")
    if analyses:
        _save_learned_query(params.get("_original_message", ""), "news_search", params, len(analyses))


def handle_not_youtube(params: dict, sender: str):
    """Non-YouTube message: polite decline with examples, or greeting."""
    raw_message = params.get("raw_message", params.get("_original_message", ""))
    is_greeting = params.get("is_greeting", False)

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
    if recent_logs:
        last_log = recent_logs[-1]
        last_action = last_log.get("intent", "unknown")
        last_params = last_log.get("params", {})
    _save_learned_error(complaint, last_action, last_params, "user_complaint", complaint[:200])


# Action routing
ACTION_HANDLERS = {
    "channel_analysis": handle_channel_analysis,
    "single_video": handle_single_video,
    "topic_search": handle_topic_search,
    "multi_creator": handle_multi_creator,
    "follow_up": handle_follow_up_intent,
    "scheduling": handle_scheduling,
    "news_search": handle_news_search,
    "feedback": handle_feedback,
    "not_youtube": handle_not_youtube,
}


# ---------------------------------------------------------------------------
# Main entry point: process a WhatsApp message
# ---------------------------------------------------------------------------

def process_whatsapp_message(message: str, sender: str = None):
    """Main entry: route message and execute immediately. No confirmation needed."""
    sender = _normalize_sender(sender or WHATSAPP_RECIPIENT)

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

    now = datetime.now()
    reloaded = 0
    for task in schedules:
        if not task.get("active", True):
            continue
        fire_at_str = task.get("fire_at", "")
        frequency = task.get("frequency", "once")

        try:
            fire_at = datetime.fromisoformat(fire_at_str)
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


def start_server(port: int = None):
    """Start HTTP server for n8n webhook."""
    if port is None:
        port = int(os.environ.get("PORT", 8787))

    # Reload scheduled tasks from previous runs
    reload_scheduled_tasks()

    server = HTTPServer(("0.0.0.0", port), WebhookHandler)
    print(f"\n🚀 SARAh, l'unclock intelligence — server running on port {port}")
    print(f"   Version: 2026-04-06-v6 (youtube-first-router)")
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

