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

# Known creators mapping — extend as needed
KNOWN_CREATORS = {
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
}

# Conversation memory for follow-up intent (in-memory, resets on restart)
_conversation_history = {}

# Pending confirmation requests: sender -> {intent, params, videos, timestamp}
_pending_requests = {}

# Message log file
MESSAGE_LOG = os.path.join(OUTPUT_DIR, "message_log.jsonl")


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

INTENT_SYSTEM_PROMPT = """Sei il parser di comandi per SARAh, l'unclock intelligence — un sistema che analizza video YouTube e produce briefing audio su WhatsApp.

Dato un messaggio in linguaggio naturale, identifica l'intent e i parametri.

INTENTS DISPONIBILI:

1. channel_analysis — L'utente vuole un riassunto degli ultimi N video di un creator specifico.
   Params: creator (nome), n (numero video, default 5), keywords (lista filtri topic, opzionale)
   Esempi: "ultimi 5 video di Chase", "cosa ha detto Cole Medin su n8n", "riassumimi gli ultimi 3 video di Liam Ottley"

2. single_video — L'utente manda un URL YouTube specifico da analizzare.
   Params: url (URL del video)
   Esempi: "analizza questo https://youtube.com/watch?v=abc123", un messaggio che contiene solo un URL YouTube

3. topic_search — L'utente vuole scoprire chi parla di un certo topic, eventualmente filtrato per paese/periodo.
   Params: topic (argomento), country (opzionale, codice paese), period (opzionale: "week", "month", "today")
   Esempi: "chi parla di MCP servers?", "creator italiani che parlano di AI agents questa settimana"

4. multi_creator — L'utente vuole confrontare cosa dicono diversi creator su un topic.
   Params: creators (lista nomi), topic (argomento), n (video per creator, default 3)
   Esempi: "confronta Chase e Cole Medin su Claude Code", "cosa dicono Chase, Liam e Cole su AI agents"

5. follow_up — L'utente vuole approfondire qualcosa da un'analisi precedente.
   Params: question (la domanda di approfondimento)
   Esempi: "approfondisci il punto sugli MCP servers", "dimmi di più sul secondo video", "cosa intendeva Chase con..."

6. scheduling — L'utente vuole impostare un aggiornamento ricorrente.
   Params: creator (nome), frequency ("daily", "weekly", "monthly"), day (opzionale: "monday", etc.), keywords (opzionale)
   Esempi: "aggiornami ogni lunedì sui video di Chase", "briefing settimanale su Cole Medin ogni venerdì"

7. news_search — L'utente vuole le novità su un topic senza specificare un creator.
   Params: topic (argomento), period ("today", "week", "month", default "week"), n (max video, default 5)
   Esempi: "novità su Claude Code", "cosa c'è di nuovo su AI agents questa settimana", "ultimi video su n8n"

8. greeting — L'utente saluta SARAh o chiede come sta, o si presenta.
   Params: name (nome dell'utente, se menzionato, opzionale)
   Esempi: "ciao SARAh", "buongiorno", "come stai?", "hey", "ciao come va?", "buonasera SARAh"

9. confirmation — L'utente conferma o rifiuta una richiesta precedente.
   Params: confirmed (boolean — true se conferma, false se rifiuta)
   Esempi: "sì", "ok", "procedi", "vai", "confermo", "no", "annulla", "stop", "non quelli"

Rispondi ESCLUSIVAMENTE con un JSON valido, nient'altro:
{"intent": "nome_intent", "params": {...}, "confidence": 0.0-1.0}

Se non riesci a classificare, usa:
{"intent": "unknown", "params": {"raw_message": "..."}, "confidence": 0.0}"""


def parse_intent(message: str) -> dict:
    """Use Claude to parse a WhatsApp message into a structured intent."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=INTENT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": message}],
    )

    text = response.content[0].text.strip()
    # Extract JSON from response
    try:
        # Handle potential markdown wrapping
        if "```" in text:
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                text = match.group(1)
        return json.loads(text)
    except json.JSONDecodeError:
        return {"intent": "unknown", "params": {"raw_message": message}, "confidence": 0.0}


# ---------------------------------------------------------------------------
# YouTube helpers
# ---------------------------------------------------------------------------

def get_channel_videos(channel_url: str, max_videos: int = 50) -> list[VideoInfo]:
    """Use yt-dlp to list videos from a channel."""
    cmd = [
        "yt-dlp",
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
            "yt-dlp",
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
            "yt-dlp",
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

    # Step 2: fetch full metadata (gets channel, views, likes, upload_date)
    urls = [f"https://www.youtube.com/watch?v={vid_id}" for vid_id in video_ids]
    cmd2 = ["yt-dlp", "--dump-json", "--no-download"] + urls
    result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=300)

    videos = []
    skipped_date = 0
    for line in result2.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            data = json.loads(line)
            vid_upload_date = data.get("upload_date", "")
            # Post-fetch date filter as safety net (upload_date is YYYYMMDD)
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
        except json.JSONDecodeError:
            continue

    if upload_date:
        print(f"  🗓 Date filter: kept {len(videos)}, skipped {skipped_date} (before {upload_date})")

    # Sort by view count (most viewed first) so we return the most popular results
    videos.sort(key=lambda v: v.view_count, reverse=True)

    # Truncate to requested max_results
    videos = videos[:max_results]
    print(f"  ✓ Returning {len(videos)} videos (sorted by views, top: {videos[0].view_count if videos else 0})")
    return videos


def get_video_info(url: str) -> Optional[VideoInfo]:
    """Get info for a single video URL."""
    cmd = ["yt-dlp", "--dump-json", "--no-download", url]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return None
    data = json.loads(result.stdout)
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
        cmd = ["yt-dlp", "--flat-playlist", "--dump-json", "--no-download", "--playlist-end", "1", f"{test_url}/videos"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and result.stdout.strip():
            # Found! Add to known creators for future use
            KNOWN_CREATORS[key] = test_url
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


def summarize_with_claude(video: VideoInfo, transcript: str, creator: str) -> dict:
    """Send transcript to Claude and get structured dual-layer summary."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    max_chars = 60000
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n\n[...trascrizione troncata per lunghezza...]"

    date_formatted = video.upload_date
    if len(date_formatted) == 8:
        date_formatted = f"{date_formatted[:4]}-{date_formatted[4:6]}-{date_formatted[6:]}"

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


def generate_audio(text: str, output_path: str, voice: str = "it-IT-DiegoNeural") -> bool:
    """Generate OGG Opus audio from text using Edge TTS."""
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
        print(f"  🔊 Audio generato: {output_path} ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"  ⚠ Audio generation failed: {e}")
        return False


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

def process_videos(videos: list[VideoInfo], creator_name: str, sender: str = None) -> list[dict]:
    """Analyze a list of videos: transcript → Claude → save markdown. Returns analyses."""
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
        result = summarize_with_claude(video, transcript, creator_name)
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


def format_confirmation_message(videos: list, est_minutes: int, label: str = "") -> str:
    """Format a confirmation message showing video titles, creator, views, likes + time estimate."""
    lines = ["*SARAh* — Ecco cosa ho trovato" + (f" per {label}" if label else "") + ":\n"]
    for i, v in enumerate(videos, 1):
        lines.append(f"{i}. *{v.title}*")
        creator_part = f"📺 {v.channel}" if v.channel else ""
        stats_parts = []
        if v.view_count:
            stats_parts.append(f"👁 {_format_count(v.view_count)}")
        if v.like_count:
            stats_parts.append(f"👍 {_format_count(v.like_count)}")
        meta_line = "   " + " | ".join(filter(None, [creator_part] + stats_parts))
        lines.append(meta_line)
        lines.append(f"   🔗 {v.url}")
        lines.append("")
    lines.append(f"⏱ Tempo stimato: ~{est_minutes} minuti")
    lines.append(f"\nVuoi che proceda? Rispondi *sì* o *no*.")
    return "\n".join(lines)


def store_pending_request(sender: str, intent: str, params: dict, videos: list):
    """Store a pending request waiting for user confirmation."""
    sender = _normalize_sender(sender)
    _pending_requests[sender] = {
        "intent": intent,
        "params": params,
        "videos": [v for v in videos],
        "timestamp": datetime.now().isoformat(),
    }


def execute_pending_request(sender: str):
    """Execute a previously confirmed pending request."""
    sender = _normalize_sender(sender)
    pending = _pending_requests.pop(sender, None)
    if not pending:
        send_whatsapp_text(sender, "⚠️ Non ho richieste in sospeso da confermare.")
        return

    intent = pending["intent"]
    params = pending["params"]
    videos = pending["videos"]

    mood = get_sarah_mood()
    est = estimate_minutes(len(videos))
    send_whatsapp_text(sender, f"{mood['emoji']} Perfetto! Ci lavoro subito.\n\n⏱ Tempo stimato: ~{est} minuti\n\nTi mando il briefing audio appena pronto.")

    creator_name = params.get("creator", params.get("label", "ricerca"))
    analyses = process_videos(videos, creator_name, sender)
    generate_and_send_briefing(analyses, sender, label=f"vo-{slugify(creator_name)}")


def handle_greeting(params: dict, sender: str):
    """Intent 8: Greet the user. Show mood only if they ask."""
    asks_mood = params.get("asks_mood", False)
    name = params.get("name", "")
    greeting_name = f" {name}" if name else ""

    hour = datetime.now().hour
    if hour < 12:
        time_greeting = "Buongiorno"
    elif hour < 18:
        time_greeting = "Buon pomeriggio"
    else:
        time_greeting = "Buonasera"

    lines = [f"*SARAh, l'unclock intelligence*\n"]
    lines.append(f"{time_greeting}{greeting_name}! Sono SARAh.")

    if asks_mood:
        mood = get_sarah_mood()
        lines.append(f"\nOggi a Milano: {mood['weather_desc']}, {mood['temp']}°C.")
        lines.append(f"Il mio umore? Sono {mood['mood']}! {mood['emoji']}\n")

    lines.append("\nCome posso aiutarti? Ecco cosa so fare:\n")
    lines.append("📹 Analisi canale — _\"ultimi 3 video di Chase\"_")
    lines.append("🔗 Video singolo — _manda un link YouTube_")
    lines.append("🔍 Cerca topic — _\"chi parla di MCP servers?\"_")
    lines.append("⚔️ Confronto — _\"confronta Chase e Cole su Claude Code\"_")
    lines.append("💬 Approfondimento — _\"approfondisci il punto su...\"_")
    lines.append("📅 Programmazione — _\"aggiornami ogni lunedì su Chase\"_")
    lines.append("📰 Novità — _\"novità su AI agents questa settimana\"_")

    send_whatsapp_text(sender, "\n".join(lines))


def handle_confirmation(params: dict, sender: str):
    """Intent 9: User confirms or rejects a pending request."""
    confirmed = params.get("confirmed", False)

    if confirmed:
        if sender in _pending_requests:
            execute_pending_request(sender)
        else:
            send_whatsapp_text(sender, "⚠️ Non ho richieste in sospeso. Mandami un comando!")
    else:
        _pending_requests.pop(sender, None)
        mood = get_sarah_mood()
        send_whatsapp_text(sender, f"{mood['emoji']} Ok, annullato! Mandami un'altra richiesta quando vuoi.")


def handle_channel_analysis(params: dict, sender: str):
    """Intent 1: Analyze last N videos from a creator."""
    creator_name = params.get("creator", "")
    n = params.get("n", 5)
    keywords = params.get("keywords", [])

    channel_url = resolve_creator(creator_name)
    if not channel_url:
        send_whatsapp_text(sender, f"❌ Non conosco il creator \"{creator_name}\". Prova con un URL YouTube o aggiungilo alla lista dei creator conosciuti.")
        return

    print(f"\n📡 Channel analysis: {creator_name} (last {n}, keywords={keywords})")

    all_videos = get_channel_videos(channel_url, max_videos=50)
    if keywords:
        all_videos = filter_videos_by_topic(all_videos, keywords)
    videos = all_videos[:n]

    if not videos:
        send_whatsapp_text(sender, f"⚠️ Nessun video trovato per {creator_name}.")
        return

    est = estimate_minutes(len(videos))
    msg = format_confirmation_message(videos, est, label=creator_name)
    send_whatsapp_text(sender, msg)
    store_pending_request(sender, "channel_analysis", {"creator": creator_name, "label": creator_name}, videos)


def handle_single_video(params: dict, sender: str):
    """Intent 2: Analyze a single video URL."""
    url = params.get("url", "")
    if not url:
        send_whatsapp_text(sender, "❌ Non ho trovato un URL YouTube valido nel messaggio.")
        return

    print(f"\n📡 Single video: {url}")

    video = get_video_info(url)
    if not video:
        send_whatsapp_text(sender, f"⚠️ Non riesco a ottenere info per questo video.")
        return

    est = estimate_minutes(1)
    msg = format_confirmation_message([video], est)
    send_whatsapp_text(sender, msg)
    store_pending_request(sender, "single_video", {"label": "video-singolo"}, [video])


def handle_topic_search(params: dict, sender: str):
    """Intent 3: Search YouTube for a topic."""
    topic = params.get("topic", "")
    country = params.get("country", "")
    period = params.get("period", "week")
    n = params.get("n", 5)

    query = topic
    if country:
        country_map = {"it": "italiano", "italia": "italiano", "us": "english", "uk": "english"}
        lang = country_map.get(country.lower(), country)
        query = f"{topic} {lang}"

    date_after = period_to_dateafter(period)
    print(f"\n📡 Topic search: \"{query}\" (period={period}, n={n})")

    videos = search_youtube(query, max_results=n, upload_date=date_after, period=period)
    if not videos:
        send_whatsapp_text(sender, f"⚠️ Nessun video trovato per \"{topic}\".")
        return

    est = estimate_minutes(len(videos))
    msg = format_confirmation_message(videos, est, label=f"ricerca \"{topic}\"")
    send_whatsapp_text(sender, msg)
    store_pending_request(sender, "topic_search", {"label": f"search-{slugify(topic)}"}, videos)


def handle_multi_creator(params: dict, sender: str):
    """Intent 4: Compare multiple creators on a topic."""
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

    est = estimate_minutes(len(all_videos))
    label = f"confronto {', '.join(creators)}"
    msg = format_confirmation_message(all_videos, est, label=label)
    send_whatsapp_text(sender, msg)
    store_pending_request(sender, "multi_creator", {"label": f"multi-{slugify(topic)}", "creator": f"multi-{slugify(topic)}"}, all_videos)


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


def handle_scheduling(params: dict, sender: str):
    """Intent 6: Set up scheduled briefing (stores config, actual scheduling done via n8n)."""
    creator_name = params.get("creator", "")
    frequency = params.get("frequency", "weekly")
    day = params.get("day", "monday")
    keywords = params.get("keywords", [])

    channel_url = resolve_creator(creator_name) if creator_name else None

    # Save schedule to a JSON file that n8n can read
    schedules_file = Path(OUTPUT_DIR) / "schedules.json"
    schedules = []
    if schedules_file.exists():
        schedules = json.loads(schedules_file.read_text())

    new_schedule = {
        "id": f"sched-{len(schedules)+1}",
        "creator": creator_name,
        "channel_url": channel_url,
        "frequency": frequency,
        "day": day,
        "keywords": keywords,
        "recipient": sender,
        "created_at": datetime.now().isoformat(),
        "active": True,
    }
    schedules.append(new_schedule)
    schedules_file.write_text(json.dumps(schedules, indent=2))

    freq_map = {"daily": "ogni giorno", "weekly": f"ogni {day}", "monthly": "ogni mese"}
    freq_text = freq_map.get(frequency, frequency)
    topic_text = f" su {', '.join(keywords)}" if keywords else ""

    send_whatsapp_text(sender,
        f"✅ Programmato! Riceverai un briefing su {creator_name}{topic_text} {freq_text}.\n\n"
        f"Per gestire le programmazioni, scrivi \"le mie programmazioni\".")
    print(f"  📅 Schedule saved: {new_schedule['id']}")


def handle_news_search(params: dict, sender: str):
    """Intent 7: Search for news/latest videos on a topic (no specific creator)."""
    topic = params.get("topic", "")
    period = params.get("period", "week")
    n = params.get("n", 5)

    date_after = period_to_dateafter(period)
    print(f"\n📡 News search: \"{topic}\" (period={period}, n={n})")

    videos = search_youtube(topic, max_results=n, upload_date=date_after, period=period)
    if not videos:
        send_whatsapp_text(sender, f"⚠️ Nessun video recente trovato su \"{topic}\".")
        return

    est = estimate_minutes(len(videos))
    msg = format_confirmation_message(videos, est, label=f"novità su \"{topic}\"")
    send_whatsapp_text(sender, msg)
    store_pending_request(sender, "news_search", {"label": f"news-{slugify(topic)}", "creator": f"news-{slugify(topic)}"}, videos)


def handle_unknown(params: dict, sender: str):
    """Fallback for unrecognized messages — stay silent so CArL can handle it."""
    # Don't respond: message is probably for CArL, not SARAh.
    # CArL gets all messages in parallel and will handle non-intelligence ones.
    print(f"  ℹ️ Unknown intent — SARAh stays silent (CArL may handle it)")


# Intent routing
INTENT_HANDLERS = {
    "channel_analysis": handle_channel_analysis,
    "single_video": handle_single_video,
    "topic_search": handle_topic_search,
    "multi_creator": handle_multi_creator,
    "follow_up": handle_follow_up_intent,
    "scheduling": handle_scheduling,
    "news_search": handle_news_search,
    "greeting": handle_greeting,
    "confirmation": handle_confirmation,
    "unknown": handle_unknown,
}


# ---------------------------------------------------------------------------
# Main entry point: process a WhatsApp message
# ---------------------------------------------------------------------------

def _is_short_greeting(lower_msg: str) -> bool:
    """True only for SHORT greeting-only messages (not commands that start with 'ciao')."""
    # Pure greeting words (message is ONLY this)
    pure_greetings = ["ciao", "hey", "salve", "buongiorno", "buonasera", "buon pomeriggio",
                      "sarah", "sara", "ciao sarah", "ciao sara", "hey sarah", "hey sara",
                      "buongiorno sarah", "buongiorno sara", "buonasera sarah", "buonasera sara",
                      "ciao come stai", "come stai", "come va", "come stai sarah", "come stai sara",
                      "come va sarah", "come sta sarah", "come sta sara", "tutto bene",
                      "ciao sarah come stai", "ciao sara come stai", "ciao come va",
                      "ciao sarah come va", "buongiorno come stai", "buonasera come stai"]
    if lower_msg in pure_greetings:
        return True
    # Very short message with sarah/sara in it — likely a greeting, not a command
    sarah_words = ["sarah", "sara"]
    if any(s in lower_msg for s in sarah_words) and len(lower_msg) < 30:
        # But NOT if it also contains command-like words
        command_words = ["video", "ultimi", "analizza", "cerca", "novità", "confronta", "riassumi", "manda", "briefing"]
        if not any(c in lower_msg for c in command_words):
            return True
    return False


def process_whatsapp_message(message: str, sender: str = None):
    """Main entry: parse intent from a message and execute the right handler."""
    sender = _normalize_sender(sender or WHATSAPP_RECIPIENT)

    print(f"\n{'='*60}")
    print(f"SARAh, l'unclock intelligence — Incoming message")
    print(f"{'='*60}")
    print(f"From: {sender}")
    print(f"Message: {message}")
    print()

    lower_msg = message.lower().strip()

    # --- Step 1: Expire stale pending requests (older than 30 min) ---
    if sender in _pending_requests:
        pending_ts = _pending_requests[sender].get("timestamp", "")
        try:
            if pending_ts and (datetime.now() - datetime.fromisoformat(pending_ts)).total_seconds() > 1800:
                print(f"  ⏰ Pending request expired, removing")
                _pending_requests.pop(sender, None)
        except Exception:
            pass

    # --- Step 2: Fast pattern matching (no API call) ---

    # 2a. Pure greeting (ONLY if message is short and greeting-only)
    if _is_short_greeting(lower_msg):
        asks_mood = any(q in lower_msg for q in ["come stai", "come va", "come ti senti", "tutto bene", "che umore", "come sta sarah", "come sta sara"])
        intent_result = {"intent": "greeting", "params": {"asks_mood": asks_mood}, "confidence": 1.0}

    # 2b. Confirmation — only if pending request exists
    elif sender in _pending_requests:
        confirm_yes = ["sì", "si", "yes", "ok", "vai", "procedi", "confermo", "conferma",
                       "certo", "perfetto", "fallo", "manda", "si grazie", "ok vai",
                       "si perfetto", "vai pure", "procedi pure"]
        confirm_no = ["no", "annulla", "stop", "cancella", "non quelli", "ferma",
                      "lascia stare", "niente", "no grazie", "non serve"]
        if any(lower_msg == w or lower_msg.startswith(w + " ") or lower_msg.startswith(w + ",") or lower_msg.startswith(w + "!") for w in confirm_yes):
            intent_result = {"intent": "confirmation", "params": {"confirmed": True}, "confidence": 1.0}
        elif any(lower_msg == w or lower_msg.startswith(w + " ") or lower_msg.startswith(w + ",") or lower_msg.startswith(w + "!") for w in confirm_no):
            intent_result = {"intent": "confirmation", "params": {"confirmed": False}, "confidence": 1.0}
        else:
            # New command while pending — discard old pending request
            print(f"  ⚠ New command while pending request active — discarding old request")
            _pending_requests.pop(sender, None)
            print("🧠 Parsing intent...")
            intent_result = parse_intent(message)

    # 2c. Direct YouTube URL
    elif re.search(r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]+)', message) and len(message.strip()) < 200:
        yt_url_match = re.search(r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]+)', message)
        intent_result = {"intent": "single_video", "params": {"url": yt_url_match.group(1)}, "confidence": 1.0}

    # 2d. Everything else → Claude intent parsing
    else:
        print("🧠 Parsing intent...")
        intent_result = parse_intent(message)

    intent = intent_result.get("intent", "unknown")
    params = intent_result.get("params", {})
    confidence = intent_result.get("confidence", 0)

    print(f"   Intent: {intent} (confidence: {confidence})")
    print(f"   Params: {params}")

    handler = INTENT_HANDLERS.get(intent, handle_unknown)
    handler(params, sender)

    log_message(sender, message, intent, params, outcome="handled")

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
        """Health check."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "service": "sarah-unclock-intelligence"}).encode())

    def log_message(self, format, *args):
        print(f"  [HTTP] {args[0]}" if args else "")


def start_server(port: int = None):
    """Start HTTP server for n8n webhook."""
    if port is None:
        port = int(os.environ.get("PORT", 8787))
    server = HTTPServer(("0.0.0.0", port), WebhookHandler)
    print(f"\n🚀 SARAh, l'unclock intelligence — server running on port {port}")
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
