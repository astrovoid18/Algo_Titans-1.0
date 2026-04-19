"""
omni — memory OS for your context
"""

import time
import uuid
import datetime
import re
import itertools
import os
import shutil
import json
import urllib.request
import urllib.error
from typing import Optional

import chromadb
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.padding import Padding
from rich.live import Live
from rich import box

from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter

# ═══════════════════════════════════════════════════════════════
#  CONFIG  (Gemini key — JSON only, never written to Chroma)
#  Merges: ./omni_config.json (next to this file) then ~/.omni_config.json
#          later files override earlier keys.
# ═══════════════════════════════════════════════════════════════

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_CONFIG_PATH = os.path.join(_CONFIG_DIR, "omni_config.json")
CONFIG_PATH = os.path.expanduser("~/.omni_config.json")


def load_config() -> dict:
    cfg: dict = {}
    for path in (LOCAL_CONFIG_PATH, CONFIG_PATH):
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg

def save_config(cfg: dict):
    os.makedirs(os.path.dirname(CONFIG_PATH) or ".", exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def get_api_key() -> Optional[str]:
    return load_config().get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")

def set_api_key(key: str):
    cfg = load_config()
    cfg["gemini_api_key"] = key
    save_config(cfg)


# ═══════════════════════════════════════════════════════════════
#  GEMINI  (stdlib only — no SDK needed)
# ═══════════════════════════════════════════════════════════════

# Override with env GEMINI_MODEL (e.g. gemini-2.5-flash, gemini-1.5-flash)
def _gemini_model_id() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip() or "gemini-2.0-flash"


def _gemini_url(api_key: str) -> str:
    mid = _gemini_model_id()
    return (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{mid}:generateContent?key={api_key}"
    )


# Gemini safety: looser thresholds so document extraction is not blanked as often
_GEMINI_SAFETY = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
]

EXTRACT_PROMPT = """\
You are a text extraction and cleaning assistant.

The user has provided raw text — it could be a paste from a document, webpage, PDF, or notes.

Your job:
1. Extract all meaningful, factual, and informational content.
2. Remove any navigation text, page numbers, headers/footers, ads, markdown artifacts, HTML tags.
3. Rewrite the content as clean, plain prose sentences.
4. Do NOT add your own opinions or summaries. Preserve the original facts.
5. Output ONLY the cleaned plain-text sentences, one per line. No bullet points, no numbering, no headings.

Raw input:
\"\"\"
{text}
\"\"\"
"""


def _gemini_response_text(data: dict) -> str:
    """Pull plain text from a generateContent JSON body; raise with a clear message on failure."""
    err = data.get("error")
    if err:
        raise ValueError(err.get("message", str(err)))

    pf = data.get("promptFeedback") or {}
    br = pf.get("blockReason")
    if br:
        raise ValueError(f"Prompt blocked ({br}).")

    cands = data.get("candidates") or []
    if not cands:
        raise ValueError("No candidates in API response (empty or blocked generation).")

    c0 = cands[0]
    reason = c0.get("finishReason")
    content = c0.get("content") or {}
    parts = content.get("parts") or []

    texts: list[str] = []
    for p in parts:
        t = p.get("text")
        if t:
            texts.append(t)

    if texts:
        return "\n".join(texts)

    if reason == "MAX_TOKENS":
        raise ValueError(
            "Output was truncated (MAX_TOKENS). Try a shorter input or set GEMINI_MODEL / larger paste limits."
        )
    if reason and reason not in ("STOP", "FINISH_REASON_UNSPECIFIED", None):
        raise ValueError(f"Model returned no text (finishReason={reason}).")
    raise ValueError("Model returned no text in parts.")


def _sentences_from_extract_output(raw_out: str) -> list[str]:
    """Normalize Gemini output into discrete sentences (handles bullets and prose)."""
    raw_out = (raw_out or "").strip()
    if not raw_out:
        return []

    lines: list[str] = []
    for line in raw_out.splitlines():
        s = line.strip()
        if not s:
            continue
        s = re.sub(r"^(?:[-*•]+\s*|\d+[\).]\s+)", "", s).strip()
        if len(s) >= 2:
            lines.append(s)

    if len(lines) <= 1 and len(raw_out) > 120:
        blob = re.sub(r"\s+", " ", raw_out)
        alt = [p.strip() for p in re.split(r"(?<=[.!?])\s+", blob) if len(p.strip()) >= 2]
        if len(alt) > len(lines):
            lines = alt

    return lines


def gemini_extract(raw_text: str, api_key: str) -> list[str]:
    """
    Send raw text to Gemini, get back clean plain sentences.
    Returns a list of sentence strings.
    """
    key = (api_key or "").strip()
    if not key:
        raise ValueError("Missing API key.")

    prompt = EXTRACT_PROMPT.format(text=raw_text[:12000])  # cap to avoid token overflow
    payload = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 8192,
            },
            "safetySettings": _GEMINI_SAFETY,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    url = _gemini_url(key)
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    raw_out = _gemini_response_text(data)
    sentences = _sentences_from_extract_output(raw_out)
    if not sentences:
        raise ValueError("Model returned no extractable sentences after parsing.")
    return sentences


# ═══════════════════════════════════════════════════════════════
#  ENGINE
# ═══════════════════════════════════════════════════════════════

ALIAS_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".omni_aliases.json")


def _load_aliases() -> dict:
    if not os.path.exists(ALIAS_DB):
        return {}
    try:
        with open(ALIAS_DB, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_aliases(aliases: dict):
    with open(ALIAS_DB, "w", encoding="utf-8") as f:
        json.dump(aliases, f, indent=2)


class MemoryEngine:
    def __init__(self, db_path: str = "./.omni_trunk"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="memory")

    @property
    def count(self) -> int:
        return self.collection.count()

    def store(self, text: str, tags: list = None, starred: bool = False, source: str = None) -> str:
        uid = f"MEM-{uuid.uuid4().hex[:6].upper()}"
        ts = datetime.datetime.now().isoformat()
        meta = {"ts": ts, "starred": starred}
        if tags:
            meta["tags"] = json.dumps(tags)
        if source:
            meta["source"] = source
        self.collection.add(
            ids=[uid],
            documents=[text],
            metadatas=[meta]
        )
        return uid

    def update(self, uid: str, text: str = None, tags: list = None, starred: bool = None) -> bool:
        try:
            existing = self.collection.get(ids=[uid])
            if not existing["ids"]:
                return False
            current_meta = existing["metadatas"][0]
            new_meta = dict(current_meta)
            if starred is not None:
                new_meta["starred"] = starred
            if tags is not None:
                new_meta["tags"] = json.dumps(tags)
            self.collection.update(
                ids=[uid],
                documents=[text] if text else None,
                metadatas=[new_meta]
            )
            return True
        except Exception:
            return False

    def recall(
        self,
        query: str,
        n: int = 5,
        tags: list = None,
        starred_only: bool = False,
        after: datetime.datetime = None,
        before: datetime.datetime = None,
        fuzzy: bool = False,
    ) -> tuple[list[str], list[dict]]:
        if self.count == 0:
            return [], []

        where = {}
        if starred_only:
            where["starred"] = True
        if tags:
            for t in tags:
                where["tags"] = {"$contains": t}
        if after or before:
            if after:
                where["ts"] = {"$gte": after.isoformat()}
            if before:
                where.setdefault("ts", {})["$lte"] = before.isoformat()

        query_lower = query.lower().strip()
        has_tags = bool(tags)

        if has_tags or fuzzy:
            n_results = min(n, self.count)
            all_ids, all_docs, all_metas = self.list_all()
            if has_tags:
                filtered = []
                for uid, doc, meta in zip(all_ids, all_docs, all_metas):
                    mem_tags = json.loads(meta.get("tags", "[]"))
                    if any(t in mem_tags for t in tags):
                        filtered.append((uid, doc, meta))
                all_ids, all_docs, all_metas = zip(*filtered) if filtered else ([], [], [])
            if fuzzy and query_lower:
                query_lower = query.lower()
                scored = []
                for uid, doc, meta in zip(all_ids, all_docs, all_metas):
                    score = 0
                    q_words = set(query_lower.split())
                    doc_words = set(doc.lower().split())
                    intersection = q_words & doc_words
                    score = len(intersection)
                    if query_lower in doc.lower():
                        score += 10
                    for qw in q_words:
                        if qw in doc.lower():
                            score += 5
                    scored.append((uid, doc, meta, score))
                scored.sort(key=lambda x: x[3], reverse=True)
                filtered = scored[:n_results]
                docs = [x[1] for x in filtered]
                metas = [x[2] for x in filtered]
            else:
                docs = list(all_docs[:n_results])
                metas = list(all_metas[:n_results])
            if not docs:
                docs, metas = [], []
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n, self.count),
                where=[where] if where else None,
            )
            docs = results["documents"][0]
            metas = results["metadatas"][0]

        return docs, metas

    def get_by_id(self, uid: str) -> tuple[str, dict]:
        try:
            existing = self.collection.get(ids=[uid])
            if not existing["ids"]:
                return "", {}
            return existing["documents"][0], existing["metadatas"][0]
        except Exception:
            return "", {}

    def add_related(self, uid: str, related_uid: str):
        try:
            doc, meta = self.get_by_id(uid)
            if not doc:
                return False
            existing_related = json.loads(meta.get("related", "[]"))
            if related_uid not in existing_related:
                existing_related.append(related_uid)
                self.collection.update(
                    ids=[uid],
                    metadatas=[{"related": json.dumps(existing_related)}]
                )
            return True
        except Exception:
            return False

    def get_related(self, uid: str) -> list:
        try:
            _, meta = self.get_by_id(uid)
            return json.loads(meta.get("related", "[]"))
        except Exception:
            return []

    def list_all(self, starred_only: bool = False) -> tuple[list, list, list]:
        where = {"starred": True} if starred_only else None
        data = self.collection.get(where=where, include=["documents", "metadatas"])
        return data["ids"], data["documents"], data["metadatas"]

    def delete_by_id(self, uid: str) -> bool:
        try:
            existing = self.collection.get(ids=[uid])
            if not existing["ids"]:
                return False
            self.collection.delete(ids=[uid])
            return True
        except Exception:
            return False

    def delete_starred(self) -> int:
        ids, _, _ = self.list_all(starred_only=True)
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def delete_by_tags(self, tags: list) -> int:
        deleted = 0
        all_ids, all_docs, all_metas = self.list_all()
        for uid, doc, meta in zip(all_ids, all_docs, all_metas):
            mem_tags = json.loads(meta.get("tags", "[]"))
            if any(t in mem_tags for t in tags):
                if self.delete_by_id(uid):
                    deleted += 1
        return deleted

    def get_recent(self, n: int = 10) -> tuple[list, list, list]:
        all_ids, all_docs, all_metas = self.list_all()
        sorted_idx = sorted(
            range(len(all_metas)),
            key=lambda i: all_metas[i].get("ts", ""),
            reverse=True
        )
        selected = sorted_idx[:n]
        return (
            [all_ids[i] for i in selected],
            [all_docs[i] for i in selected],
            [all_metas[i] for i in selected]
        )

    def get_random(self, n: int = 1) -> tuple[list, list, list]:
        all_ids, all_docs, all_metas = self.list_all()
        import random
        selected = random.sample(range(len(all_ids)), min(n, len(all_ids)))
        return (
            [all_ids[i] for i in selected],
            [all_docs[i] for i in selected],
            [all_metas[i] for i in selected]
        )

    def get_timeline(self) -> list:
        all_ids, all_docs, all_metas = self.list_all()
        timeline = []
        for uid, doc, meta in zip(all_ids, all_docs, all_metas):
            ts = meta.get("ts", "")
            date = ts[:10] if ts else "unknown"
            timeline.append((date, uid, doc, meta))
        timeline.sort(key=lambda x: x[0], reverse=True)
        return timeline

    def get_top_tags(self, n: int = 10) -> list:
        from collections import Counter
        all_ids, all_docs, all_metas = self.list_all()
        tags = []
        for m in all_metas:
            tags.extend(json.loads(m.get("tags", "[]")))
        return Counter(tags).most_common(n)

    def wipe(self):
        data = self.collection.get()
        if data["ids"]:
            self.collection.delete(ids=data["ids"])


# ═══════════════════════════════════════════════════════════════
#  CONSOLE & SOFT WARM THEME
# ═══════════════════════════════════════════════════════════════

console = Console(highlight=False)

# ── soft warm palette ──────────────────────────────────────────
# background feel: deep ink blue-black
# accents: warm amber, rose, sage green, soft lavender
# user input: warm amber so it reads distinctly from system output

C_USER    = "#e8c07a"   # warm amber  — user typed text
C_SYSTEM  = "#9fa8b8"   # cool steel  — system/dim output
C_ACCENT  = "#c8a96e"   # gold        — IDs, highlights
C_OK      = "#87b87f"   # sage green  — success ticks
C_ERR     = "#c97070"   # dusty rose  — errors
C_DIM     = "#4a4f5c"   # muted slate — timestamps, secondary
C_ROSE    = "#c47e8a"   # soft rose   — mode:fact badge
C_SAGE    = "#7a9e7e"   # sage        — mode:chunk/para badge
C_LAVENDER= "#8b8fc7"   # lavender    — spinner, recall header

INPUT_STYLE = Style.from_dict({
    "":                         C_USER,          # all user-typed text is warm amber
    "prompt":                   C_DIM,           # › glyph
    "prompt.fact":              C_ACCENT,
    "ansired":                  C_ERR,
    "ansigreen":                C_OK,
    "completion-menu.completion":          f"bg:#1e2030 {C_DIM}",
    "completion-menu.completion.current":  f"bg:#2a2d3e {C_ACCENT} bold",
    "auto-suggestion":                     "#333740",
})


def _omni_quiet() -> bool:
    """Set OMNI_QUIET=1 to skip splash motion, spinners, and staggered prints."""
    return os.environ.get("OMNI_QUIET", "").strip().lower() in ("1", "true", "yes")


MEMORY_FORMAT = {
    "id": "",
    "content": "",
    "source": "",
    "tags": [],
    "related": [],
    "starred": False,
    "created": "",
    "updated": "",
}


def parse_text_to_memory(filepath: str) -> dict:
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)

    filename = os.path.basename(filepath)
    content = ""
    raw_lines = []

    with open(filepath, encoding="utf-8", errors="replace") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    content = "\n".join(raw_lines)

    date_match = re.search(r"\d{4}-\d{2}-\d{2}", content)
    extracted_date = date_match.group(0) if date_match else ""

    title_line = raw_lines[0] if raw_lines else filename
    if len(title_line) > 80:
        title_line = title_line[:80] + "..."

    tags_from_filename = [os.path.splitext(filename)[0].lower()]

    keywords = ["meeting", "incident", "project", "log", "quote", "idea", "note", "task"]
    for kw in keywords:
        if kw in filename.lower():
            tags_from_filename.append(kw)

    return {
        "source": filename,
        "content": content,
        "title": title_line,
        "date": extracted_date,
        "tags": tags_from_filename,
    }


def memory_to_rich_json(mem: dict, meta: dict) -> dict:
    return {
        "id": mem.get("id", ""),
        "title": mem.get("title", mem.get("content", "")[:50]),
        "content": mem.get("content", ""),
        "source": meta.get("source", ""),
        "tags": json.loads(meta.get("tags", "[]")),
        "related": json.loads(meta.get("related", "[]")),
        "starred": meta.get("starred", False),
        "created": meta.get("ts", ""),
        "updated": meta.get("ts", ""),
    }


def _parse_template(text: str, template: dict) -> dict:
    result = {}
    for field_name, _ in template["fields"]:
        pattern = rf"-?\s*{field_name}:?\s*(.+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result[field_name] = match.group(1).strip()
    return result


def _render_template_fields(template: dict, values: dict = None) -> str:
    lines = []
    for field_name, description in template["fields"]:
        value = values.get(field_name, "") if values else ""
        lines.append(f"- {field_name}: {value}  ({description})")
    return "\n".join(lines)


def _template_to_text(template: dict, values: dict) -> str:
    lines = [f"# {template['name']}", ""]
    for field_name, description in template["fields"]:
        value = values.get(field_name, "")
        if value:
            lines.append(f"- {field_name}: {value}")
    return "\n".join(lines)


def _template_to_json(template: dict, values: dict, uid: str) -> dict:
    return {
        "id": uid,
        "template": list(TEMPLATES.keys())[list(TEMPLATES.values()).index(template)],
        "data": values,
    }


def _template_to_markdown(values: dict, tags: list) -> str:
    title = values.get("title") or values.get("name") or values.get("entry") or ""
    parts = [title]
    content = []
    for k, v in values.items():
        if k not in ("title", "name") and v:
            content.append(f"**{k}**: {v}")
    if content:
        parts.append("\n".join(content))
    if tags:
        parts.append(" ".join(f"#{t}" for t in tags))
    return " | ".join(parts)


def _template_to_csv(memories: list, tags: list) -> str:
    all_fields = set()
    for m in memories:
        all_fields.update(m.get("data", {}).keys())
    fields = sorted(all_fields)
    rows = ["id," + ",".join(fields) + ",tags"]
    for m, t in zip(memories, tags):
        row = [m.get("id", "")]
        for f in fields:
            val = m.get("data", {}).get(f, "").replace(",", ";")
            row.append(val)
        row.append(" ".join(t))
        rows.append(",".join(row))
    return "\n".join(rows)


COMMANDS = [
    ("/add",     "store a fact with optional #tags"),
    ("/import",  "import structured from template"),
    ("/templates", "show available templates"),
    ("/list",    "browse memories with filters"),
    ("/recent",  "show last N memories"),
    ("/random",  "remind me of something"),
    ("/stats",    "memory analytics"),
    ("/related", "show linked memories"),
    ("/del",     "remove by ID, tag, or starred"),
    ("/star",    "pin/unpin memories as important"),
    ("/starred", "list pinned memories"),
    ("/tag",     "add tags to multiple memories"),
    ("/merge",   "combine memories into one"),
    ("/timeline", "view memories by date"),
    ("/alias",   "save/run query aliases"),
    ("/export",  "export to CSV/MD/JSON"),
    ("/wipe",    "erase the entire trunk"),
    ("/help",    "show this reference"),
    ("/exit",    "quit omni"),
]

COMPLETER = WordCompleter(
    [c for c, _ in COMMANDS] + ["alias save", "alias list", "alias run", "export json", "export md", "export csv", "#", "import", "templates"],
    pattern=re.compile(r"(/\w*)"),
    sentence=True,
)

SEARCH_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".omni_searches.json")


def _load_searches() -> dict:
    if not os.path.exists(SEARCH_DB):
        return {}
    try:
        with open(SEARCH_DB, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_searches(searches: dict):
    with open(SEARCH_DB, "w", encoding="utf-8") as f:
        json.dump(searches, f, indent=2)


def _track_search(query: str):
    if not query or query.startswith("/"):
        return
    searches = _load_searches()
    searches[query] = searches.get(query, 0) + 1
    _save_searches(searches)

# First token on a line must not be stored as /para body (e.g. user types /key while pasting)
COMMAND_WORDS = frozenset(cmd.lower() for cmd, _ in COMMANDS)


def _line_is_cli_command(line: str) -> bool:
    parts = line.strip().split(maxsplit=1)
    if not parts or not parts[0].startswith("/"):
        return False
    return parts[0].lower() in COMMAND_WORDS


# ═══════════════════════════════════════════════════════════════
#  ANIMATIONS
# ═══════════════════════════════════════════════════════════════

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

def _spin(label: str, duration: float = 0.7, color: str = C_LAVENDER):
    if _omni_quiet():
        return
    end   = time.time() + duration
    cycle = itertools.cycle(SPINNER_FRAMES)
    with Live(console=console, refresh_per_second=18, transient=True) as live:
        while time.time() < end:
            f = next(cycle)
            live.update(Text.assemble(
                (f"  {f} ", color),
                (label, C_DIM),
            ))
            time.sleep(0.055)


def _cascade_ids(uids: list[str], delay: float = 0.065):
    d = 0.0 if _omni_quiet() else delay
    for uid in uids:
        console.print(f"  [{C_DIM}]·[/{C_DIM}] [{C_ACCENT}]{uid}[/{C_ACCENT}]")
        if d:
            time.sleep(d)


# ═══════════════════════════════════════════════════════════════
#  BRANDING (static — no ANSI cursor tricks; works on all terminals)
# ═══════════════════════════════════════════════════════════════

def _render_logo_minimal():
    title = Text()
    title.append("  omni", style=f"bold {C_USER}")
    title.append("\n")
    title.append("  context memory", style=C_DIM)
    console.print(title)
    console.print()


# ═══════════════════════════════════════════════════════════════
#  SPLASH SCREEN
# ═══════════════════════════════════════════════════════════════

def splash(engine: MemoryEngine):
    console.clear()
    if not _omni_quiet():
        time.sleep(0.04)

    _render_logo_minimal()

    console.print(
        f"  [{C_DIM}]Search the trunk with plain text, or run [/{C_DIM}]"
        f"[{C_ACCENT}]/help[/{C_ACCENT}]"
        f"[{C_DIM}] for commands.[/{C_DIM}]"
    )
    console.print()

    if not _omni_quiet():
        _spin("connecting", duration=0.32, color=C_DIM)
    console.print(f"  [{C_OK}]✓[/{C_OK}] [{C_DIM}]ready[/{C_DIM}]")
    console.print()

    # session stats
    count = engine.count
    ids, docs, metas = engine.list_all()
    starred = sum(1 for m in metas if m.get("starred", False))
    all_tags = set()
    for m in metas:
        all_tags.update(json.loads(m.get("tags", "[]")))

    stat_table = Table(box=None, show_header=False, padding=(0, 3))
    stat_table.add_column("k", style=C_DIM,          no_wrap=True, width=12)
    stat_table.add_column("v", style="bold " + C_USER, no_wrap=True)

    stat_table.add_row("trunk",  f"{count} {'memory' if count == 1 else 'memories'}")
    stat_table.add_row("starred", str(starred))
    stat_table.add_row("tags",  str(len(all_tags)))
    stat_table.add_row("db",    ".omni_trunk/")

    console.print(
        Panel(
            stat_table,
            box=box.ROUNDED,
            border_style=C_DIM,
            padding=(0, 2),
            title=f"[{C_DIM}]session[/{C_DIM}]",
            title_align="left",
        )
    )
    console.print()

    hints = [
        ("/add",     "fact"),
        ("/list",    "browse"),
        ("/help",   "commands"),
    ]
    parts = []
    for cmd, desc in hints:
        parts.append(f"[{C_ACCENT}]{cmd}[/{C_ACCENT}] [{C_DIM}]{desc}[/{C_DIM}]")
    console.print("  " + f"[{C_DIM}] · [/{C_DIM}]".join(parts))
    if not _omni_quiet():
        console.print(f"  [{C_DIM}]Tip: OMNI_QUIET=1 skips spinners and delays.[/{C_DIM}]")
    console.print()


# ═══════════════════════════════════════════════════════════════
#  STATUS BAR
# ═══════════════════════════════════════════════════════════════

def print_statusbar(engine: MemoryEngine):
    count = engine.count
    mem = f"{count} {'memory' if count == 1 else 'memories'}"
    t = Text()
    t.append("  omni", style="bold " + C_USER)
    t.append("  ·  ", style=C_DIM)
    t.append(mem, style=C_LAVENDER)
    t.append("  ·  ", style=C_DIM)
    t.append("chromadb", style=C_DIM)
    console.print(t)
    console.print()


# ═══════════════════════════════════════════════════════════════
#  HELP
# ═══════════════════════════════════════════════════════════════

def show_help():
    console.print()
    table = Table(box=None, padding=(0, 2), show_header=False)
    table.add_column("cmd",  style=C_ACCENT, no_wrap=True, width=10)
    table.add_column("desc", style=C_DIM)
    for cmd, desc in COMMANDS:
        table.add_row(cmd, desc)
    console.print(
        Panel(
            Padding(table, (0, 1)),
            box=box.ROUNDED,
            border_style=C_DIM,
            title=f"[{C_DIM}]commands[/{C_DIM}]",
            title_align="left",
            padding=(0, 1),
        )
    )
    console.print()
    console.print(f"  [{C_DIM}]Query filters:[/{C_DIM}]")
    console.print(f"  [{C_DIM}]  #tag      — filter by tag[/{C_DIM}]")
    console.print(f"  [{C_DIM}]  after:2024-01-01 — date filter[/{C_DIM}]")
    console.print(f"  [{C_DIM}]  last:week    — recent only[/{C_DIM}]")
    console.print(f"  [{C_DIM}]  starred:yes — pinned only[/{C_DIM}]")
    console.print(f"  [{C_DIM}]  fuzzy:yes   — keyword match[/{C_DIM}]")
    console.print()
    console.print(f"  [{C_DIM}]Lines without a leading / are recall queries.[/{C_DIM}]")
    console.print()


# ═══════════════════════════════════════════════════════════════
#  RECALL DISPLAY
# ═══════════════════════════════════════════════════════════════

def show_recall(docs: list[str], metas: list[dict], query: str, highlight_query: str = None):
    if not docs:
        console.print()
        console.print(
            f"  [{C_DIM}]no memories matched[/{C_DIM}]  [{C_DIM}]\"{query}\"[/{C_DIM}]"
        )
        console.print(
            f"  [{C_DIM}]use [/{C_DIM}][{C_ACCENT}]/add[/{C_ACCENT}]"
            f"[{C_DIM}] to store something first[/{C_DIM}]"
        )
        console.print()
        return

    console.print()
    console.print(
        f"  [{C_DIM}]recalled[/{C_DIM}]  [{C_LAVENDER}]{len(docs)}[/{C_LAVENDER}]"
        f"  [{C_DIM}]{'result' if len(docs)==1 else 'results'} for[/{C_DIM}]"
        f"  [{C_ACCENT}]\"{query}\"[/{C_ACCENT}]"
    )
    console.print()

    for doc, meta in zip(docs, metas):
        ts_raw = meta.get("ts", "")
        ts     = ts_raw[:16].replace("T", " ") if ts_raw else "—"
        starred = meta.get("starred", False)
        tags = json.loads(meta.get("tags", "[]"))

        display_doc = _highlight_matches(doc, highlight_query) if highlight_query else doc

        body = Text()
        body.append(f"  {display_doc}\n", style=C_SYSTEM)
        if starred:
            body.append(f"  [{C_OK}]★[/{C_OK}]", style="")
        for t in tags:
            body.append(f" [{C_LAVENDER}]#{t}[/{C_LAVENDER}]", style="")
        body.append(f"   {ts}", style=C_DIM)

        console.print(
            Panel(
                body,
                box=box.ROUNDED,
                border_style=C_DIM,
                padding=(0, 1),
                expand=True,
            )
        )
        if not _omni_quiet():
            time.sleep(0.05)

    console.print()


# ═══════════════════════════════════════════════════════════════
#  LIST DISPLAY
# ═══════════════════════════════════════════════════════════════

def show_list(ids: list, docs: list, metas: list):
    if not ids:
        console.print()
        console.print(f"  [{C_DIM}]trunk is empty — try[/{C_DIM}] [{C_ACCENT}]/add[/{C_ACCENT}]")
        console.print()
        return

    console.print()
    starred_count = sum(1 for m in metas if m.get("starred", False))
    all_tags = set()
    for m in metas:
        all_tags.update(json.loads(m.get("tags", "[]")))
    tags_str = ", ".join(f"#{t}" for t in sorted(all_tags)[:5])
    if len(all_tags) > 5:
        tags_str += f" +{len(all_tags)-5} more"

    console.print(
        f"  [{C_DIM}]{len(ids)} stored[/{C_DIM}]  [{C_DIM}]·[/{C_DIM}]  "
        f"[{C_OK}]{starred_count} starred[/{C_OK}]"
    )
    if tags_str:
        console.print(f"  [{C_DIM}]tags:[/{C_DIM}] {tags_str}")
    console.print()

    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style=C_DIM,
        border_style=C_DIM,
        padding=(0, 1),
    )
    table.add_column("#",      style=C_DIM,     width=3,  justify="right")
    table.add_column("ID",     style=C_ACCENT,  width=12, no_wrap=True)
    table.add_column("",      style=C_DIM,    width=2)
    table.add_column("memory", style="bold " + C_USER)
    table.add_column("tags",  style=C_LAVENDER, width=20)
    table.add_column("stored", style=C_DIM,    width=16)

    for i, (uid, doc, meta) in enumerate(zip(ids, docs, metas), 1):
        ts_raw  = meta.get("ts", "")
        ts      = ts_raw[:16].replace("T", " ") if ts_raw else "—"
        preview = doc[:48] + ("…" if len(doc) > 48 else "")
        tags = json.loads(meta.get("tags", "[]"))
        tags_display = " ".join(f"#{t}" for t in tags[:3]) + (" ..." if len(tags) > 3 else "")
        star = "★" if meta.get("starred", False) else ""

        table.add_row(str(i), uid, star, preview, tags_display, ts)

    console.print(Padding(table, (0, 1)))
    console.print()


# ═══════════════════════════════════════════════════════════════
#  COMMAND HANDLERS
# ═══════════════════════════════════════════════════════════════

def _parse_tags(text: str) -> tuple[str, list]:
    tags = re.findall(r"#(\w+)", text)
    clean = re.sub(r"#\w+", "", text).strip()
    return clean, tags


def _parse_query_filters(query: str) -> tuple[str, dict]:
    filters = {}
    after_match = re.search(r"after:(\d{4}-\d{2}-\d{2})", query)
    before_match = re.search(r"before:(\d{4}-\d{2}-\d{2})", query)
    last_match = re.search(r"last:(\w+)", query)
    starred_match = re.search(r"starred:(\w+)", query)
    fuzzy_match = re.search(r"fuzzy:(\w+)", query)
    tags_match = re.findall(r"#(\w+)", query)

    if after_match:
        try:
            filters["after"] = datetime.datetime.fromisoformat(after_match.group(1))
        except ValueError:
            pass
    if before_match:
        try:
            filters["before"] = datetime.datetime.fromisoformat(before_match.group(1))
        except ValueError:
            pass
    if last_match:
        unit = last_match.group(1)
        now = datetime.datetime.now()
        if unit in ("hour", "hours"):
            filters["after"] = now - datetime.timedelta(hours=1)
        elif unit in ("day", "days"):
            filters["after"] = now - datetime.timedelta(days=int(last_match.group(1).rstrip("s")))
        elif unit in ("week", "weeks"):
            filters["after"] = now - datetime.timedelta(weeks=1)
        elif unit in ("month", "months"):
            filters["after"] = now - datetime.timedelta(days=30)
    if starred_match:
        filters["starred_only"] = starred_match.group(1).lower() in ("1", "yes", "true")
    if fuzzy_match:
        filters["fuzzy"] = fuzzy_match.group(1).lower() in ("1", "yes", "true")
    if tags_match:
        filters["tags"] = tags_match

    clean = re.sub(r"(after:|before:|last:|starred:|fuzzy:)\S+", "", query)
    clean = re.sub(r"#\w+", "", clean).strip()

    return clean, filters


def _autocomplete_tags() -> list[str]:
    all_tags = set()
    _, _, metas = MemoryEngine().list_all()
    for m in metas:
        tags = json.loads(m.get("tags", "[]"))
        all_tags.update(tags)
    return sorted(all_tags)


def cmd_add(engine: MemoryEngine, session: PromptSession):
    console.print()
    console.print(f"  [{C_DIM}]enter fact — use #tag to add tags[/{C_DIM}]\n")
    try:
        fact = session.prompt(
            HTML(f'  <style color="{C_DIM}">fact ›</style> '),
            style=INPUT_STYLE,
        ).strip()
    except (KeyboardInterrupt, EOFError):
        console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
        return

    if not fact:
        console.print(f"  [{C_DIM}]nothing stored[/{C_DIM}]\n")
        return

    text, tags = _parse_tags(fact)
    console.print()
    _spin("vectorising", duration=0.65)
    uid = engine.store(text, tags=tags)

    console.print(
        f"  [{C_OK}]✓[/{C_OK}] [{C_DIM}]stored as[/{C_DIM}] [{C_ACCENT}]{uid}[/{C_ACCENT}]"
    )
    if tags:
        console.print(
            f"  [{C_DIM}]tags:[/{C_DIM}] " + " ".join(f"[{C_LAVENDER}]#{t}[/{C_LAVENDER}]" for t in tags)
        )
    console.print(f"  [{C_DIM}]  {text[:80]}{'…' if len(text) > 80 else ''}[/{C_DIM}]")
    console.print()





def cmd_star(engine: MemoryEngine, session: PromptSession):
    console.print()
    try:
        uid = session.prompt(
            HTML(f'  <style color="{C_DIM}">ID to pin/unpin ›</style> '),
            style=INPUT_STYLE,
        ).strip().upper()
    except (KeyboardInterrupt, EOFError):
        console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
        return

    if not uid:
        return

    doc, meta = engine.get_by_id(uid)
    if not doc:
        console.print(f"  [{C_ERR}]✗[/{C_ERR}] [{C_DIM}]ID not found[/{C_DIM}]")
        console.print()
        return

    current = meta.get("starred", False)
    new_state = not current
    engine.update(uid, starred=new_state)

    action = "pinned" if new_state else "unpinned"
    console.print(
        f"  [{C_OK}]✓[/{C_OK}] [{C_DIM}]{uid} {action}[/{C_DIM}]"
    )
    console.print()


def cmd_starred(engine: MemoryEngine):
    ids, docs, metas = engine.list_all(starred_only=True)
    if not ids:
        console.print()
        console.print(f"  [{C_DIM}]no pinned memories[/{C_DIM}]")
        console.print()
        return
    show_list(ids, docs, metas)


def cmd_alias(engine: MemoryEngine, session: PromptSession, action: str = "list", query_override: str = None):
    aliases = _load_aliases()

    if action == "list":
        if not aliases:
            console.print()
            console.print(f"  [{C_DIM}]no saved aliases[/{C_DIM}]")
            console.print()
            return
        console.print()
        for name, query in aliases.items():
            console.print(
                f"  [{C_ACCENT}]{name}[/{C_ACCENT}] → {query}"
            )
        console.print()
        return

    if action == "run":
        name = query_override
        if not name:
            try:
                name = session.prompt(
                    HTML(f'  <style color="{C_DIM}">alias name ›</style> '),
                    style=INPUT_STYLE,
                ).strip()
            except (KeyboardInterrupt, EOFError):
                console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
                return

        if not name or name not in aliases:
            console.print(f"  [{C_ERR}]✗[/{C_ERR}] [{C_DIM}]alias not found[/{C_DIM}]")
            console.print()
            return

        query = aliases[name]
        handle_recall(engine, query)
        return

    if action == "save":
        try:
            name = session.prompt(
                HTML(f'  <style color="{C_DIM}">alias name ›</style> '),
                style=INPUT_STYLE,
            ).strip()
        except (KeyboardInterrupt, EOFError):
            console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
            return

        if not name:
            console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
            return

        query = session.prompt(
            HTML(f'  <style color="{C_DIM}">query ›</style> '),
            style=INPUT_STYLE,
        ).strip()
        if not query:
            console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
            return

        aliases[name] = query
        _save_aliases(aliases)
        console.print(
            f"  [{C_OK}]✓[/{C_OK}] [{C_DIM}]saved[/{C_DIM}] [{C_ACCENT}]{name}[/{C_ACCENT}] → {query}"
        )
        console.print()


def cmd_recent(engine: MemoryEngine, session: PromptSession):
    console.print()
    try:
        n_str = session.prompt(
            HTML(f'  <style color="{C_DIM}">how many (default 10) ›</style> '),
            style=INPUT_STYLE,
        ).strip()
    except (KeyboardInterrupt, EOFError):
        console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
        return

    n = int(n_str) if n_str.isdigit() else 10
    ids, docs, metas = engine.get_recent(n)
    show_list(ids, docs, metas)


def cmd_templates(engine: MemoryEngine):
    console.print()
    table = Table(box=None, padding=(0, 2), show_header=False)
    table.add_column("name", style=C_ACCENT, no_wrap=True, width=12)
    table.add_column("desc", style=C_DIM)

    for key, tmpl in TEMPLATES.items():
        table.add_row(key, tmpl["name"])

    console.print(
        Panel(
            table,
            box=box.ROUNDED,
            border_style=C_DIM,
            title=f"[{C_DIM}]templates[/{C_DIM}]",
            padding=(0, 1),
        )
    )

    console.print(f"  [{C_DIM}]use /import to create from template[/{C_DIM}]")
    console.print()


def _autoimport_file(filepath: str, template: dict, tags: list = None) -> str:
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)

    with open(filepath, encoding="utf-8") as f:
        raw = f.read()

    parsed = _parse_template(raw, template)

    values = {}
    for field_name, description in template["fields"]:
        value = parsed.get(field_name, "")
        if value:
            values[field_name] = value

    if not values:
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        for field_name, _ in template["fields"]:
            if lines:
                values[field_name] = lines.pop(0)
            else:
                values[field_name] = ""

    text = _template_to_text(template, values)
    return text


IMPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drop")


def cmd_import(engine: MemoryEngine, session: PromptSession, file_path: str = None):
    files_to_import = []

    if not file_path:
        if not os.path.exists(IMPORT_DIR):
            os.makedirs(IMPORT_DIR, exist_ok=True)
            console.print(f"  [{C_DIM}]created drop folder:[/{C_DIM}] {IMPORT_DIR}")
            console.print()
            console.print(f"  [{C_DIM}]drop files there, then run[/{C_DIM}] /import")
            console.print()
            return

        files_to_import = [
            f for f in os.listdir(IMPORT_DIR)
            if os.path.isfile(os.path.join(IMPORT_DIR, f)) and not f.startswith(".")
        ]

        if not files_to_import:
            console.print(f"  [{C_DIM}]no files in drop folder[/{C_DIM}]")
            console.print(f"  [{C_DIM}]drop files in:[/{C_DIM}] {IMPORT_DIR}")
            console.print()
            return

        console.print(f"  [{C_DIM}]found[/{C_DIM}] {len(files_to_import)} files")
        console.print()

        for filename in files_to_import:
            full_path = os.path.join(IMPORT_DIR, filename)
            try:
                mem_data = parse_text_to_memory(full_path)
                content = mem_data["content"]
                tags = mem_data["tags"]

                if content:
                    _spin(f"storing {filename}", duration=0.3)
                    uid = engine.store(content, tags=tags, source=full_path)
                    print(f"  [{C_OK}]✓[/{C_OK}] {filename} → {uid}")
            except Exception as e:
                print(f"  [{C_ERR}]✗[/{C_ERR}] {filename}: {e}")

        console.print()
        console.print(f"  [{C_OK}]✓[/{C_OK}] [{C_DIM}]imported[/{C_DIM}] {len(files_to_import)} files")
        console.print()

        for f in files_to_import:
            dst = f"{IMPORT_DIR}/imported/{f}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(os.path.join(IMPORT_DIR, f), dst)

        return

    if os.path.isfile(file_path):
        console.print(f"  [{C_DIM}]importing[/{C_DIM}] {file_path}")

        try:
            mem_data = parse_text_to_memory(file_path)
        except Exception as e:
            console.print(f"  [{C_ERR}]✗[/{C_ERR}] [{C_DIM}]error: {e}[/{C_DIM}]")
            return

        content = mem_data["content"]
        tags = mem_data["tags"]

        if not content:
            console.print(f"  [{C_ERR}]✗[/{C_ERR}] [{C_DIM}]empty file[/{C_DIM}]")
            return

        _spin("storing", duration=0.5)
        uid = engine.store(content, tags=tags, source=file_path)

        console.print()
        console.print(
            f"  [{C_OK}]✓[/{C_OK}] [{C_DIM}]stored as[/{C_DIM}] [{C_ACCENT}]{uid}[/{C_ACCENT}]"
        )
        console.print(f"  [{C_DIM}]source:[/{C_DIM}] {mem_data['source']}")
        console.print(f"  [{C_DIM}]tags:[/{C_DIM}] " + " ".join(f"#{t}" for t in tags))
        console.print(f"  [{C_DIM}]date:[/{C_DIM}] {mem_data['date'] or 'auto'}")
        console.print()
        return


def cmd_related(engine: MemoryEngine, session: PromptSession):
    console.print()
    try:
        uid = session.prompt(
            HTML(f'  <style color="{C_DIM}">memory ID ›</style> '),
            style=INPUT_STYLE,
        ).strip().upper()
    except (KeyboardInterrupt, EOFError):
        console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
        return

    if not uid:
        return

    related = engine.get_related(uid)
    if not related:
        console.print(f"  [{C_DIM}]no related memories[/{C_DIM}]")
        console.print()
        return

    related_docs = []
    related_metas = []
    for r_uid in related:
        doc, meta = engine.get_by_id(r_uid)
        if doc:
            related_docs.append(doc)
            related_metas.append(meta)

    console.print(f"  [{C_DIM}]related to {uid}:[/{C_DIM}]")
    show_list(related, related_docs, related_metas)


def cmd_random(engine: MemoryEngine, session: PromptSession):
    console.print()
    try:
        n_str = session.prompt(
            HTML(f'  <style color="{C_DIM}">how many (default 1) ›</style> '),
            style=INPUT_STYLE,
        ).strip()
    except (KeyboardInterrupt, EOFError):
        console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
        return

    n = int(n_str) if n_str.isdigit() else 1
    ids, docs, metas = engine.get_random(n)
    console.print()
    _cascade_ids(ids, delay=0)
    show_list(ids, docs, metas)


def cmd_stats(engine: MemoryEngine):
    ids, docs, metas = engine.list_all()
    if not ids:
        console.print()
        console.print(f"  [{C_DIM}]no memories yet[/{C_DIM}]")
        console.print()
        return

    total = len(ids)
    starred = sum(1 for m in metas if m.get("starred", False))
    top_tags = engine.get_top_tags(5)

    searches = _load_searches()
    top_searches = sorted(searches.items(), key=lambda x: x[1], reverse=True)[:5]

    stat_table = Table(box=None, show_header=False, padding=(0, 3))
    stat_table.add_column("k", style=C_DIM, no_wrap=True, width=14)
    stat_table.add_column("v", style="bold " + C_USER, no_wrap=True)

    stat_table.add_row("memories", str(total))
    stat_table.add_row("starred", str(starred))
    stat_table.add_row("tags", str(sum(1 for m in metas if m.get("tags", "[]") != "[]")))

    console.print()
    console.print(
        Panel(
            stat_table,
            box=box.ROUNDED,
            border_style=C_DIM,
            padding=(0, 2),
            title=f"[{C_DIM}]overview[/{C_DIM}]",
        )
    )

    if top_tags:
        console.print()
        console.print(f"  [{C_DIM}]top tags:[/{C_DIM}]")
        for tag, count in top_tags:
            bar = "█" * count
            console.print(f"    [{C_LAVENDER}]{tag}[/{C_LAVENDER}] {bar} {count}")

    if top_searches:
        console.print()
        console.print(f"  [{C_DIM}]frequent searches:[/{C_DIM}]")
        for query, count in top_searches:
            console.print(f"    [{C_ACCENT}]{query}[/{C_ACCENT}] ×{count}")

    console.print()


def cmd_timeline(engine: MemoryEngine):
    timeline = engine.get_timeline()
    if not timeline:
        console.print()
        console.print(f"  [{C_DIM}]no memories[/{C_DIM}]")
        console.print()
        return

    console.print()
    current_date = None
    for date, uid, doc, meta in timeline:
        if date != current_date:
            console.print()
            console.print(f"  [{C_DIM}]─ {date} ─[/{C_DIM}]")
            current_date = date
        tags = json.loads(meta.get("tags", "[]"))
        tags_str = " ".join(f"#{t}" for t in tags) if tags else ""
        star = "★" if meta.get("starred", False) else " "
        console.print(f"  [{C_ACCENT}]{uid}[/{C_ACCENT}] {star} {doc[:60]}{'...' if len(doc) > 60 else ''}")
        if tags_str:
            console.print(f"       [{C_LAVENDER}]{tags_str}[/{C_LAVENDER}]")
    console.print()


def cmd_tag(engine: MemoryEngine, session: PromptSession):
    ids, docs, metas = engine.list_all()
    if not ids:
        console.print()
        console.print(f"  [{C_DIM}]no memories to tag[/{C_DIM}]")
        console.print()
        return

    console.print()
    console.print(f"  [{C_DIM}]IDs to tag (e.g. MEM-123 MEM-456) ›[/{C_DIM}]")
    try:
        ids_input = session.prompt(
            HTML(f'  <style color="{C_DIM}">IDs ›</style> '),
            style=INPUT_STYLE,
        ).strip()
    except (KeyboardInterrupt, EOFError):
        console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
        return

    target_ids = [i.strip().upper() for i in ids_input.split() if i.strip()]
    if not target_ids:
        return

    console.print(f"  [{C_DIM}]tags to add (space separated) ›[/{C_DIM}]")
    try:
        tags_input = session.prompt(
            HTML(f'  <style color="{C_DIM}">tags ›</style> '),
            style=INPUT_STYLE,
        ).strip()
    except (KeyboardInterrupt, EOFError):
        console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
        return

    new_tags = [t.strip().lstrip("#") for t in tags_input.split() if t.strip()]
    if not new_tags:
        return

    tagged = 0
    for uid in target_ids:
        doc, meta = engine.get_by_id(uid)
        if doc:
            existing = json.loads(meta.get("tags", "[]"))
            combined = list(set(existing + new_tags))
            engine.update(uid, tags=combined)
            tagged += 1

    console.print(
        f"  [{C_OK}]✓[/{C_OK}] [{C_DIM}]added #[/{C_DIM}]" + ", #".join(new_tags) +
        f" [{C_DIM}]to[/{C_DIM}] {tagged} memories"
    )
    console.print()


def cmd_merge(engine: MemoryEngine, session: PromptSession):
    console.print()
    console.print(f"  [{C_DIM}]IDs to merge (2+) ›[/{C_DIM}]")
    try:
        ids_input = session.prompt(
            HTML(f'  <style color="{C_DIM}">IDs ›</style> '),
            style=INPUT_STYLE,
        ).strip()
    except (KeyboardInterrupt, EOFError):
        console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
        return

    target_ids = [i.strip().upper() for i in ids_input.split() if i.strip()]
    if len(target_ids) < 2:
        console.print(f"  [{C_ERR}]✗[/{C_ERR}] [{C_DIM}]need 2+ IDs[/{C_DIM}]")
        console.print()
        return

    docs = []
    all_tags = set()
    existing_ids = []
    for uid in target_ids:
        doc, meta = engine.get_by_id(uid)
        if doc:
            docs.append(doc)
            existing_ids.append(uid)
            all_tags.update(json.loads(meta.get("tags", "[]")))

    if len(docs) < 2:
        console.print(f"  [{C_ERR}]✗[/{C_ERR}] [{C_DIM}]couldn't find memories[/{C_DIM}]")
        console.print()
        return

    console.print()
    for i, doc in enumerate(docs):
        console.print(f"  [{C_DIM}]{i+1}.[/{C_DIM}] {doc[:60]}")
    console.print()

    console.print(f"  [{C_DIM}]merged text ›[/{C_DIM}]")
    try:
        merged_text = session.prompt(
            HTML(f'  <style color="{C_DIM}">text ›</style> '),
            style=INPUT_STYLE,
        ).strip()
    except (KeyboardInterrupt, EOFError):
        console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
        return

    if not merged_text:
        merged_text = " | ".join(docs)

    new_tags = list(all_tags)
    _spin("merging", duration=0.5)
    new_uid = engine.store(merged_text, tags=new_tags)

    for uid in existing_ids:
        engine.delete_by_id(uid)

    console.print(
        f"  [{C_OK}]✓[/{C_OK}] [{C_DIM}]merged into[/{C_DIM}] [{C_ACCENT}]{new_uid}[/{C_ACCENT}]"
    )
    console.print()


def cmd_export(engine: MemoryEngine, session: PromptSession):
    try:
        fmt = session.prompt(
            HTML(f'  <style color="{C_DIM}">format (json/md/csv/frontmatter) ›</style> '),
            style=INPUT_STYLE,
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
        return

    if not fmt:
        fmt = "json"

    ids, docs, metas = engine.list_all()

    if not ids:
        console.print(f"  [{C_DIM}]nothing to export[/{C_DIM}]")
        console.print()
        return

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    if fmt == "json":
        data = []
        for uid, doc, meta in zip(ids, docs, metas):
            data.append(memory_to_rich_json({"id": uid, "content": doc}, meta))
        filename = f"omni_memory_{ts}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    elif fmt == "rich":
        data = []
        for uid, doc, meta in zip(ids, docs, metas):
            data.append({
                "id": uid,
                "content": doc,
                "source": meta.get("source", ""),
                "tags": json.loads(meta.get("tags", "[]")),
                "related": json.loads(meta.get("related", "[]")),
                "starred": meta.get("starred", False),
                "created": meta.get("ts", ""),
                "updated": meta.get("ts", ""),
                "metadata": {
                    "imported_from_file": True,
                    "format_version": "1.0",
                }
            })
        filename = f"omni_rich_{ts}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({
                "version": "1.0",
                "exported": datetime.datetime.now().isoformat(),
                "count": len(data),
                "memories": data,
            }, f, indent=2)

    elif fmt == "csv":
        filename = f"omni_export_{ts}.csv"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("id,text,tags,related,starred,ts\n")
            for uid, doc, meta in zip(ids, docs, metas):
                tags = " ".join(json.loads(meta.get("tags", "[]")))
                related = " ".join(json.loads(meta.get("related", "[]")))
                starred = str(meta.get("starred", False))
                ts_val = meta.get("ts", "")
                doc_escaped = doc.replace('"', '""').replace("\n", " ")
                f.write(f'"{uid}","{doc_escaped}","{tags}","{related}","{starred}","{ts_val}"\n')

    elif fmt == "frontmatter":
        filename = f"omni_export_{ts}.md"
        with open(filename, "w", encoding="utf-8") as f:
            for uid, doc, meta in zip(ids, docs, metas):
                tags = json.loads(meta.get("tags", "[]"))
                related = json.loads(meta.get("related", "[]"))
                starred = meta.get("starred", False)
                ts_val = meta.get("ts", "")

                frontmatter = f"""---
id: {uid}
tags: {', '.join(tags)}
related: {', '.join(related)}
starred: {starred}
date: {ts_val[:10] if ts_val else ''}
---

{doc}

"""
                f.write(frontmatter)
    else:
        filename = f"omni_export_{ts}.md"
        with open(filename, "w", encoding="utf-8") as f:
            for uid, doc, meta in zip(ids, docs, metas):
                tags = json.loads(meta.get("tags", "[]"))
                f.write(f"## {uid}\n\n{doc}\n\n")
                if tags:
                    f.write(f"tags: {' '.join(f'#{t}' for t in tags)}\n\n")
                f.write(f"---\n\n")

    console.print(
        f"  [{C_OK}]✓[/{C_OK}] [{C_DIM}]exported[/{C_DIM}] {filename}"
    )
    console.print()


def cmd_del(engine: MemoryEngine, session: PromptSession):
    console.print()
    console.print(f"  [{C_DIM}]ID, #tag, or 'starred' to delete[/{C_DIM}]\n")
    try:
        target = session.prompt(
            HTML(f'  <style color="{C_DIM}">delete ›</style> '),
            style=INPUT_STYLE,
        ).strip()
    except (KeyboardInterrupt, EOFError):
        console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
        return

    if not target:
        return

    if target.lower() == "starred":
        try:
            ans = session.prompt(
                HTML(f'  <ansired>delete ALL pinned? (y/N) ›</ansired> '),
                style=INPUT_STYLE,
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
            return

        if ans != "y":
            console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
            return

        _spin("removing pinned", duration=0.45)
        count = engine.delete_starred()
        console.print(f"  [{C_OK}]✓[/{C_OK}] [{C_DIM}]deleted[/{C_DIM}] {count} pinned")
        console.print()
        return

    if target.startswith("#"):
        tags = [target[1:]]
        try:
            ans = session.prompt(
                HTML(f'  <ansired>delete all #{target[1:]}? (y/N) ›</ansired> '),
                style=INPUT_STYLE,
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
            return

        if ans != "y":
            console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
            return

        _spin("removing", duration=0.45)
        count = engine.delete_by_tags(tags)
        console.print(f"  [{C_OK}]✓[/{C_OK}] [{C_DIM}]deleted[/{C_DIM}] {count} with #{tags[0]}")
        console.print()
        return

    uid = target.upper()
    doc, meta = engine.get_by_id(uid)
    if not doc:
        console.print(
            f"  [{C_ERR}]✗[/{C_ERR}] [{C_DIM}]ID not found[/{C_DIM}]"
        )
        console.print()
        return

    try:
        ans = session.prompt(
            HTML(f'  <ansired>delete {uid}? (y/N) ›</ansired> '),
            style=INPUT_STYLE,
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
        return

    if ans != "y":
        console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
        return

    console.print()
    _spin("removing", duration=0.45)
    ok = engine.delete_by_id(uid)
    if ok:
        console.print(
            f"  [{C_OK}]✓[/{C_OK}] [{C_DIM}]deleted[/{C_DIM}] [{C_ACCENT}]{uid}[/{C_ACCENT}]"
        )
    else:
        console.print(
            f"  [{C_ERR}]✗[/{C_ERR}] [{C_DIM}]ID not found:[/{C_DIM}] [{C_ACCENT}]{uid}[/{C_ACCENT}]"
        )
    console.print()


def cmd_wipe(engine: MemoryEngine, session: PromptSession):
    console.print()
    try:
        ans = session.prompt(
            HTML('  <ansired>erase ALL memories? this is permanent. (y/N) ›</ansired> '),
            style=INPUT_STYLE,
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
        return

    if ans != "y":
        console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]\n")
        return

    console.print()
    _spin("wiping trunk", duration=0.65)
    engine.wipe()
    console.print(f"  [{C_OK}]✓[/{C_OK}] [{C_DIM}]trunk cleared[/{C_DIM}]")
    console.print()


def _highlight_matches(text: str, query: str) -> str:
    if not query or not text:
        return text
    query_lower = query.lower()
    words = [w for w in query_lower.split() if len(w) > 2]
    if not words:
        return text
    result = text
    for word in words:
        if word.lower() in text.lower():
            result = result.replace(word, f"[{C_OK}]{word}[/{C_OK}]", 1)
    return result


def handle_recall(engine: MemoryEngine, query: str):
    _track_search(query)
    query_text, filters = _parse_query_filters(query)
    _spin("scanning memory", duration=0.6, color=C_LAVENDER)
    docs, metas = engine.recall(
        query_text,
        n=filters.get("n", 10),
        tags=filters.get("tags"),
        starred_only=filters.get("starred_only", False),
        after=filters.get("after"),
        before=filters.get("before"),
        fuzzy=filters.get("fuzzy", False),
    )
    show_recall(docs, metas, query, highlight_query=query_text if query_text else None)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    engine  = MemoryEngine()
    session = PromptSession(
        history=InMemoryHistory(),
        auto_suggest=AutoSuggestFromHistory(),
        completer=COMPLETER,
        complete_while_typing=True,
    )

    splash(engine)
    print_statusbar(engine)

    while True:
        try:
            raw = session.prompt(
                HTML(f'  <style color="{C_DIM}">›</style> '),
                style=INPUT_STYLE,
            ).strip()
        except KeyboardInterrupt:
            continue
        except EOFError:
            console.print(f"\n  [{C_DIM}]bye[/{C_DIM}]\n")
            break

        if not raw:
            continue

        cmd = raw.split()[0].lower() if raw.startswith("/") else None

        if cmd == "/exit":
            console.print()
            console.print(f"  [{C_DIM}]bye[/{C_DIM}]")
            console.print()
            break

        elif cmd == "/help":
            show_help()

        elif cmd == "/add":
            cmd_add(engine, session)
            print_statusbar(engine)

        elif cmd == "/templates":
            cmd_templates(engine)

        elif cmd == "/import":
            parts = raw.split(maxsplit=1)
            file_arg = parts[1].strip() if len(parts) > 1 else None
            cmd_import(engine, session, file_arg)
            print_statusbar(engine)

        elif cmd == "/related":
            cmd_related(engine, session)

        elif cmd == "/list":
            ids, docs, metas = engine.list_all()
            show_list(ids, docs, metas)

        elif cmd == "/recent":
            cmd_recent(engine, session)

        elif cmd == "/random":
            cmd_random(engine, session)

        elif cmd == "/stats":
            cmd_stats(engine)

        elif cmd == "/timeline":
            cmd_timeline(engine)

        elif cmd == "/tag":
            cmd_tag(engine, session)
            print_statusbar(engine)

        elif cmd == "/merge":
            cmd_merge(engine, session)
            print_statusbar(engine)

        elif cmd == "/del":
            cmd_del(engine, session)
            print_statusbar(engine)

        elif cmd == "/star":
            cmd_star(engine, session)
            print_statusbar(engine)

        elif cmd == "/starred":
            cmd_starred(engine)
            print_statusbar(engine)

        elif cmd == "/alias":
            parts = raw.split(maxsplit=2)
            sub = parts[1] if len(parts) > 1 else None
            if sub == "run" and len(parts) > 2:
                cmd_alias(engine, session, "run", query_override=parts[2])
            elif sub == "list":
                cmd_alias(engine, session, "list")
            elif sub == "save":
                cmd_alias(engine, session, "save")
            else:
                cmd_alias(engine, session, "list")

        elif cmd == "/export":
            cmd_export(engine, session)

        elif cmd == "/wipe":
            cmd_wipe(engine, session)
            print_statusbar(engine)

        elif cmd:
            console.print()
            console.print(
                f"  [{C_ERR}]✗[/{C_ERR}] [{C_DIM}]unknown command[/{C_DIM}] "
                f"[bold {C_USER}]{cmd}[/bold {C_USER}]  "
                f"[{C_DIM}]/help for reference[/{C_DIM}]"
            )
            console.print()

        else:
            handle_recall(engine, raw)


if __name__ == "__main__":
    main()
