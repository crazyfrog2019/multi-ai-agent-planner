# pig.py
# Autonomous multi-agent Ask API with Ollama + tiny JSONL RAG
# FastAPI endpoints: /health, /ingest, /ask

import os, re, json, hashlib, math
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel

# =========================
# Config
# =========================
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL   = os.environ.get("LLM_MODEL", "qwen2.5:3b-instruct")
EMB_MODEL   = os.environ.get("EMB_MODEL", "nomic-embed-text")
INDEX_DIR   = os.environ.get("INDEX_DIR", "./_rag_index")

TOP_K_FILES_DEFAULT = 4
TOP_K_WEB_DEFAULT   = 2
CHUNK_CHARS = 2000

SUBJECTS = ["Bio", "Tech", "Finance", "General"]

# =========================
# Tiny JSONL Vector Store
# =========================
def ensure_index_dir():
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

def index_path(collection: str) -> Path:
    ensure_index_dir()
    return Path(INDEX_DIR) / f"{collection}.jsonl"

def add_to_index(collection: str, docs: List[Dict[str, Any]]) -> None:
    p = index_path(collection)
    with p.open("a", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def read_index(collection: str) -> List[Dict[str, Any]]:
    p = index_path(collection)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def build_doc_id(path: str, chunk_idx: int, text: str) -> str:
    h = hashlib.sha256((path + str(chunk_idx) + text[:128]).encode("utf-8")).hexdigest()[:16]
    return f"{Path(path).name}-{chunk_idx}-{h}"

def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS) -> List[str]:
    return [text[i:i+chunk_chars] for i in range(0, len(text), chunk_chars)]

# =========================
# Ollama Helpers
# =========================
def _ollama_ok() -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        return r.ok
    except Exception:
        return False

def ollama_embed(texts: List[str]) -> List[List[float]]:
    """
    Call Ollama embeddings per text for portability across versions.
    """
    vecs: List[List[float]] = []
    for t in texts:
        r = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": EMB_MODEL, "prompt": t},
            headers={"Accept": "application/json"},
            timeout=60
        )
        r.raise_for_status()
        data = r.json()
        vecs.append(data.get("embedding") or (data.get("embeddings") or [])[0])
    return vecs

def _parse_ndjson_concat_content(ndjson_text: str) -> str:
    """
    For streamed NDJSON: join 'message.content' across chunks.
    Ignores non-JSON lines safely.
    """
    content_parts = []
    for line in ndjson_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            msg = obj.get("message") or {}
            part = msg.get("content") or obj.get("content") or ""
            if part:
                content_parts.append(part)
        except Exception:
            continue
    return "".join(content_parts)

def ollama_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    """
    Prefer non-streaming JSON. If server still streams NDJSON, parse manually.
    Handles both {"message":{"content":...}} and legacy {"content":...}.
    """
    payload = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature},
        "stream": False  # key fix to prevent NDJSON in most builds
    }
    r = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json=payload,
        headers={"Accept": "application/json"},
        timeout=120
    )

    # Try clean JSON first
    try:
        r.raise_for_status()
        j = r.json()
        if isinstance(j, dict):
            if "message" in j and isinstance(j["message"], dict) and "content" in j["message"]:
                return j["message"]["content"]
            if "content" in j:
                return j["content"]
        if isinstance(j, list) and j and isinstance(j[0], dict):
            m0 = j[0].get("message") or {}
            return m0.get("content") or j[0].get("content") or ""
        if isinstance(j, str):
            return j
    except requests.exceptions.JSONDecodeError:
        # If server still streams NDJSON, fall through
        pass

    # Parse NDJSON or raw text if necessary
    text = r.text or ""
    content = _parse_ndjson_concat_content(text)
    return content or text

def cosine(a: List[float], b: List[float]) -> float:
    da = math.sqrt(sum(x*x for x in a)) or 1e-9
    db = math.sqrt(sum(y*y for y in b)) or 1e-9
    return sum(x*y for x, y in zip(a, b)) / (da * db)

# =========================
# Tools
# =========================
def tool_local_list_files(root: str, patterns: Optional[List[str]] = None) -> List[str]:
    patterns = patterns or ["**/*.txt", "**/*.md", "**/*.csv", "**/*.log"]
    rp = Path(root)
    files: List[str] = []
    for pat in patterns:
        files.extend([str(p) for p in rp.glob(pat)])
    return files

def tool_local_read(path: str, max_chars: int = 4000) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception as e:
        return f"[read_error:{e}]"

def tool_local_ingest(root: str, collection: str = "default") -> Dict[str, Any]:
    files = tool_local_list_files(root)
    docs_to_add, texts, metas = [], [], []
    for fp in files:
        raw = tool_local_read(fp, max_chars=200_000)
        for i, chunk in enumerate(chunk_text(raw, CHUNK_CHARS)):
            texts.append(chunk)
            metas.append({"path": fp, "chunk": i})
    if not texts:
        return {"ok": True, "added": 0, "reason": "no files found"}
    vecs = ollama_embed(texts)
    for vec, meta, text in zip(vecs, metas, texts):
        doc = {
            "id": build_doc_id(meta["path"], meta["chunk"], text),
            "text": text,
            "meta": meta,
            "vector": vec
        }
        docs_to_add.append(doc)
    add_to_index(collection, docs_to_add)
    return {"ok": True, "added": len(docs_to_add)}

def tool_local_search(query: str, top_k: int = TOP_K_FILES_DEFAULT, collection: str = "default") -> List[Dict[str, Any]]:
    docs = read_index(collection)
    if not docs:
        return []
    qvec = ollama_embed([query])[0]
    scored = [(cosine(qvec, d["vector"]), d) for d in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for s, d in scored[:top_k]:
        out.append({
            "score": float(s),
            "text": d["text"][:1200],
            "path": d["meta"]["path"],
            "chunk": d["meta"]["chunk"]
        })
    return out

def tool_web_search(query: str, top_k: int = TOP_K_WEB_DEFAULT, timeout: int = 10) -> List[Dict[str, str]]:
    """
    Very light best-effort DuckDuckGo HTML search (no API key).
    """
    try:
        url = "https://duckduckgo.com/html/"
        resp = requests.post(url, data={"q": query}, timeout=timeout,
                             headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        results = []
        for m in re.finditer(r'<a rel="nofollow" class="result__a" href="([^"]+)".*?>(.*?)</a>', resp.text):
            link = m.group(1)
            title = re.sub("<.*?>", "", m.group(2))
            results.append({"title": title, "url": link})
            if len(results) >= top_k:
                break
        return results
    except Exception as e:
        return [{"title": "[search_error]", "url": str(e)}]

# =========================
# Agent System Prompts
# =========================
SYSTEM_ROUTER = """You are a Router Agent. Select the most relevant SME subjects (1-2) for the user's question.
Return STRICT JSON only: {"subjects": ["Bio","Tech","Finance","General"]}"""

SYSTEM_RESEARCHER = """You are a Research Agent. Turn the RAG & web hits into very short, high-signal bullet notes (<=8 bullets)."""

# IMPORTANT: Do NOT use .format(...) on a string with JSON braces. We'll .replace("{subject}", subject).
SYSTEM_SME_TMPL = """You are the {subject} SME.
Use the research memo below and draft up to 6 concise bullets answering the user.
If research looks insufficient, propose 1 focused follow-up query.
Return STRICT JSON only as one of:
{"action":"draft","content":"<short bullet answer>"}
{"action":"need_more_research","followup_query":"<query>","notes":"<why>"}"""

SYSTEM_COMPOSER = """You are the Composer merging multiple SME bullet drafts into a clean, user-facing answer (short, actionable).
If drafts conflict or are thin, request 'need_more'.
Return STRICT JSON only as:
{"status":"ready","final":"<clean answer>"} OR
{"status":"need_more","request":{"type":"research","detail":"<what extra to look up>"}}"""

# =========================
# Orchestration helpers
# =========================
def parse_first_json(s: str) -> Dict[str, Any]:
    try:
        m = re.search(r"\{.*\}", s, re.S)
        if not m:
            return {}
        return json.loads(m.group(0))
    except Exception:
        return {}

def router_select_subjects(query: str, subjects_hint: Optional[List[str]] = None) -> List[str]:
    if subjects_hint:
        return [s for s in subjects_hint if s in SUBJECTS][:2] or ["General"]
    msg = [{"role": "system", "content": SYSTEM_ROUTER},
           {"role": "user", "content": query}]
    raw = ollama_chat(LLM_MODEL, msg)
    data = parse_first_json(raw)
    subs = [s for s in data.get("subjects", []) if s in SUBJECTS]
    return subs[:2] or ["General"]

def build_research_memo(query: str,
                        rag_hits: List[Dict[str, Any]],
                        web_hits: List[Dict[str, str]],
                        followup: Optional[str] = None) -> str:
    rag_lines = []
    for h in rag_hits:
        clean_text = h["text"].replace("\n", " ")[:200]
        rag_lines.append(f"- {h['path']} (score={h['score']:.3f}) :: {clean_text}")
    web_lines = [f"- {w['title']} -> {w['url']}" for w in web_hits]
    rag_txt = "\n".join(rag_lines) or "[none]"
    web_txt = "\n".join(web_lines) or "[none]"
    prompt = [
        {"role": "system", "content": SYSTEM_RESEARCHER},
        {"role": "user",
         "content": f"USER QUERY:\n{query}\nFOLLOWUP:\n{followup or '[none]'}\nRAG:\n{rag_txt}\nWEB:\n{web_txt}"}
    ]
    return ollama_chat(LLM_MODEL, prompt)

def run_sme_for_subject(subject: str, query: str, research_memo: str) -> Dict[str, Any]:
    system_prompt = SYSTEM_SME_TMPL.replace("{subject}", subject)  # <-- SAFE replacement
    msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"USER:\n{query}\n\nRESEARCH MEMO:\n{research_memo}"}
    ]
    raw = ollama_chat(LLM_MODEL, msg)
    data = parse_first_json(raw)
    if not data:
        data = {"action": "draft", "content": "• No additional insights."}
    return data

def run_composer(drafts: Dict[str, str]) -> Dict[str, Any]:
    merged = "\n".join([f"{k}: {v}" for k, v in drafts.items()])
    msg = [{"role": "system", "content": SYSTEM_COMPOSER},
           {"role": "user", "content": f"SME DRAFTS:\n{merged}"}]
    raw = ollama_chat(LLM_MODEL, msg)
    return parse_first_json(raw) or {"status": "ready", "final": "\n".join(drafts.values())}

# =========================
# The autonomous loop
# =========================
def autonomous_answer(
    query: str,
    collection: str = "default",
    top_k_files: int = TOP_K_FILES_DEFAULT,
    top_k_web: int = TOP_K_WEB_DEFAULT,
    subjects_hint: Optional[List[str]] = None
) -> Dict[str, Any]:
    if not _ollama_ok():
        raise RuntimeError(f"Ollama not reachable at {OLLAMA_HOST}. Is the daemon running?")

    subjects = router_select_subjects(query, subjects_hint)
    # First research pass
    rag_hits = tool_local_search(query, top_k=top_k_files, collection=collection) if top_k_files > 0 else []
    web_hits = tool_web_search(query, top_k=top_k_web) if top_k_web > 0 else []
    research = build_research_memo(query, rag_hits, web_hits)

    # SMEs
    sme_outcomes: Dict[str, Any] = {}
    drafts: Dict[str, str] = {}
    need_more_prompts: List[str] = []

    for subj in subjects:
        out = run_sme_for_subject(subj, query, research)
        sme_outcomes[subj] = out
        if out.get("action") == "draft":
            drafts[subj] = out.get("content", "").strip()
        elif out.get("action") == "need_more_research":
            fq = out.get("followup_query") or ""
            if fq:
                need_more_prompts.append(fq)

    # Optional extra research round
    if need_more_prompts:
        followup_q = need_more_prompts[0]
        rag_hits2 = tool_local_search(followup_q, top_k=top_k_files, collection=collection) if top_k_files > 0 else []
        web_hits2 = tool_web_search(followup_q, top_k=top_k_web) if top_k_web > 0 else []
        research2 = build_research_memo(query, rag_hits + rag_hits2, web_hits + web_hits2, followup=followup_q)
        for subj, out in sme_outcomes.items():
            if out.get("action") != "draft":
                out2 = run_sme_for_subject(subj, query, research2)
                if out2.get("action") == "draft":
                    drafts[subj] = out2.get("content", "").strip()
                else:
                    drafts[subj] = "• More data may be required for a definitive answer."
        research = research2

    # Compose final
    compose = run_composer(drafts or {"General": "• No strong evidence found; consider refining the question."})
    if compose.get("status") == "ready":
        final_answer = compose.get("final", "")
    else:
        final_answer = "I need a bit more context or sources to be certain. Try adding one clarifying detail."

    return {
        "subjects": subjects,
        "research": research,
        "drafts": drafts,
        "answer": final_answer.strip()
    }

# =========================
# FastAPI
# =========================
app = FastAPI(title="Agentic AI Autonomous API", version="0.3.2")

class IngestRequest(BaseModel):
    folder: str
    collection: str = "default"

class AskRequest(BaseModel):
    query: str
    collection: str = "default"
    top_k_files: int = TOP_K_FILES_DEFAULT
    top_k_web: int = TOP_K_WEB_DEFAULT
    subjects_hint: Optional[List[str]] = None

class AskResponse(BaseModel):
    subjects: List[str]
    research: str
    drafts: Dict[str, str]
    answer: str

@app.get("/health")
def health():
    return {"ok": True, "model": LLM_MODEL, "ollama": _ollama_ok()}

@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        return tool_local_ingest(req.folder, req.collection)
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # Guard against literal "{prompt}" placeholders
    q = req.query if req.query and req.query != "{prompt}" else "Hello. Summarize the indexed knowledge base."
    out = autonomous_answer(
        query=q,
        collection=req.collection,
        top_k_files=req.top_k_files,
        top_k_web=req.top_k_web,
        subjects_hint=req.subjects_hint
    )
    return AskResponse(**out)

if __name__ == "__main__":
    import uvicorn
    # Run: uvicorn pig:app --reload --host 0.0.0.0 --port 8011
    uvicorn.run(app, host="127.0.0.1", port=8011)

