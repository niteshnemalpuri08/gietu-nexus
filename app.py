import streamlit as st
import os, requests, smtplib, shutil, json, hashlib, time, re
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from bs4 import BeautifulSoup
from pypdf import PdfReader

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM

from langchain_community.vectorstores import FAISS

# ───────────────────────────────────────────
#  CONFIG
# ───────────────────────────────────────────
st.set_page_config(
    page_title="GIETU Nexus · Digital Archaeology",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

EMAIL_SENDER      = "niteshnemalpuri17@gmail.com"
EMAIL_PASSWORD    = "pumi amru foqn cvfc"
TARGET_URL        = "https://www.giet.edu/happenings/notices/"
NOTICES_DIR       = "notices_vault"
INDEX_PATH        = "faiss_index_local"
HISTORY_FILE      = "chat_history.json"
METADATA_FILE     = "vault_metadata.json"
MAX_PDF_DL        = 10
MAX_PAGES         = 15
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 150

USERS = {
    "admin":   hashlib.sha256("gietu".encode()).hexdigest(),
    "student": hashlib.sha256("giet2024".encode()).hexdigest(),
    "faculty": hashlib.sha256("faculty123".encode()).hexdigest(),
}

ROLE_COLORS = {"admin": "#f59e0b", "student": "#22d3ee", "faculty": "#a78bfa"}

# ───────────────────────────────────────────
#  GLASSMORPHISM UI — UNIQUE AMBER/TEAL THEME
# ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg0:        #04080f;
    --bg1:        #080f1a;
    --bg2:        #0d1829;
    --glass:      rgba(8, 20, 40, 0.55);
    --glass-edge: rgba(255,255,255,0.08);
    --amber:      #f59e0b;
    --amber-dim:  rgba(245,158,11,0.18);
    --teal:       #2dd4bf;
    --teal-dim:   rgba(45,212,191,0.15);
    --rose:       #fb7185;
    --violet:     #a78bfa;
    --text:       #e8f4f8;
    --muted:      #64748b;
    --border:     rgba(255,255,255,0.07);
    --glow-a:     0 0 40px rgba(245,158,11,0.20);
    --glow-t:     0 0 40px rgba(45,212,191,0.20);
    --radius:     16px;
    --radius-sm:  10px;
}

/* ── Root ── */
html, body, .stApp {
    background: var(--bg0) !important;
    font-family: 'Outfit', sans-serif !important;
    color: var(--text) !important;
}

/* ── Animated mesh background ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 70% 50% at 10% 20%,  rgba(245,158,11,0.07) 0%, transparent 65%),
        radial-gradient(ellipse 60% 45% at 85% 75%,  rgba(45,212,191,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 40% 30% at 50% 50%,  rgba(167,139,250,0.04) 0%, transparent 55%);
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, rgba(8,15,28,0.97) 0%, rgba(4,8,15,0.99) 100%) !important;
    border-right: 1px solid var(--border) !important;
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* ── Global button ── */
.stButton > button {
    background: var(--glass) !important;
    border: 1px solid var(--glass-edge) !important;
    color: var(--teal) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: .04em !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 18px !important;
    transition: all .25s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: var(--teal-dim) !important;
    border-color: var(--teal) !important;
    box-shadow: var(--glow-t) !important;
    transform: translateY(-1px) !important;
    color: #fff !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div {
    background: rgba(8,20,40,0.7) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
    transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px rgba(45,212,191,0.12) !important;
    outline: none !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: rgba(8,20,40,0.80) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    backdrop-filter: blur(12px) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.12) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--glass) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid var(--glass-edge) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 12px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(8,20,40,0.6) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(8,20,40,0.5) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: var(--radius-sm) !important;
    color: var(--muted) !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
    transition: all .2s !important;
}
.stTabs [aria-selected="true"] {
    background: var(--amber-dim) !important;
    color: var(--amber) !important;
    border: 1px solid rgba(245,158,11,0.3) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--glass) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid var(--glass-edge) !important;
    border-radius: var(--radius) !important;
    padding: 16px !important;
}
[data-testid="stMetricValue"] {
    color: var(--amber) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.8rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 12px !important; }

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, var(--amber), var(--teal)) !important; border-radius: 4px !important; }
.stProgress > div { background: rgba(255,255,255,0.05) !important; border-radius: 4px !important; }

/* ── Slider ── */
.stSlider [data-baseweb="slider"] [role="slider"] { background: var(--amber) !important; }
.stSlider [data-baseweb="slider"] [data-testid="stSliderTrackFill"] { background: linear-gradient(90deg, var(--amber), var(--teal)) !important; }

/* ── Toggle ── */
.stToggle [data-baseweb="checkbox"] { accent-color: var(--teal) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg0); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--amber); }

/* ── Custom glass card ── */
.glass-card {
    background: var(--glass);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--glass-edge);
    border-radius: var(--radius);
    padding: 20px 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
    margin-bottom: 16px;
}

/* ── Hero banner ── */
.hero {
    text-align: center;
    padding: 32px 0 16px;
    position: relative;
}
.hero-title {
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 900;
    letter-spacing: -.03em;
    background: linear-gradient(135deg, var(--amber) 0%, #fcd34d 40%, var(--teal) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.hero-sub {
    font-size: 14px;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: .06em;
    margin-top: 8px;
}
.sdg-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(34,211,238,0.1);
    border: 1px solid rgba(34,211,238,0.3);
    color: var(--teal);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 5px 14px;
    border-radius: 20px;
    letter-spacing: .08em;
    margin-top: 12px;
}
.pill-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin: 14px 0;
}
.pill {
    flex: 1; min-width: 90px;
    background: var(--glass);
    border: 1px solid var(--glass-edge);
    border-radius: var(--radius-sm);
    padding: 12px 10px;
    text-align: center;
    backdrop-filter: blur(12px);
}
.pill .v { font-size: 1.4rem; font-weight: 900; font-family: 'JetBrains Mono', monospace; color: var(--amber); }
.pill .l { font-size: 10px; color: var(--muted); letter-spacing: .08em; text-transform: uppercase; margin-top: 2px; }
.source-chip {
    display: inline-block;
    background: rgba(245,158,11,0.12);
    border: 1px solid rgba(245,158,11,0.3);
    color: var(--amber);
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 2px 3px;
}
.answer-block {
    background: rgba(45,212,191,0.04);
    border-left: 3px solid var(--teal);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 14px 18px;
    margin: 8px 0;
}
.notice-card {
    background: var(--glass);
    border: 1px solid var(--glass-edge);
    border-radius: var(--radius-sm);
    padding: 14px 18px;
    margin-bottom: 10px;
    backdrop-filter: blur(12px);
    transition: border-color .2s, box-shadow .2s;
}
.notice-card:hover { border-color: rgba(245,158,11,0.3); box-shadow: var(--glow-a); }
.notice-meta { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--muted); margin-top: 6px; }
.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    letter-spacing: .06em;
}
.tag-exam     { background: rgba(251,113,133,0.15); color: var(--rose); border: 1px solid rgba(251,113,133,0.3); }
.tag-scholar  { background: rgba(167,139,250,0.15); color: var(--violet); border: 1px solid rgba(167,139,250,0.3); }
.tag-general  { background: rgba(45,212,191,0.12); color: var(--teal); border: 1px solid rgba(45,212,191,0.25); }
.tag-schedule { background: rgba(245,158,11,0.12); color: var(--amber); border: 1px solid rgba(245,158,11,0.25); }
.divider { height: 1px; background: var(--border); margin: 16px 0; }
.login-wrap {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.login-card {
    background: var(--glass);
    backdrop-filter: blur(32px);
    border: 1px solid var(--glass-edge);
    border-radius: 24px;
    padding: 40px 44px;
    width: 100%;
    max-width: 440px;
    box-shadow: 0 24px 64px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.07);
}
.user-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--amber-dim);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 13px;
    font-weight: 600;
    color: var(--amber);
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────
#  HELPERS — PERSISTENCE
# ───────────────────────────────────────────
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_history(msgs):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(msgs[-120:], f)
    except Exception:
        pass

def load_metadata():
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_metadata(meta):
    try:
        with open(METADATA_FILE, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

# ───────────────────────────────────────────
#  LOGIN
# ───────────────────────────────────────────
def _hash(pw): return hashlib.sha256(pw.encode()).hexdigest()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username      = ""
    st.session_state.role          = ""

if not st.session_state.authenticated:
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""
        <div style="padding:60px 0 10px;text-align:center;">
            <div style="font-size:3.5rem;margin-bottom:8px;">🏛️</div>
            <div style="font-size:2rem;font-weight:900;background:linear-gradient(135deg,#f59e0b,#2dd4bf);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
                GIETU Nexus
            </div>
            <div style="font-size:12px;color:#64748b;font-family:'JetBrains Mono',monospace;
                letter-spacing:.08em;margin-top:4px;">
                DIGITAL ARCHAEOLOGY ENGINE · SDG 4
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            uname = st.text_input("Username", placeholder="admin / student / faculty")
            pw    = st.text_input("Password", type="password", placeholder="••••••••")
            sub   = st.form_submit_button("⚡ Enter Vault", use_container_width=True)

        if sub:
            if uname in USERS and USERS[uname] == _hash(pw):
                st.session_state.authenticated = True
                st.session_state.username      = uname
                st.session_state.role          = uname
                st.rerun()
            else:
                st.error("Invalid credentials. Try admin / gietu")

        st.markdown("""
        <div style="text-align:center;margin-top:16px;font-family:'JetBrains Mono',monospace;
            font-size:11px;color:#334155;">
            admin/gietu &nbsp;·&nbsp; student/giet2024 &nbsp;·&nbsp; faculty/faculty123
        </div>""", unsafe_allow_html=True)
    st.stop()

# ───────────────────────────────────────────
#  LOAD EMBEDDINGS (cached)
# ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

with st.spinner("🧠 Warming up neural embeddings…"):
    embedding_model = load_embeddings()

# ───────────────────────────────────────────
#  CORE FUNCTIONS
# ───────────────────────────────────────────
def detect_category(text: str) -> str:
    text_l = text.lower()
    if any(w in text_l for w in ["exam", "examination", "test", "assessment", "marks", "result"]):
        return "exam"
    if any(w in text_l for w in ["scholarship", "fellowship", "stipend", "obc", "sc", "st", "financial"]):
        return "scholar"
    if any(w in text_l for w in ["schedule", "timetable", "routine", "calendar", "syllabus"]):
        return "schedule"
    return "general"

def scrape_giet(progress_cb=None):
    os.makedirs(NOTICES_DIR, exist_ok=True)
    meta     = load_metadata()
    new_files = []
    try:
        r    = requests.get(TARGET_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")
        links = [
            a["href"] for a in soup.find_all("a", href=True)
            if a["href"].lower().endswith(".pdf")
        ]
        for i, url in enumerate(links[:MAX_PDF_DL]):
            fname = url.split("/")[-1]
            path  = os.path.join(NOTICES_DIR, fname)
            if not os.path.exists(path):
                try:
                    res = requests.get(url, timeout=15)
                    with open(path, "wb") as f:
                        f.write(res.content)
                    new_files.append(fname)
                    meta[fname] = {
                        "url":        url,
                        "downloaded": datetime.now().isoformat(),
                        "size_kb":    round(len(res.content) / 1024, 1),
                    }
                except Exception:
                    pass
            if progress_cb:
                progress_cb((i + 1) / MAX_PDF_DL)
    except Exception as e:
        st.warning(f"Scrape issue: {e}")
    save_metadata(meta)
    return new_files

def process_vault(force=False):
    if os.path.exists(INDEX_PATH) and not force:
        return None, None

    raw_text   = ""
    file_stats = []
    meta       = load_metadata()
    os.makedirs(NOTICES_DIR, exist_ok=True)
    pdfs       = [f for f in os.listdir(NOTICES_DIR) if f.lower().endswith(".pdf")]

    bar = st.progress(0, text="Analysing documents…")
    for idx, fname in enumerate(pdfs):
        path = os.path.join(NOTICES_DIR, fname)
        try:
            reader = PdfReader(path)
            text   = ""
            method = "Native PDF"

            for page in reader.pages[:MAX_PAGES]:
                text += page.extract_text() or ""

            if len(text.strip()) < 200:
                method = "Vision OCR"
                try:
                    import easyocr
                    ocr    = easyocr.Reader(["en"], gpu=False)
                    result = ocr.readtext(path, detail=0, paragraph=True)
                    text   = " ".join(result)
                except Exception:
                    pass

            text = " ".join(text.split())
            if text.strip():
                cat = detect_category(text)
                raw_text += f"\nSOURCE: {fname}\nCATEGORY: {cat}\nCONTENT: {text}\n"
                meta[fname] = {**meta.get(fname, {}), "category": cat, "method": method, "indexed": datetime.now().isoformat()}
                file_stats.append({"File": fname, "Category": cat, "Method": method, "Pages": min(len(reader.pages), MAX_PAGES)})
        except Exception:
            pass
        bar.progress((idx + 1) / max(len(pdfs), 1), text=f"Processing {fname}…")

    bar.empty()
    save_metadata(meta)
    return raw_text, file_stats

def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks   = splitter.split_text(text)
    vs       = FAISS.from_texts(chunks, embedding_model)
    vs.save_local(INDEX_PATH)
    return vs

def load_vector_store():
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
    return None

def text_to_speech(text: str):
    try:
        from gtts import gTTS
        tts   = gTTS(text=text[:450], lang="en")
        fname = f"audio_{int(time.time())}.mp3"
        tts.save(fname)
        return fname
    except Exception:
        return None

def send_email(to_addr, subject, body):
    """
    Returns (True, "") on success, or (False, error_message) on failure.
    Tries SSL port 465 first, then STARTTLS port 587 as fallback.
    """
    msg = MIMEMultipart("alternative")
    msg["From"]    = EMAIL_SENDER
    msg["To"]      = to_addr
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # ── Attempt 1: SSL on port 465 ──────────────────────────────
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as s:
            s.ehlo()
            s.login(EMAIL_SENDER, EMAIL_PASSWORD)
            s.sendmail(EMAIL_SENDER, to_addr, msg.as_string())
        return True, ""
    except smtplib.SMTPAuthenticationError as e:
        return False, f"AUTH_FAILED: {e.smtp_error.decode() if hasattr(e, 'smtp_error') else str(e)}"
    except smtplib.SMTPRecipientsRefused:
        return False, "INVALID_RECIPIENT: The recipient address was rejected by the server."
    except Exception as e1:
        pass  # try STARTTLS fallback

    # ── Attempt 2: STARTTLS on port 587 ─────────────────────────
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(EMAIL_SENDER, EMAIL_PASSWORD)
            s.sendmail(EMAIL_SENDER, to_addr, msg.as_string())
        return True, ""
    except smtplib.SMTPAuthenticationError as e:
        err = e.smtp_error.decode() if hasattr(e, "smtp_error") else str(e)
        if "BadCredentials" in err or "534" in str(e):
            return False, "APP_PASSWORD_NEEDED"
        return False, f"AUTH_FAILED: {err}"
    except Exception as e2:
        return False, str(e2)

def count_docs():
    if not os.path.exists(NOTICES_DIR):
        return 0
    return len([f for f in os.listdir(NOTICES_DIR) if f.lower().endswith(".pdf")])

def get_recent_notices(days=30):
    meta  = load_metadata()
    cutoff = datetime.now() - timedelta(days=days)
    return {k: v for k, v in meta.items() if "downloaded" in v and datetime.fromisoformat(v["downloaded"]) > cutoff}

# ───────────────────────────────────────────
#  SIDEBAR
# ───────────────────────────────────────────
rc = ROLE_COLORS.get(st.session_state.role, "#64748b")
with st.sidebar:
    st.markdown(f"""
    <div style="padding:20px 0 12px;">
        <div class="user-chip" style="color:{rc};background:rgba(0,0,0,0.3);border-color:{rc}40;">
            👤 {st.session_state.username} <span style="font-size:10px;opacity:.6;">({st.session_state.role})</span>
        </div>
    </div>""", unsafe_allow_html=True)

    doc_count   = count_docs()
    index_ready = os.path.exists(INDEX_PATH)

    st.markdown(f"""
    <div class="pill-row">
        <div class="pill"><div class="v">{doc_count}</div><div class="l">Docs</div></div>
        <div class="pill"><div class="v">{'✅' if index_ready else '❌'}</div><div class="l">Index</div></div>
        <div class="pill"><div class="v">{len(get_recent_notices())}</div><div class="l">New</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("**⚙️ Pipeline Control**")

    if st.button("🔄 Sync & Index Notices"):
        with st.status("🏛️ Digital Archaeology Running…", expanded=True) as status:
            st.write("🌐 Scraping giet.edu notices…")
            bar_scrape = st.progress(0)
            new = scrape_giet(progress_cb=lambda p: bar_scrape.progress(p))
            bar_scrape.empty()
            if new:
                st.toast(f"📥 {len(new)} new notice(s) downloaded!")
                for n in new: st.write(f"  📄 {n}")
            else:
                st.write("  ℹ️ No new files found.")

            st.write("👁️ OCR + PDF text extraction…")
            text, stats = process_vault(force=True)

            if text:
                st.write("🧠 Building FAISS vector index…")
                st.session_state.vector_store = build_vector_store(text)
                st.write(f"✅ Indexed **{len(stats)}** document(s)!")
                st.dataframe(stats, use_container_width=True, hide_index=True)
                status.update(label="✅ Pipeline complete!", state="complete")
            else:
                st.session_state.vector_store = load_vector_store()
                st.info("Loaded existing index.")
                status.update(label="⚡ Cache loaded", state="complete")

    if st.button("⚡ Quick Load (Cache)"):
        vs = load_vector_store()
        if vs:
            st.session_state.vector_store = vs
            st.success("Vector store loaded from cache!")
        else:
            st.warning("No index found. Run Sync first.")

    if st.button("🗑️ Reset Index"):
        if os.path.exists(INDEX_PATH):
            shutil.rmtree(INDEX_PATH)
        st.session_state.pop("vector_store", None)
        st.success("Index cleared.")

    if st.button("💬 Clear Chat"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Memory cleared. Ask me anything about notices.",
            "timestamp": datetime.now().isoformat()
        }]
        save_history(st.session_state.messages)
        st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("**🤖 LLM Settings**")
    llm_model = st.selectbox("Model", ["phi3", "llama3", "mistral", "gemma", "llama3.2"], index=0)
    top_k     = st.slider("Context chunks (K)", 2, 8, 4)
    tts_on    = st.toggle("🔊 Voice replies", value=False)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("**📧 Email Summary**")
    email_to = st.text_input("Recipient", placeholder="student@example.com")
    if st.button("📨 Send Session Summary"):
        if not email_to:
            st.warning("Enter recipient email.")
        else:
            msgs   = st.session_state.get("messages", [])
            recent = [m["content"] for m in msgs if m["role"] == "assistant"][-5:]
            body   = "GIETU Nexus — Recent Answers\n\n" + "\n\n---\n\n".join(recent) if recent else "No answers yet."
            ok, err = send_email(email_to, "GIETU Nexus · Notice Summary", body)
            if ok:
                st.success("📬 Email sent!")
            elif err == "APP_PASSWORD_NEEDED":
                st.error("❌ Gmail blocked sign-in.")
                st.info(
                    "**Fix in 3 steps:**\n\n"
                    "1. Go to [myaccount.google.com/security](https://myaccount.google.com/security)\n"
                    "2. Enable **2-Step Verification** (required)\n"
                    "3. Search **'App passwords'** → Create one for *Mail* → "
                    "paste the 16-char code as `EMAIL_PASSWORD` in `app.py`\n\n"
                    "_Example:_ `abcd efgh ijkl mnop`"
                )
            elif "AUTH_FAILED" in err:
                st.error(f"❌ Authentication failed.\n\n`{err}`\n\nCheck `EMAIL_SENDER` and `EMAIL_PASSWORD` in `app.py`.")
            elif "INVALID_RECIPIENT" in err:
                st.error("❌ Invalid recipient address. Check the email you entered.")
            else:
                st.error(f"❌ Send failed: `{err}`")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if st.button("🚪 Logout"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

# ───────────────────────────────────────────
#  MAIN CONTENT
# ───────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🏛️ GIETU Nexus</div>
    <div class="hero-sub">DIGITAL ARCHAEOLOGY ENGINE · SEMANTIC NOTICE SEARCH</div>
    <div><span class="sdg-badge">🌍 SDG 4 — Quality Education</span></div>
</div>
""", unsafe_allow_html=True)

tab_search, tab_chat, tab_vault, tab_recent, tab_about = st.tabs([
    "🔍 Semantic Search", "💬 Chat Assistant", "📂 Notice Vault", "🆕 Recent Notices", "ℹ️ About"
])

# ─────────────────────────
#  TAB 1 · SEMANTIC SEARCH  (Perplexity-style)
# ─────────────────────────
with tab_search:
    st.markdown("""
    <div class="glass-card">
        <div style="font-size:1.2rem;font-weight:800;margin-bottom:4px;">Ask Anything About Notices</div>
        <div style="color:#64748b;font-size:13px;">Perplexity-style semantic search across all indexed university documents</div>
    </div>""", unsafe_allow_html=True)

    example_queries = [
        "What is the syllabus for 3rd semester Java lab?",
        "Show me all scholarship notices for OBC students",
        "When is the next examination schedule?",
        "Bus route and transport notices",
        "Mandatory disclosure requirements",
    ]

    st.markdown("**💡 Try these:**")
    eq_cols = st.columns(len(example_queries))
    if "search_prefill" not in st.session_state:
        st.session_state.search_prefill = ""

    for i, eq in enumerate(example_queries):
        with eq_cols[i]:
            if st.button(eq[:28] + "…", key=f"eq_{i}", help=eq):
                st.session_state.search_prefill = eq

    search_q = st.text_input(
        "Search query",
        value=st.session_state.search_prefill,
        placeholder="e.g. scholarship deadline for OBC students last month…",
        label_visibility="collapsed"
    )

    filter_col1, filter_col2, _ = st.columns([1, 1, 4])
    with filter_col1:
        cat_filter = st.selectbox("Category", ["All", "exam", "scholar", "schedule", "general"])
    with filter_col2:
        date_filter = st.selectbox("Time range", ["All time", "Last 7 days", "Last 30 days", "Last 90 days"])

    if st.button("🔍 Search", use_container_width=False) and search_q.strip():
        if "vector_store" not in st.session_state:
            vs = load_vector_store()
            if vs:
                st.session_state.vector_store = vs
            else:
                st.warning("⚠️ No index found. Click **Sync & Index Notices** in the sidebar first.")
                st.stop()

        with st.spinner("🔍 Scanning the notice vault…"):
            try:
                # Semantic retrieval
                docs = st.session_state.vector_store.similarity_search(search_q, k=top_k + 2)

                # Filter by category if needed
                if cat_filter != "All":
                    docs = [d for d in docs if cat_filter in d.page_content.lower()]

                if not docs:
                    st.info("No relevant notices found. Try a different query or sync more documents.")
                    st.stop()

                ctx = "\n\n".join([d.page_content for d in docs])

                # LLM answer
                llm = OllamaLLM(model=llm_model)
                prompt = f"""You are GIETU Nexus, a precise university notice search engine aligned with SDG 4 (Quality Education).
Your mission: extract and present DIRECT, FACTUAL answers from the notice documents.

Rules:
- Answer ONLY from the CONTEXT provided
- If information is missing, say: "This specific detail isn't in the current indexed notices."
- Be concise and structured
- Highlight deadlines, dates, and key requirements clearly
- For scholarship queries, mention eligibility criteria explicitly

CONTEXT:
{ctx}

QUERY: {search_q}

STRUCTURED ANSWER:"""

                response = llm.invoke(prompt)

                # Render answer
                st.markdown(f"""
                <div class="answer-block">
                    <div style="font-size:11px;font-family:'JetBrains Mono',monospace;color:#2dd4bf;
                        letter-spacing:.08em;margin-bottom:10px;">▶ NEXUS ANSWER</div>
                    {response.replace(chr(10), '<br>')}
                </div>""", unsafe_allow_html=True)

                # Source chunks
                with st.expander("📎 Source evidence (retrieved chunks)"):
                    for i, doc in enumerate(docs[:top_k], 1):
                        cat = detect_category(doc.page_content)
                        st.markdown(f'<span class="tag tag-{cat}">{cat.upper()}</span> **Chunk {i}**', unsafe_allow_html=True)
                        st.code(doc.page_content[:500], language="text")

                if tts_on:
                    audio = text_to_speech(response)
                    if audio: st.audio(audio)

            except Exception as e:
                st.error(f"Search error: {e}")

# ─────────────────────────
#  TAB 2 · CHAT ASSISTANT
# ─────────────────────────
with tab_chat:
    if "messages" not in st.session_state:
        persisted = load_history()
        st.session_state.messages = persisted if persisted else [{
            "role":      "assistant",
            "content":   "👋 Hello! I'm your GIETU Nexus assistant. Ask me anything about university notices, exams, scholarships, or schedules.",
            "timestamp": datetime.now().isoformat()
        }]

    # Render history
    for msg in st.session_state.messages:
        av = "🏛️" if msg["role"] == "assistant" else "🧑‍🎓"
        with st.chat_message(msg["role"], avatar=av):
            if msg["role"] == "assistant":
                st.markdown(f'<div class="answer-block">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])
            ts = msg.get("timestamp", "")
            if ts:
                st.caption(f"🕐 {ts[:16].replace('T',' ')}")

    if prompt := st.chat_input("Ask about exams, scholarships, bus routes, syllabus…"):
        ts_now = datetime.now().isoformat()
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": ts_now})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)

        if "vector_store" not in st.session_state:
            vs = load_vector_store()
            if vs:
                st.session_state.vector_store = vs
            else:
                with st.chat_message("assistant", avatar="🏛️"):
                    st.warning("⚠️ Index missing. Please run **Sync & Index Notices** in the sidebar.")
                st.stop()

        with st.chat_message("assistant", avatar="🏛️"):
            with st.spinner("⚡ Searching notices…"):
                try:
                    docs = st.session_state.vector_store.similarity_search(prompt, k=top_k)
                    ctx  = "\n\n".join([d.page_content for d in docs])

                    with st.expander("🕵️ Evidence retrieved"):
                        st.code(ctx[:1200], language="text")

                    llm = OllamaLLM(model=llm_model)
                    full_prompt = f"""You are a helpful GIETU university assistant. Use ONLY the CONTEXT below.
If the answer isn't there, say: "I couldn't find that in the current notices."
Be concise, highlight dates and deadlines.

CONTEXT:
{ctx}

QUESTION: {prompt}
ANSWER:"""
                    response = llm.invoke(full_prompt)

                    st.markdown(f'<div class="answer-block">{response}</div>', unsafe_allow_html=True)
                    ts_r = datetime.now().isoformat()
                    st.caption(f"🕐 {ts_r[:16].replace('T',' ')} · {llm_model} · k={top_k}")

                    st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": ts_r})
                    save_history(st.session_state.messages)

                    c1, c2, _ = st.columns([1, 1, 10])
                    c1.button("👍", key=f"up_{ts_r}")
                    c2.button("👎", key=f"dn_{ts_r}")

                    if tts_on:
                        audio = text_to_speech(response)
                        if audio: st.audio(audio)

                except Exception as e:
                    st.error(f"Error: {e}")

# ─────────────────────────
#  TAB 3 · NOTICE VAULT
# ─────────────────────────
with tab_vault:
    st.markdown('<div class="glass-card"><b>📂 Notice Vault</b> — All locally indexed PDF documents</div>', unsafe_allow_html=True)

    if not os.path.exists(NOTICES_DIR):
        st.info("Vault is empty. Run **Sync & Index Notices** to populate.")
    else:
        pdfs = sorted([f for f in os.listdir(NOTICES_DIR) if f.lower().endswith(".pdf")])
        meta = load_metadata()

        filt_col1, filt_col2 = st.columns([2, 1])
        with filt_col1:
            search_v = st.text_input("🔍 Filter", placeholder="Search by name…", key="vault_filter")
        with filt_col2:
            cat_v = st.selectbox("Category", ["All", "exam", "scholar", "schedule", "general"], key="vault_cat")

        filtered = pdfs
        if search_v:
            filtered = [p for p in filtered if search_v.lower() in p.lower()]
        if cat_v != "All":
            filtered = [p for p in filtered if meta.get(p, {}).get("category") == cat_v]

        st.markdown(f"**{len(filtered)}** of **{len(pdfs)}** document(s)")

        for fname in filtered:
            path  = os.path.join(NOTICES_DIR, fname)
            size  = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
            mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M") if os.path.exists(path) else "—"
            cat   = meta.get(fname, {}).get("category", "general")
            method = meta.get(fname, {}).get("method", "—")

            tag_class = f"tag-{cat}"
            st.markdown(f"""
            <div class="notice-card">
                <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                    <span style="font-size:1.1rem;">📄</span>
                    <span style="font-weight:600;flex:1;">{fname}</span>
                    <span class="tag {tag_class}">{cat.upper()}</span>
                </div>
                <div class="notice-meta">
                    📦 {size:.1f} KB &nbsp;·&nbsp; 🕐 {mtime} &nbsp;·&nbsp; 🔬 {method}
                </div>
            </div>""", unsafe_allow_html=True)

            exp_col1, exp_col2, exp_col3 = st.columns([2, 2, 1])
            with exp_col1:
                if st.button("👁️ Preview", key=f"prev_{fname}"):
                    try:
                        r = PdfReader(path)
                        preview = r.pages[0].extract_text() or "(No extractable text on page 1)"
                        st.text_area("Page 1", preview[:1800], height=160, key=f"ta_{fname}")
                    except Exception as e:
                        st.error(f"Preview failed: {e}")
            with exp_col2:
                if st.button("🔍 Search in this doc", key=f"srch_{fname}"):
                    st.info(f"Use the Semantic Search tab and mention '{fname}' in your query.")
            with exp_col3:
                if st.session_state.role == "admin":
                    if st.button("🗑️ Del", key=f"del_{fname}"):
                        os.remove(path)
                        if os.path.exists(INDEX_PATH):
                            shutil.rmtree(INDEX_PATH)
                        st.session_state.pop("vector_store", None)
                        st.success(f"Deleted. Re-sync to update index.")
                        st.rerun()

# ─────────────────────────
#  TAB 4 · RECENT NOTICES
# ─────────────────────────
with tab_recent:
    st.markdown('<div class="glass-card"><b>🆕 Recent Notices</b> — Notices downloaded in the last 30 days</div>', unsafe_allow_html=True)

    days_filter = st.slider("Show notices from last N days", 7, 90, 30)
    recent      = get_recent_notices(days=days_filter)
    meta        = load_metadata()

    if not recent:
        st.info("No recent notices. Run **Sync & Index Notices** to fetch the latest from giet.edu.")
    else:
        st.markdown(f"**{len(recent)}** notice(s) in the last {days_filter} days")
        for fname, info in sorted(recent.items(), key=lambda x: x[1].get("downloaded", ""), reverse=True):
            cat = info.get("category", meta.get(fname, {}).get("category", "general"))
            tag_class = f"tag-{cat}"
            dl_time = info.get("downloaded", "")[:16].replace("T", " ")
            size    = info.get("size_kb", "?")
            url     = info.get("url", "#")

            st.markdown(f"""
            <div class="notice-card">
                <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                    <span style="font-size:1.1rem;">📋</span>
                    <span style="font-weight:600;flex:1;">{fname}</span>
                    <span class="tag {tag_class}">{cat.upper()}</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#64748b;">NEW</span>
                </div>
                <div class="notice-meta">
                    📅 Downloaded: {dl_time} &nbsp;·&nbsp; 📦 {size} KB
                </div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────
#  TAB 5 · ABOUT
# ─────────────────────────
with tab_about:
    st.markdown("""
    <div class="glass-card">
        <div style="font-size:1.3rem;font-weight:800;margin-bottom:12px;">🏛️ Digital Archaeology Engine</div>
        <div style="color:#94a3b8;line-height:1.8;">
        University websites are often <b style="color:#f59e0b;">PDF graveyards</b> — critical notices buried in
        non-searchable scanned images. Students miss exam deadlines, scholarship cutoffs, and route changes
        because finding specific information requires clicking through dozens of links.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="glass-card">
            <div style="font-weight:700;margin-bottom:10px;color:#2dd4bf;">🔧 How It Works</div>
            <div style="color:#94a3b8;font-size:14px;line-height:1.9;">
            1. <b>Scrape</b> — Periodically fetches PDFs from giet.edu/notices<br>
            2. <b>OCR</b> — EasyOCR reads text from scanned/image PDFs<br>
            3. <b>Embed</b> — all-MiniLM-L6-v2 converts text to vectors<br>
            4. <b>Index</b> — FAISS stores vectors for sub-second retrieval<br>
            5. <b>Answer</b> — Ollama LLM generates direct answers from context
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="glass-card">
            <div style="font-weight:700;margin-bottom:10px;color:#f59e0b;">🌍 SDG 4 Alignment</div>
            <div style="color:#94a3b8;font-size:14px;line-height:1.9;">
            <b>Quality Education</b> — ensuring equitable access to information.<br><br>
            Students from all backgrounds can now find:<br>
            • Scholarship deadlines (OBC, SC, ST, merit-based)<br>
            • Exam schedules and syllabus details<br>
            • Fee structures and mandatory disclosures<br>
            • Bus routes and campus services
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#475569;">
    <pre style="background:transparent;border:none;padding:0;margin:0;color:#64748b;">
  Student Query
       │
       ▼
  ┌────────────────────────────────────────────────┐
  │  GIETU Nexus — Digital Archaeology Engine      │
  │                                                │
  │  ┌──────────┐   ┌──────────┐  ┌────────────┐  │
  │  │ Scraper  │──▶│ OCR/PDF  │─▶│  Embedder  │  │
  │  │ giet.edu │   │ EasyOCR  │  │ MiniLM-L6  │  │
  │  └──────────┘   └──────────┘  └─────┬──────┘  │
  │                                     │          │
  │                               ┌─────▼──────┐  │
  │                               │   FAISS    │  │
  │                               │  Vector DB │  │
  │                               └─────┬──────┘  │
  │                                     │ top-K   │
  │                               ┌─────▼──────┐  │
  │                               │ Ollama LLM │  │
  │                               │ phi3/llama │  │
  │                               └─────┬──────┘  │
  └─────────────────────────────────────┼──────────┘
                                        ▼
                              Direct Answer + TTS
    </pre>
    </div>
    """, unsafe_allow_html=True)

    # ── Team Credits ──────────────────────────────────────────────
    st.html("""
<div style="margin-top:32px;font-family:Outfit,sans-serif;">

  <div style="text-align:center;margin-bottom:24px;">
    <div style="font-size:11px;color:#64748b;letter-spacing:.15em;text-transform:uppercase;margin-bottom:8px;">
      Built with &#10084;&#65039; by
    </div>
    <div style="font-size:1.7rem;font-weight:900;
      background:linear-gradient(135deg,#f59e0b 0%,#fcd34d 50%,#2dd4bf 100%);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
      background-clip:text;letter-spacing:-.01em;">
      Team Tech Four
    </div>
  </div>

  <div style="display:flex;gap:14px;flex-wrap:wrap;justify-content:center;margin-bottom:24px;">

    <div style="background:linear-gradient(135deg,rgba(245,158,11,0.15),rgba(245,158,11,0.05));
      border:1px solid rgba(245,158,11,0.4);border-radius:16px;padding:22px 28px;
      text-align:center;min-width:190px;position:relative;overflow:hidden;
      box-shadow:0 0 32px rgba(245,158,11,0.14);">
      <div style="position:absolute;top:10px;right:12px;background:rgba(245,158,11,0.2);
        border:1px solid rgba(245,158,11,0.4);color:#f59e0b;font-size:9px;
        padding:2px 8px;border-radius:20px;letter-spacing:.1em;font-weight:700;">
        TEAM LEAD
      </div>
      <div style="font-size:2.2rem;margin-bottom:10px;">&#128104;&#8205;&#128187;</div>
      <div style="font-weight:800;font-size:1.05rem;color:#fcd34d;">Nitesh Nemalpuri</div>
      <div style="font-size:11px;color:#94a3b8;margin-top:5px;">Architecture &#183; AI &#183; Backend</div>
    </div>

    <div style="background:rgba(8,20,40,0.7);border:1px solid rgba(255,255,255,0.09);
      border-radius:16px;padding:22px 28px;text-align:center;min-width:170px;">
      <div style="font-size:2.2rem;margin-bottom:10px;">&#129489;&#8205;&#128187;</div>
      <div style="font-weight:700;font-size:1rem;color:#e2f0fb;">Sameer Ranjan Nayak</div>
      <div style="font-size:11px;color:#64748b;margin-top:5px;">Development &#183; Testing</div>
    </div>

    <div style="background:rgba(8,20,40,0.7);border:1px solid rgba(255,255,255,0.09);
      border-radius:16px;padding:22px 28px;text-align:center;min-width:170px;">
      <div style="font-size:2.2rem;margin-bottom:10px;">&#129489;&#8205;&#128187;</div>
      <div style="font-weight:700;font-size:1rem;color:#e2f0fb;">Somen Mishra</div>
      <div style="font-size:11px;color:#64748b;margin-top:5px;">Development &#183; Research</div>
    </div>

    <div style="background:rgba(8,20,40,0.7);border:1px solid rgba(255,255,255,0.09);
      border-radius:16px;padding:22px 28px;text-align:center;min-width:170px;">
      <div style="font-size:2.2rem;margin-bottom:10px;">&#129489;&#8205;&#128187;</div>
      <div style="font-weight:700;font-size:1rem;color:#e2f0fb;">Priyanshu Panda</div>
      <div style="font-size:11px;color:#64748b;margin-top:5px;">Development &#183; UI/UX</div>
    </div>

  </div>

  <div style="text-align:center;padding:16px 20px;background:rgba(8,20,40,0.6);
    border:1px solid rgba(255,255,255,0.06);border-radius:12px;">
    <div style="font-size:11px;color:#475569;letter-spacing:.06em;">
      &#127963;&#65039; &nbsp; GIETU Nexus &nbsp;&#183;&nbsp; Digital Archaeology Engine &nbsp;&#183;&nbsp;
      SDG 4 &#8212; Quality Education &nbsp;&#183;&nbsp;
      <span style="color:#f59e0b;font-weight:700;">Team Tech Four</span> &#169; 2026
    </div>
  </div>

</div>
""")