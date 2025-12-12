%%writefile /content/drive/MyDrive/Genesis_Project/genesis_viewer.py
import streamlit as st
import json
import os
import re
import time
import numpy as np
import faiss
import google.generativeai as genai
import urllib.parse

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(page_title="Torah AI: Command Center", layout="wide", initial_sidebar_state="expanded")

# PATHS (Colab Specific)
BASE_PATH = "."
VECTOR_DB_PATH = BASE_PATH

# DATABASE CONFIG
BOOKS_CONFIG = {
    "Genesis":     {"path": "genesis_translated_master.json",     "prefix": "Gen", "chapters": 50},
    "Exodus":      {"path": "exodus_translated_master.json",      "prefix": "Exo", "chapters": 40},
    "Leviticus":   {"path": "leviticus_translated_master.json",    "prefix": "Lev", "chapters": 27},
    "Numbers":     {"path": "numbers_translated_master.json",      "prefix": "Num", "chapters": 36},
    "Deuteronomy": {"path": "deuteronomy_translated_master.json",  "prefix": "Deu", "chapters": 34}
}

# --- SESSION STATE ---
if "current_verses" not in st.session_state: st.session_state.current_verses = []
if "chat_history" not in st.session_state: 
    st.session_state.chat_history = [{"role": "assistant", "content": "I am the Query Router. Ask for a specific range (e.g., 'Show me Genesis 1:3-10')."}]
if "audit_log" not in st.session_state: st.session_state.audit_log = []
if "translation_states" not in st.session_state: st.session_state.translation_states = {}

# --- HELPERS ---
@st.cache_data
def load_library():
    library = {}
    for book, config in BOOKS_CONFIG.items():
        path_v1 = f"{BASE_PATH}/{config['path']}"
        path_v2 = f"{BASE_PATH}/output/{config['path']}"
        target_path = path_v1 if os.path.exists(path_v1) else path_v2
        
        if os.path.exists(target_path):
            with open(target_path, 'r', encoding='utf-8') as f:
                library[book] = json.load(f)
    return library

@st.cache_resource
def load_engine():
    try:
        index_path = f"{VECTOR_DB_PATH}/torah_vectors.index"
        meta_path = f"{VECTOR_DB_PATH}/torah_metadata.json"
        if not os.path.exists(index_path): return None, None
        
        index = faiss.read_index(index_path)
        with open(meta_path, 'r', encoding='utf-8') as f:
            raw_meta = json.load(f)
            metadata = {int(k): v for k, v in raw_meta.items()}
        return index, metadata
    except Exception as e:
        return None, None

def translate_text(text, target_lang, api_key):
    if not api_key: return "⚠️ Missing API Key"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = (
            f"Act as a strict translator. Translate the following English text into {target_lang}. "
            f"Return ONLY the final translated text. "
            f"Do NOT provide options. Do NOT include explanations or notes. "
            f"Maintain the rugged, direct tone of the source.\n\n"
            f"Source: \"{text}\""
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def save_change(book_name, verse_id, new_ai_data):
    config = BOOKS_CONFIG[book_name]
    path_v1 = f"{BASE_PATH}/{config['path']}"
    path_v2 = f"{BASE_PATH}/output/{config['path']}"
    target_path = path_v1 if os.path.exists(path_v1) else path_v2
    try:
        with open(target_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            if item['id'] == verse_id:
                item['ai_data'] = new_ai_data
                break
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except:
        return False

# ==========================================
# 2. SIDEBAR
# ==========================================
library = load_library()
vector_index, vector_meta = load_engine()

with st.sidebar:
    st.title("⚙️ Torah AI")
    
    # API KEY
    try:
        sys_key = st.secrets.get("GOOGLE_API_KEY", None)
        admin_pass = st.secrets.get("MASTER_ADMIN_KEY", "TorahBuilder2025")
    except:
        sys_key = None
        admin_pass = "TorahBuilder2025"

    user_api_key = st.text_input("Google API Key", type="password")
    active_key = user_api_key if user_api_key else sys_key
    
    if active_key: 
        st.success("✅ Key Active")
        os.environ["GOOGLE_API_KEY"] = active_key
    else: 
        st.warning("⚠️ Key Required for AI")

    st.divider()

    modes = ["🚀 Command Center (Split)", "📖 Library (Classic)", "🔍 Search", "📝 Suggest Edit"]
    if user_api_key == admin_pass: 
        modes.append("🔒 Direct Edit (Admin)")
        
    app_mode = st.radio("Select Interface:", modes)
    
    st.divider()
    
    st.header("📊 System Status")
    if library: st.success(f"📚 Library: {sum(len(v) for v in library.values())} verses")
    if vector_index: st.success(f"🧠 Cortex: {vector_index.ntotal} vectors")
    else: st.warning("⚠️ Cortex Offline")

# ==========================================
# 3. MAIN INTERFACE
# ==========================================

def render_verse_card(verse, show_hebrew=True, show_english=True):
    with st.container():
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.caption(f"{verse['id']}")
        
        if show_hebrew:
            st.markdown(f"<div style='font-size: 1.8em; font-family: Serif; margin-bottom: 8px; direction: rtl;'>{verse['raw']}</div>", unsafe_allow_html=True)
        
        if show_english:
            rugged_text = verse['ai_data'].get('translation_rugged')
            st.markdown(f"<div style='font-size: 1.1em; font-style: italic; color: #444;'>{rugged_text}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🌐 Translate"):
        t_col1, t_col2 = st.columns([1, 4])
        with t_col1:
            target_lang = st.selectbox("Language", ["Select...", "Spanish", "French", "German", "Portuguese", "Russian", "Arabic"], key=f"sel_lang_{verse['id']}", label_visibility="collapsed")
        
        with t_col2:
            if target_lang and target_lang != "Select...":
                if not active_key:
                    st.error("API Key Required")
                else:
                    with st.spinner("Translating..."):
                        trans = translate_text(verse['ai_data'].get('translation_rugged'), target_lang, active_key)
                        st.markdown(f"**{target_lang}:** {trans}")
    st.divider()


# --- MODE 1: COMMAND CENTER ---
if "Command Center" in app_mode:
    # 🌟 CSS Removed: No more global column hacking that breaks other pages
    st.markdown("""<style>.chat-container { height: 500px; overflow-y: auto; }</style>""", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.subheader("💬 Command Agent")
        with st.container():
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]): st.write(msg["content"])
        
        st.markdown("---") 
        if prompt := st.chat_input("Enter command (e.g., 'Genesis 1:1-5')"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            found = False
            for bk in library:
                if bk.lower() in prompt.lower():
                    match = re.search(r'(\d+):(\d+)-(\d+)', prompt)
                    if match:
                        c, s, e = map(int, match.groups())
                        prefix = BOOKS_CONFIG[bk]['prefix']
                        data = []
                        capturing = False
                        for item in library[bk]:
                            if item['id'] == f"{prefix}{c}:{s}": capturing = True
                            if capturing:
                                data.append(item)
                                if item['id'] == f"{prefix}{c}:{e}": break
                        if data:
                            st.session_state.current_verses = data
                            st.session_state.chat_history.append({"role": "assistant", "content": f"✅ Routing {len(data)} verses."})
                            found = True
            if not found:
                 st.session_state.chat_history.append({"role": "assistant", "content": "I couldn't find that range. Try: **'Genesis 1:1-5'**"})
            st.rerun()

    with col_right:
        st.subheader("📄 Reader Workspace")
        if not st.session_state.current_verses:
            st.info("👈 **Waiting for Command...**")
        else:
            c1, c2 = st.columns([3,1])
            with c1: st.caption(f"Showing {len(st.session_state.current_verses)} Verses")
            with c2: toggle_hebrew = st.toggle("Show Hebrew", value=True)
            st.divider()
            for verse in st.session_state.current_verses:
                render_verse_card(verse, show_hebrew=toggle_hebrew)

# --- MODE 2: CLASSIC LIBRARY ---
elif "Library" in app_mode:
    st.title("📖 Classic Library")
    bk = st.selectbox("Select Book", list(BOOKS_CONFIG.keys()))
    bk_data = library[bk]
    prefix = BOOKS_CONFIG[bk]['prefix']
    
    st.subheader("Select Range")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**From:**")
        start_ch = st.selectbox("Start Chapter", range(1, BOOKS_CONFIG[bk]["chapters"]+1), key="s_ch")
        start_verses_count = len([v for v in bk_data if v['id'].startswith(f"{prefix}{start_ch}:")])
        start_v = st.selectbox("Start Verse", range(1, start_verses_count+1), key="s_v")
    with c2:
        st.markdown("**To:**")
        end_ch = st.selectbox("End Chapter", range(1, BOOKS_CONFIG[bk]["chapters"]+1), index=start_ch-1, key="e_ch")
        end_verses_count = len([v for v in bk_data if v['id'].startswith(f"{prefix}{end_ch}:")])
        default_idx = start_v - 1 if end_ch == start_ch else 0
        end_v = st.selectbox("End Verse", range(1, end_verses_count+1), index=min(default_idx, end_verses_count-1), key="e_v")

    st.markdown("---")
    t1, t2, t3 = st.columns([1,1,2])
    with t1: show_hebrew = st.toggle("Show Hebrew (Raw)", value=True)
    with t2: show_english = st.toggle("Show English (Rugged)", value=True)
    
    start_id = f"{prefix}{start_ch}:{start_v}"
    end_id = f"{prefix}{end_ch}:{end_v}"
    start_idx = next((i for i, item in enumerate(bk_data) if item['id'] == start_id), None)
    end_idx = next((i for i, item in enumerate(bk_data) if item['id'] == end_id), None)
    
    if start_idx is not None and end_idx is not None and start_idx <= end_idx:
        selection = bk_data[start_idx : end_idx + 1]
        st.info(f"Displaying {len(selection)} verses ({start_id} - {end_id})")
        for verse in selection:
            render_verse_card(verse, show_hebrew, show_english)

# --- MODE 3: SEARCH ---
elif "Search" in app_mode:
    st.title("🔍 Semantic Search")
    q = st.text_input("Search Concept:")
    if q and active_key and vector_index:
        genai.configure(api_key=active_key)
        res = genai.embed_content(model="models/text-embedding-004", content=q, task_type="retrieval_query")
        vec = np.array([res['embedding']]).astype('float32')
        D, I = vector_index.search(vec, 5)
        for idx in I[0]:
            v = vector_meta.get(idx)
            if not v: v = vector_meta.get(str(idx))
            if v:
                with st.expander(v['id']):
                     render_verse_card(v)

# --- MODE 4: SUGGEST EDIT (Confirmed Present) ---
elif "Suggest Edit" in app_mode:
    st.title("📝 Suggest Edit")
    st.info("Submit corrections via Gmail.")
    
    c_book, c_chap, c_verse = st.columns([2, 1, 1])
    with c_book: sel_book = st.selectbox("Book", list(BOOKS_CONFIG.keys()), key="edit_book")
    curr_config = BOOKS_CONFIG[sel_book]
    book_data = library[sel_book]
    with c_chap: sel_chap = st.selectbox("Chapter", range(1, curr_config["chapters"] + 1), key="edit_chap")
    with c_verse:
        prefix = curr_config['prefix']
        target_id_start = f"{prefix}{sel_chap}:"
        verses_in_chap = [i for i in book_data if i['id'].startswith(target_id_start)]
        v_count = len(verses_in_chap) if verses_in_chap else 1
        sel_verse = st.selectbox("Verse", range(1, v_count + 1), key="edit_verse")

    target_id = f"{prefix}{sel_chap}:{sel_verse}"
    verse_obj = next((item for item in book_data if item['id'] == target_id), None)

    if verse_obj:
        ai = verse_obj.get('ai_data', {})
        st.markdown(f"### {verse_obj['raw']}")
        
        with st.form("email_form"):
            user_edit = st.text_area("Propose Translation Edit:", value=ai.get('translation_rugged', ''))
            user_notes = st.text_input("Reason for change:")
            
            if st.form_submit_button("📨 Generate Gmail Draft"):
                subject = f"Correction: {sel_book} {target_id}"
                body = f"Hebrew: {verse_obj.get('raw')}\nCurrent: {ai.get('translation_rugged')}\n\nProposed: {user_edit}\nNotes: {user_notes}"
                safe_subject = urllib.parse.quote(subject)
                safe_body = urllib.parse.quote(body)
                gmail_link = f"https://mail.google.com/mail/?view=cm&fs=1&to=your_email@gmail.com&su={safe_subject}&body={safe_body}"
                
                st.markdown(f'<a href="{gmail_link}" target="_blank"><button style="background-color:#DB4437; color:white; padding:10px 15px; border:none; border-radius:5px; cursor:pointer;">📮 Open in Gmail</button></a>', unsafe_allow_html=True)

# --- MODE 5: ADMIN ---
elif "Admin" in app_mode:
    st.title("🔒 Admin Editor")
    bk = st.selectbox("Book", list(BOOKS_CONFIG.keys()), key="adm_bk")
    ch = st.number_input("Chapter", 1, 50, key="adm_ch")
    vs = st.number_input("Verse", 1, 100, key="adm_vs")
    prefix = BOOKS_CONFIG[bk]['prefix']
    target = f"{prefix}{ch}:{vs}"
    
    bk_data = library[bk] 
    verse = next((v for v in bk_data if v['id'] == target), None)
    
    if verse:
        st.markdown(f"Editing **{target}**")
        with st.form("edit"):
            new_txt = st.text_area("Translation", verse['ai_data'].get('translation_rugged'))
            if st.form_submit_button("Save"):
                verse['ai_data']['translation_rugged'] = new_txt
                save_change(bk, target, verse['ai_data'])
                st.success("Saved!")
