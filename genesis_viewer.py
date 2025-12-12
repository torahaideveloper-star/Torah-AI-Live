import streamlit as st
import json
import os
import numpy as np
import faiss
import google.generativeai as genai
import urllib.parse

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(page_title="Torah AI: Master System", layout="wide")

# 🔒 MASTER ADMIN KEY (For Admin Mode Access Only)
MASTER_ADMIN_KEY = "PASTE_YOUR_KEY_HERE"

# PATHS
BASE_PATH = "."
VECTOR_DB_PATH = f"{BASE_PATH}/vector_db"
DEVELOPER_EMAIL = "torahaideveloper@gmail.com"

# Master Database Config
BOOKS_CONFIG = {
    "Genesis":     {"path": "genesis_translated_master.json",     "prefix": "Gen", "chapters": 50},
    "Exodus":      {"path": "exodus_translated_master.json",      "prefix": "Exo", "chapters": 40},
    "Leviticus":   {"path": "leviticus_translated_master.json",    "prefix": "Lev", "chapters": 27},
    "Numbers":     {"path": "numbers_translated_master.json",      "prefix": "Num", "chapters": 36},
    "Deuteronomy": {"path": "deuteronomy_translated_master.json",  "prefix": "Deu", "chapters": 34}
}
# --- HELPER: ROBUST ROOT DISPLAY ---
def display_roots(ai_data):
    roots = ai_data.get('hebrew_roots', [])
    extracted_roots = []
    if isinstance(roots, list):
        for item in roots:
            if isinstance(item, dict):
                root_val = item.get('root')
                if root_val: extracted_roots.append(root_val)
                else: extracted_roots.append("Unknown")
            elif isinstance(item, str):
                extracted_roots.append(item)
    if extracted_roots:
        return ", ".join(str(r) for r in extracted_roots)
    return "N/A"

# --- HELPER: SAVE FUNCTION ---
def save_change(book_name, verse_id, new_ai_data):
    config = BOOKS_CONFIG[book_name]
    full_path = f"{BASE_PATH}/{config['path']}"
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            if item['id'] == verse_id:
                item['ai_data'] = new_ai_data
                break
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False

# ==========================================
# 2. SIDEBAR: CONFIGURATION
# ==========================================
with st.sidebar:
    st.title("⚙️ Torah AI")
    
    # 1. API KEY INPUT (Required for AI Features)
    st.markdown("### 🔑 Access")
    user_api_key = st.text_input("Enter Google API Key", type="password", help="Enter your own key to use Chat/Search features.")
    
    if user_api_key and "AIza" in user_api_key:
        st.success("✅ Key Detected")
    else:
        st.warning("⚠️ Enter Key to unlock AI features")
    
    st.divider()
    
    # APP MODE SELECTOR
    available_modes = ["📖 Library", "🔍 Semantic Search", "💬 AI Chat", "📝 Suggest Edit"]
    
    if user_api_key == MASTER_ADMIN_KEY:
        available_modes.append("🔒 Direct Edit (Admin)")
        st.success("🔓 Admin Unlocked")
        
    app_mode = st.radio("Select Interface:", available_modes)
    
    st.divider()
    st.header("📊 System Status")

    # Load All Data
    @st.cache_data
    def load_all_books():
        library = {}
        total_verses = 0
        for book, config in BOOKS_CONFIG.items():
            path = f"{BASE_PATH}/{config['path']}"
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    library[book] = data
                    total_verses += len(data)
        return library, total_verses

    library, count = load_all_books()
    
    if app_mode == "🔒 Direct Edit (Admin)":
        pass 
    
    if library:
        st.success(f"📚 Library: {count} Verses")
    else:
        st.error("❌ Library Offline")
        st.stop()

    # Load Vector Engine
    @st.cache_resource
    def load_engine():
        try:
            index = faiss.read_index(f"{VECTOR_DB_PATH}/torah_vectors.index")
            with open(f"{VECTOR_DB_PATH}/torah_metadata.json", 'r', encoding='utf-8') as f:
                raw_meta = json.load(f)
                metadata = {int(k): v for k, v in raw_meta.items()}
            return index, metadata
        except Exception as e:
            return None, None

    vector_index, vector_meta = load_engine()
    if vector_index:
        st.success(f"🧠 Cortex: {vector_index.ntotal} Vectors")
    else:
        st.warning("⚠️ Cortex Offline")


# ==========================================
# MODE 1: LIBRARY (READ & EDIT)
# ==========================================
if app_mode == "📖 Library":
    st.title("📖 Torah Library")
    
    c_book, c_chap, c_verse = st.columns([2, 1, 1])
    
    with c_book:
        sel_book = st.selectbox("Book", list(BOOKS_CONFIG.keys()))
        curr_config = BOOKS_CONFIG[sel_book]
        book_data = library[sel_book]
    
    with c_chap:
        sel_chap = st.selectbox("Chapter", range(1, curr_config["chapters"] + 1))
        
    with c_verse:
        prefix = curr_config['prefix']
        target_id_start = f"{prefix}{sel_chap}:"
        verses_in_chap = [i for i in book_data if i['id'].startswith(target_id_start)]
        v_count = len(verses_in_chap) if verses_in_chap else 1
        sel_verse = st.selectbox("Verse", range(1, v_count + 1))

    target_id = f"{prefix}{sel_chap}:{sel_verse}"
    verse_obj = next((item for item in book_data if item['id'] == target_id), None)
    
    if verse_obj:
        ai = verse_obj.get('ai_data', {})
        st.divider()
        st.markdown(f"### {verse_obj['raw']}")
        st.caption(f"**{target_id}**")
        st.success(f"**{ai.get('translation_rugged', 'N/A')}**")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Tone", ai.get('tone', 'N/A'))
        m2.metric("Difficulty", f"{ai.get('difficulty_score', 0)}/10")
        m3.markdown(f"**Roots:** `{display_roots(ai)}`")
        
        st.info(f"📚 **Narrative:** {ai.get('narrative_arc', 'N/A')}")
        st.write(f"**Keywords:** {', '.join(ai.get('keywords', []))}")
        
        # --- RESTORED: DEVELOPER VIEW ---
        st.divider()
        with st.expander("🛠️ Developer View (Raw JSON)"):
            st.json(verse_obj)


# ==========================================
# MODE 2: SEMANTIC SEARCH
# ==========================================
elif app_mode == "🔍 Semantic Search":
    st.title("🔍 Semantic Search Engine")
    
    if not user_api_key:
        st.warning("👈 Please enter a Google API Key in the sidebar to activate Search.")
    elif not vector_index:
        st.error("Vector Database not found.")
    else:
        query = st.text_input("Search the Torah:", placeholder="e.g., 'Siblings fighting'")
        
        if query:
            genai.configure(api_key=user_api_key)
            with st.spinner("Searching..."):
                try:
                    res = genai.embed_content(model="models/text-embedding-004", content=query, task_type="retrieval_query")
                    q_vec = np.array([res['embedding']]).astype('float32')
                    D, I = vector_index.search(q_vec, 10)
                    
                    st.subheader(f"Results for: '{query}'")
                    for rank, idx in enumerate(I[0]):
                        verse = vector_meta.get(idx)
                        if not verse: verse = vector_meta.get(str(idx))
                        
                        if verse:
                            score = D[0][rank]
                            ai = verse.get('ai_data', {})
                            with st.expander(f"**{verse['id']}** (Match: {score:.4f})", expanded=(rank < 2)):
                                st.markdown(f"### {verse['raw']}")
                                st.write(f"**{ai.get('translation_rugged')}**")
                                st.caption(f"Context: {ai.get('narrative_arc')}")
                                # Developer View inside Search Results too!
                                st.json(verse) 
                except Exception as e:
                    st.error(f"Search Error: {e}")

# ==========================================
# MODE 3: AI CHAT
# ==========================================
elif app_mode == "💬 AI Chat":
    st.title("💬 Chat with Torah")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "I am connected to the Torah Database. Ask me anything."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if not user_api_key:
        st.warning("👈 API Key required.")
    elif not vector_index:
        st.error("Vector DB required.")
    else:
        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                genai.configure(api_key=user_api_key)
                
                # RAG Retrieval
                res = genai.embed_content(model="models/text-embedding-004", content=prompt, task_type="retrieval_query")
                q_vec = np.array([res['embedding']]).astype('float32')
                D, I = vector_index.search(q_vec, 5)
                
                context_text, sources = "", []
                for idx in I[0]:
                    v = vector_meta.get(idx)
                    if not v: v = vector_meta.get(str(idx))
                    if v:
                        ai = v.get('ai_data', {})
                        context_text += f"[{v['id']}] {ai.get('translation_rugged')} ({ai.get('narrative_arc')})\n"
                        sources.append(v['id'])
                
                # Chat Response
                chat_model = genai.GenerativeModel('gemini-2.0-flash')
                chat_prompt = f"Answer using ONLY these verses:\n{context_text}\n\nQuestion: {prompt}"
                response = chat_model.generate_content(chat_prompt)
                
                full_resp = response.text + f"\n\n**Sources:** {', '.join(sources)}"
                message_placeholder.markdown(full_resp)
                st.session_state.messages.append({"role": "assistant", "content": full_resp})

# ==========================================
# MODE 4: SUGGEST EDIT
# ==========================================
elif app_mode == "📝 Suggest Edit":
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
                gmail_link = f"https://mail.google.com/mail/?view=cm&fs=1&to={DEVELOPER_EMAIL}&su={safe_subject}&body={safe_body}"
                
                st.markdown(f'<a href="{gmail_link}" target="_blank"><button style="background-color:#DB4437; color:white; padding:10px 15px; border:none; border-radius:5px; cursor:pointer;">📮 Open in Gmail</button></a>', unsafe_allow_html=True)
        
        # Developer View in Edit Mode too
        with st.expander("🛠️ Developer View"):
            st.json(verse_obj)

# ==========================================
# MODE 5: ADMIN EDIT
# ==========================================
elif app_mode == "🔒 Direct Edit (Admin)":
    st.warning("⚠️ **ADMIN MODE:** Changes save directly to Google Drive.")
    
    c_book, c_chap, c_verse = st.columns([2, 1, 1])
    with c_book: sel_book = st.selectbox("Book", list(BOOKS_CONFIG.keys()), key="admin_book")
    curr_config = BOOKS_CONFIG[sel_book]
    # FORCE RELOAD FOR ADMIN
    full_path = f"{BASE_PATH}/{curr_config['path']}"
    with open(full_path, 'r', encoding='utf-8') as f:
        book_data = json.load(f)

    with c_chap: sel_chap = st.selectbox("Chapter", range(1, curr_config["chapters"] + 1), key="admin_chap")
    with c_verse:
        prefix = curr_config['prefix']
        target_id_start = f"{prefix}{sel_chap}:"
        verses_in_chapter = [i for i in book_data if i['id'].startswith(target_id_start)]
        v_count = len(verses_in_chapter) if verses_in_chapter else 1
        sel_verse = st.selectbox("Verse", range(1, v_count + 1), key="admin_verse")

    target_id = f"{prefix}{sel_chap}:{sel_verse}"
    verse_obj = next((item for item in book_data if item['id'] == target_id), None)

    if verse_obj:
        ai = verse_obj.get('ai_data', {})
        st.markdown(f"### {verse_obj['raw']}")
        st.caption(f"ID: {target_id}")

        with st.form("admin_form"):
            new_trans = st.text_area("Translation", value=ai.get('translation_rugged', ''))
            c1, c2 = st.columns(2)
            with c1: new_tone = st.text_input("Tone", value=ai.get('tone', ''))
            with c2: new_narrative = st.text_input("Narrative Arc", value=ai.get('narrative_arc', ''))
            
            current_roots = ai.get('hebrew_roots', [])
            simple_roots = []
            if isinstance(current_roots, list):
                for r in current_roots:
                    if isinstance(r, dict): simple_roots.append(r.get('root', ''))
                    elif isinstance(r, str): simple_roots.append(r)
            new_roots_str = st.text_input("Roots (comma separated)", value=", ".join(simple_roots))
            new_keywords_str = st.text_input("Keywords (comma separated)", value=", ".join(ai.get('keywords', [])))
            
            if st.form_submit_button("💾 SAVE TO DRIVE"):
                updated_ai = ai.copy()
                updated_ai['translation_rugged'] = new_trans
                updated_ai['tone'] = new_tone
                updated_ai['narrative_arc'] = new_narrative
                updated_ai['hebrew_roots'] = [r.strip() for r in new_roots_str.split(',') if r.strip()]
                updated_ai['keywords'] = [k.strip() for k in new_keywords_str.split(',') if k.strip()]
                
                if save_change(sel_book, target_id, updated_ai):
                    st.success("✅ Saved! Reloading...")
                    time.sleep(1)
                    st.rerun()
        
        # Developer View
        with st.expander("🛠️ Developer View"):
            st.json(verse_obj)
