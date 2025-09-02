import streamlit as st
import json
import os
import string
import asyncio
import numpy as np
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain.embeddings.base import Embeddings
from together import Together
from dotenv import load_dotenv

# --------------------- PAGE CONFIG MUST BE FIRST --------------------- #
st.set_page_config(
    page_title="E-Commerce Support",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------- LOAD ENVIRONMENT VARIABLES --------------------- #
load_dotenv()

# --------------------- WINDOWS COMPATIBILITY FIX --------------------- #
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --------------------- CONFIGURATION --------------------- #
SEMANTIC_THRESHOLD = 0.65
HYBRID_WEIGHTS = (0.7, 0.3)
SEMANTIC_TOP_K = 100
BM25_TOP_K = 100
CANDIDATE_SET_SIZE = 200
INDEX_DIR = "faiss_index"
USER_DATA_FILE = "user_sessions.json"

# --------------------- PRELOADED RESOURCES --------------------- #
@st.cache_resource
def load_embeddings_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()

@st.cache_resource
def get_llm_client():
    return Together(api_key=os.getenv("TOGETHER_API_KEY"))

@st.cache_resource
def load_vector_store(_embeddings):
    # Create directory if it doesn't exist
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
    
    # Check if vector store exists
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        return FAISS.load_local(
            INDEX_DIR,
            _embeddings,
            allow_dangerous_deserialization=True
        )
    
    # Create new vector store if not exists
    faq_data = load_faq_data()
    documents = []
    for i, item in enumerate(faq_data):
        doc = Document(
            page_content=f"{item['question']} {' '.join(item.get('keywords', []))}",
            metadata={
                "answer": item["answer"],
                "original_question": item["question"],
                "keywords": item.get("keywords", []),
                "index": i  # Add index for reference
            }
        )
        documents.append(doc)
    
    vector_store = FAISS.from_documents(documents, _embeddings)
    vector_store.save_local(INDEX_DIR)
    return vector_store

@st.cache_data
def load_faq_data():
    with open("newdata.json") as f:
        return json.load(f)

@st.cache_resource
def create_bm25_index(faq_data):
    tokenized_questions = []
    for item in faq_data:
        text = f"{item['question']} {' '.join(item.get('keywords', []))}"
        processed = text.lower().translate(str.maketrans('', '', string.punctuation))
        tokenized_questions.append(processed.split())
    return BM25Okapi(tokenized_questions), tokenized_questions

# --------------------- SESSION MANAGEMENT --------------------- #
def save_user_session():
    """Save current session to JSON file"""
    if "user_name" in st.session_state:
        try:
            # Load existing sessions
            if os.path.exists(USER_DATA_FILE):
                with open(USER_DATA_FILE, "r") as f:
                    sessions = json.load(f)
            else:
                sessions = {}
            
            # Create user session data
            user_id = st.session_state.user_name.lower().replace(" ", "_")
            sessions[user_id] = {
                "user_name": st.session_state.user_name,
                "last_active": datetime.now().isoformat(),
                "chat_history": st.session_state.chat_history
            }
            
            # Save to file
            with open(USER_DATA_FILE, "w") as f:
                json.dump(sessions, f, indent=2)
                
        except Exception as e:
            st.toast(f"⚠️ Failed to save session: {str(e)}", icon="⚠️")

def load_user_session(user_id):
    """Load user session from JSON file"""
    try:
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, "r") as f:
                sessions = json.load(f)
                return sessions.get(user_id)
        return None
    except:
        return None

# --------------------- LLM RESPONSE GENERATION --------------------- #
def generate_llm_response(user_query: str, static_answer: str, client: Together, user_name: str) -> str:
    try:
        prompt = f"""
        You are a friendly and helpful customer support assistant for an e-commerce company. 
        A customer named {user_name} asked: "{user_query}"
        
        Here is the official answer from our knowledge base:
        {static_answer}
        
        Create a natural, conversational, and concise reply (2-4 sentences) that:
        - Addresses the customer by name
        - Keeps all information accurate
        - Sounds engaging and helpful
        - Offers additional assistance if needed
        - Avoids formal email closings
        """
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "You are a helpful and friendly customer support agent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=350,
        )
    #Qwen/Qwen2.5-7B-Instruct-Turbo 
    #meta-llama/Llama-3.3-70B-Instruct-Turbo-Free
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Hi {user_name}, {static_answer}"  # Fallback with personalization

# --------------------- HYBRID SEARCH --------------------- #
def hybrid_search(query, vector_store, bm25_index, tokenized_corpus):
    processed_query = query.lower().translate(
        str.maketrans('', '', string.punctuation)).split()
    
    # 1. Semantic Search
    semantic_results = vector_store.similarity_search_with_score(query, k=SEMANTIC_TOP_K)
    semantic_candidates = set()
    semantic_scores = {}
    
    for doc, score in semantic_results:
        idx = doc.metadata["index"]
        semantic_candidates.add(idx)
        semantic_scores[idx] = 1 - (score / 2)
    
    # 2. BM25 Search
    bm25_scores = bm25_index.get_scores(processed_query)
    top_bm25_indices = np.argsort(bm25_scores)[-BM25_TOP_K:]
    bm25_candidates = set(top_bm25_indices.tolist())
    
    # 3. Combine candidate sets
    all_candidates = semantic_candidates | bm25_candidates
    if len(all_candidates) > CANDIDATE_SET_SIZE:
        all_candidates = set(list(semantic_candidates)[:CANDIDATE_SET_SIZE])
    
    # 4. Hybrid Scoring
    combined_scores = []
    bm25_subset = [bm25_scores[i] for i in all_candidates]
    min_bm25 = min(bm25_subset) if bm25_subset else 0
    max_bm25 = max(bm25_subset) if bm25_subset else 1
    
    for idx in all_candidates:
        sem_score = semantic_scores.get(idx, 0)
        bm25_norm = (bm25_scores[idx] - min_bm25) / (max_bm25 - min_bm25 + 1e-9)
        combined = (HYBRID_WEIGHTS[0] * sem_score) + (HYBRID_WEIGHTS[1] * bm25_norm)
        combined_scores.append((combined, idx))
    
    return sorted(combined_scores, key=lambda x: x[0], reverse=True)[:3]

# --------------------- MAIN INTERFACE --------------------- #
# Preload resources (will cache after first run)
sentence_model = load_embeddings_model()
embeddings = SentenceTransformerEmbeddings(sentence_model)
vector_store = load_vector_store(embeddings)  # Updated to create if needed
faq_data = load_faq_data()
bm25_index, tokenized_corpus = create_bm25_index(faq_data)
llm_client = get_llm_client()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "processing" not in st.session_state:
    st.session_state.processing = False

# App title
st.title("🛍️ E-Commerce Support Assistant")

# Sidebar for session management
with st.sidebar:
    st.title("Your Session")
    st.divider()
    
    if st.session_state.user_name:
        st.success(f"**Welcome back, {st.session_state.user_name}!**")
        st.caption(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        if st.button("🧹 Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.toast("Conversation cleared", icon="🧹")
            
        if st.button("💾 Save Session", use_container_width=True):
            save_user_session()
            st.toast("Session saved!", icon="💾")
            
        if st.button("🚪 Start New Session", use_container_width=True):
            st.session_state.user_name = None
            st.session_state.chat_history = []
            st.toast("New session started", icon="🚪")
            
        st.divider()
        st.subheader("Previous Chats")
        for i, chat in enumerate(st.session_state.chat_history):
            if i > 0 and st.session_state.chat_history[i-1]["role"] == "assistant":
                st.divider()
            if chat["role"] == "user":
                st.markdown(f"**You**: {chat['content']}")
            else:
                st.markdown(f"**Assistant**: {chat['content']}")
    else:
        st.subheader("Get Started")
        with st.form("user_form"):
            name = st.text_input("Enter your name:")
            submitted = st.form_submit_button("Start Chatting")
            
            if submitted and name:
                st.session_state.user_name = name
                st.session_state.chat_history = []
                st.toast(f"Welcome {name}! How can I help?", icon="👋")

# Main chat container
chat_container = st.container()

# Display chat history in main container
with chat_container:
    if st.session_state.user_name:
        for chat in st.session_state.chat_history:
            with st.chat_message(chat["role"]):
                st.write(chat["content"])
    else:
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px; border-radius: 15px; background: linear-gradient(135deg, #f5f7fa, #e4edf9);">
            <h2 style="color: #2563eb;">Welcome to Customer Support</h2>
            <p style="font-size: 18px;">Get instant answers to your questions with our AI assistant</p>
            <div style="margin: 30px auto; max-width: 300px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" fill="#2563eb" viewBox="0 0 24 24" style="margin: 0 auto;">
                    <path d="M20 2H4c-1.1 0-1.99.9-1.99 2L2 22l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-3 12H7v-2h10v2zm0-3H7V9h10v2zm0-3H7V6h10v2z"/>
                </svg>
            </div>
            <p style="font-size: 16px;">Please enter your name in the sidebar to begin</p>
        </div>
        """, unsafe_allow_html=True)

# Chat input and processing
if st.session_state.user_name and not st.session_state.processing:
    if user_query := st.chat_input(f"Hi {st.session_state.user_name}, how can I help?"):
        # Set processing flag
        st.session_state.processing = True
        
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })
        
        # Rerender immediately - FIXED
        st.rerun()

# Response generation
if st.session_state.user_name and st.session_state.processing:
    # Get the last user message
    user_query = st.session_state.chat_history[-1]["content"]
    
    # Perform hybrid search
    sorted_scores = hybrid_search(
        user_query, 
        vector_store, 
        bm25_index, 
        tokenized_corpus
    )

    # Get best match
    best_score, best_idx = sorted_scores[0]
    best_item = faq_data[best_idx]

    # Generate response
    if best_score >= SEMANTIC_THRESHOLD:
        with st.spinner("💭 Thinking..."):
            response = generate_llm_response(
                user_query,
                best_item["answer"],
                llm_client,
                st.session_state.user_name
            )
    else:
        response = f"I'm not sure I understand, {st.session_state.user_name}. Could you please rephrase your question?"

    # Add assistant response to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response
    })
    
    # Reset processing flag
    st.session_state.processing = False
    
    # Auto-save session
    save_user_session()
    st.rerun()  # FIXED

# Footer
st.divider()
st.caption("© 2024 E-Commerce Support | AI-Powered Customer Service")

# Add some custom styling
st.markdown("""
<style>
    /* Main container */
    .stApp {
        background-color: #f9fafb;
    }
    
    /* Chat message styling */
    [data-testid="stChatMessage"] {
        padding: 15px 20px;
        border-radius: 18px;
        margin-bottom: 10px;
        max-width: 85%;
    }
    
    /* User message */
    [data-testid="stChatMessage"][aria-label="You"] {
        background-color: #2563eb;
        color: white;
        margin-left: auto;
    }
    
    /* Assistant message */
    [data-testid="stChatMessage"][aria-label="Assistant"] {
        background-color: #e5e7eb;
        color: #1f2937;
        margin-right: auto;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
    }
    
    /* Sidebar header */
    [data-testid="stSidebar"] h1 {
        color: white !important;
    }
    
    /* Button styling */
    .stButton button {
        background: white !important;
        color: #2563eb !important;
        border: none;
        border-radius: 12px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background: #f0f9ff !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    
    /* Input field styling */
    .stTextInput input {
        border-radius: 12px;
        padding: 12px 15px;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #2563eb transparent transparent transparent !important;
    }
</style>
""", unsafe_allow_html=True)