import os
import ast
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# LangChain / RAG Imports
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import download_hugging_face_embeddings

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

SYMPTOMS_DICT = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

DISEASES_LIST = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 
    33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 
    23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 
    28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 
    19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 
    36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 
    18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 
    25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 
    2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}

# ---------------------------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MediBot AI — Smart Medical Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data Loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_svc_model():
    model_path = BASE_DIR / "svc.pkl"
    if not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_recommendation_data():
    data = {}
    try:
        data["description"] = pd.read_csv(DATA_DIR / "description.csv")
        data["precautions"] = pd.read_csv(DATA_DIR / "precautions_df.csv")
        data["medications"] = pd.read_csv(DATA_DIR / "medications.csv")
        data["diets"] = pd.read_csv(DATA_DIR / "diets.csv")
        data["workout"] = pd.read_csv(DATA_DIR / "workout_df.csv")
    except Exception as e:
        st.error(f"Error loading CSV data: {e}")
    return data

@st.cache_resource
def get_rag_chain():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if not PINECONE_API_KEY or not GROQ_API_KEY:
        return None
    
    try:
        embeddings = download_hugging_face_embeddings()
        index_name = "medical-chatbot"
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        chatModel = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)
        
        system_prompt = (
            "You are a professional Medical Assistant. "
            "Use the following retrieved medical context to answer the question. "
            "If you don't know the answer, politely say you don't know. "
            "Provide concise, accurate, and helpful medical information. "
            "Always include a disclaimer that this is not a substitute for professional medical advice.\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
        return create_retrieval_chain(retriever, question_answer_chain)
    except Exception as e:
        st.error(f"RAG Initialization Error: {e}")
        return None

# ---------------------------------------------------------------------------
# Custom CSS - Modern Glassmorphic Design
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Global Styles */
    html, body, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1729 100%);
        color: #e2e8f0;
    }
    
    /* Remove default padding and white space */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Remove top white space */
    .main .block-container {
        padding-top: 0rem;
    }
    
    header {
        background-color: transparent !important;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
        display: none;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { 
        width: 10px; 
        height: 10px;
    }
    ::-webkit-scrollbar-track { 
        background: rgba(15, 23, 42, 0.3); 
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb { 
        background: linear-gradient(180deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 10px; 
        border: 2px solid rgba(15, 23, 42, 0.3);
    }
    ::-webkit-scrollbar-thumb:hover { 
        background: linear-gradient(180deg, #60a5fa 0%, #a78bfa 100%);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(10, 14, 39, 0.98) 100%);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(59, 130, 246, 0.1);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 1.8rem;
        letter-spacing: -0.5px;
    }

    /* Hero Section */
    .hero-container {
        padding: 5rem 2rem;
        text-align: center;
        background: radial-gradient(ellipse at top, rgba(59, 130, 246, 0.15) 0%, transparent 60%),
                    radial-gradient(ellipse at bottom, rgba(139, 92, 246, 0.1) 0%, transparent 60%);
        border-radius: 32px;
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(59, 130, 246, 0.1);
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.03) 1px, transparent 1px);
        background-size: 50px 50px;
        animation: gridMove 20s linear infinite;
        pointer-events: none;
    }
    
    @keyframes gridMove {
        0% { transform: translate(0, 0); }
        100% { transform: translate(50px, 50px); }
    }
    
    .hero-title {
        font-size: 4.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        line-height: 1.1;
        letter-spacing: -2px;
        position: relative;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #94a3b8;
        max-width: 900px;
        margin: 0 auto;
        line-height: 1.8;
        font-weight: 400;
        position: relative;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.4);
        backdrop-filter: blur(16px) saturate(180%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.5), transparent);
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(59, 130, 246, 0.4);
        box-shadow: 0 20px 60px rgba(59, 130, 246, 0.15),
                    0 0 0 1px rgba(59, 130, 246, 0.1) inset;
        transform: translateY(-8px);
    }
    
    .glass-card:hover::before {
        opacity: 1;
    }
    
    .glass-card h2 {
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 1rem;
        letter-spacing: -0.5px;
    }
    
    .glass-card p {
        color: #cbd5e1;
        line-height: 1.7;
        font-size: 1.05rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: #ffffff;
        border: none;
        border-radius: 16px;
        padding: 0.875rem 2.5rem;
        font-weight: 600;
        font-size: 1.05rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.5);
        transform: translateY(-2px) scale(1.02);
    }
    
    .stButton > button:active {
        transform: translateY(0) scale(0.98);
    }

    /* Multiselect Styling */
    .stMultiSelect [data-baseweb="select"] {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 16px;
    }
    
    .stMultiSelect [data-baseweb="tag"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        color: #60a5fa;
    }

    /* Status Pills */
    .status-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 100px;
        font-size: 0.875rem;
        font-weight: 600;
        margin-right: 10px;
        margin-bottom: 8px;
    }
    
    .status-ready {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .status-missing {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Prediction Result Card */
    .prediction-result {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 2px solid;
        border-image: linear-gradient(135deg, #3b82f6, #8b5cf6) 1;
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-result::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.05); }
    }
    
    .prediction-label {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #94a3b8;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .prediction-disease {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        position: relative;
        letter-spacing: -1px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.3);
        border-radius: 16px;
        padding: 6px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: #94a3b8;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59, 130, 246, 0.1);
        color: #60a5fa;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
        color: #60a5fa !important;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    /* Chat Styling */
    .stChatFloatingInputContainer { 
        background-color: transparent !important; 
        border-top: 1px solid rgba(59, 130, 246, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .stChatMessage {
        background: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(59, 130, 246, 0.1) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
    }
    
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Fix chat message text color - make it readable */
    .stChatMessage p,
    .stChatMessage div,
    .stChatMessage span,
    .stChatMessage li {
        color: #e2e8f0 !important;
    }
    
    .stChatMessage strong {
        color: #60a5fa !important;
    }
    
    .stChatMessage code {
        background: rgba(59, 130, 246, 0.1);
        color: #60a5fa;
        padding: 2px 6px;
        border-radius: 4px;
    }

    /* Info/Warning/Error Boxes */
    .stAlert {
        border-radius: 16px;
        border: 1px solid;
        backdrop-filter: blur(10px);
    }
    
    div[data-baseweb="notification"] {
        border-radius: 16px;
    }

    /* Markdown in tabs */
    .stTabs [data-baseweb="tab-panel"] li {
        color: #cbd5e1;
        margin-bottom: 0.75rem;
        line-height: 1.7;
    }
    
    .stTabs [data-baseweb="tab-panel"] li::marker {
        color: #60a5fa;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.3), transparent);
        margin: 2rem 0;
    }

    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 3rem 2rem;
        font-size: 0.9rem;
        border-top: 1px solid rgba(59, 130, 246, 0.1);
        margin-top: 4rem;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: none;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
        border: 1px solid rgba(59, 130, 246, 0.4);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
    }
    
    /* Feature cards on home */
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Navigation Logic
# ---------------------------------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "predicted_disease" not in st.session_state:
    st.session_state.predicted_disease = None

def set_page(page_name):
    st.session_state.page = page_name

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("<h1>🩺 MediBot AI</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    nav_buttons = [
        ("🏠 Home", "Home"),
        ("🔬 Disease Prediction", "Prediction"),
        ("💬 Medical Chatbot", "Chatbot"),
        ("ℹ️ About", "About")
    ]
    
    for label, page in nav_buttons:
        if st.button(label, use_container_width=True, key=f"nav_{page}"):
            set_page(page)
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    
    svc_model_loaded = load_svc_model() is not None
    rag_ready = get_rag_chain() is not None
    
    model_status = "🟢 Ready" if svc_model_loaded else "🔴 Missing"
    rag_status = "🟢 Ready" if rag_ready else "🔴 Not Configured"
    
    st.markdown(f'<div class="status-pill status-{"ready" if svc_model_loaded else "missing"}">ML Model: {model_status}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="status-pill status-{"ready" if rag_ready else "missing"}">RAG Engine: {rag_status}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("🔄 Reload Model (Clear Cache)", use_container_width=True):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Cache cleared! App will reload shortly...")
        st.rerun()

# ---------------------------------------------------------------------------
# Page: Home
# ---------------------------------------------------------------------------

if st.session_state.page == "Home":
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">🩺 MediBot AI</div>
            <div class="hero-subtitle">
                Your intelligent health companion powered by machine learning and advanced AI. 
                Analyze symptoms, predict conditions, and get instant medical insights from our RAG-powered assistant.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
            <div class="glass-card">
                <div class="feature-icon">🔬</div>
                <h2 style='color:#60a5fa;'>Disease Prediction</h2>
                <p>Input your symptoms and receive instant analysis powered by our trained SVM model. 
                Get comprehensive recommendations including medications, dietary suggestions, and precautionary measures.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Start Diagnosis", key="home_predict"):
            set_page("Prediction")
            st.rerun()
            
    with col2:
        st.markdown("""
            <div class="glass-card">
                <div class="feature-icon">💬</div>
                <h2 style='color:#a78bfa;'>Medical Assistant</h2>
                <p>Ask questions about any medical condition and get evidence-based answers instantly. 
                Our RAG-powered assistant queries an extensive medical knowledge base to provide accurate information.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("💡 Consult Assistant", key="home_chat"):
            set_page("Chatbot")
            st.rerun()
    
    # Additional Features
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; margin: 2rem 0;'>Why Choose MediBot AI?</h2>", unsafe_allow_html=True)
    
    feat1, feat2, feat3 = st.columns(3, gap="large")
    
    with feat1:
        st.markdown("""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">⚡</div>
                <h3 style="color: #60a5fa; margin-bottom: 1rem;">Lightning Fast</h3>
                <p>Get instant predictions and answers in seconds, powered by optimized ML algorithms and vector databases.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with feat2:
        st.markdown("""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">🎯</div>
                <h3 style="color: #a78bfa; margin-bottom: 1rem;">High Accuracy</h3>
                <p>Our SVM model is trained on extensive medical datasets ensuring reliable symptom-based predictions.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with feat3:
        st.markdown("""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">🔒</div>
                <h3 style="color: #ec4899; margin-bottom: 1rem;">Privacy First</h3>
                <p>Your health data stays private. No storage of personal information, all processing is done locally.</p>
            </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page: Prediction
# ---------------------------------------------------------------------------

elif st.session_state.page == "Prediction":
    st.markdown("<h1 style='text-align:center; background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900;'>🔬 Disease Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#94a3b8; font-size: 1.2rem; margin-bottom: 2rem;'>Select your symptoms to receive an AI-powered health analysis</p>", unsafe_allow_html=True)
    
    svc_model = load_svc_model()
    rec_data = load_recommendation_data()
    
    if not svc_model:
        st.error("⚠️ Model file (svc.pkl) not found. Please ensure it exists in the root directory.")
    else:
        symptom_list = sorted(list(SYMPTOMS_DICT.keys()))
        display_symptoms = [s.replace("_", " ").title() for s in symptom_list]
        symptom_map = {s.replace("_", " ").title(): s for s in symptom_list}
        
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📝 Symptom Selection")
            selected = st.multiselect(
                "Choose all symptoms you're experiencing:",
                options=display_symptoms,
                help="Select multiple symptoms for more accurate prediction"
            )
            
            col_analyze, col_clear = st.columns([3, 1])
            
            with col_analyze:
                analyze_btn = st.button("🔍 Analyze Symptoms", use_container_width=True)
            
            with col_clear:
                if st.button("🔄 Clear", use_container_width=True):
                    st.session_state.predicted_disease = None
                    st.rerun()
            
            if analyze_btn:
                if not selected:
                    st.warning("⚠️ Please select at least one symptom to begin analysis.")
                else:
                    with st.spinner("🔬 Analyzing your symptoms..."):
                        raw_selected = [symptom_map[s] for s in selected]
                        input_vector = np.zeros(len(SYMPTOMS_DICT))
                        for s in raw_selected:
                            input_vector[SYMPTOMS_DICT[s]] = 1
                        
                        pred = svc_model.predict([input_vector])[0]
                        disease = DISEASES_LIST.get(pred, pred) if isinstance(pred, (int, np.integer)) else pred
                        st.session_state.predicted_disease = disease
                        
                        st.markdown(f"""
                            <div class="prediction-result">
                                <div class="prediction-label">Predicted Condition</div>
                                <div class="prediction-disease">{disease}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Fetch recommendations
                        desc = rec_data["description"][rec_data["description"]["Disease"] == disease]["Description"].values
                        pre = rec_data["precautions"][rec_data["precautions"]["Disease"] == disease].iloc[:, 1:].values
                        med = rec_data["medications"][rec_data["medications"]["Disease"] == disease]["Medication"].values
                        diet = rec_data["diets"][rec_data["diets"]["Disease"] == disease]["Diet"].values
                        work = rec_data["workout"][rec_data["workout"]["disease"] == disease]["workout"].values
                        
                        st.markdown("### 📚 Detailed Recommendations")
                        tabs = st.tabs(["📋 Overview", "🛡️ Precautions", "💊 Medications", "🥗 Diet Plan", "🏋️ Exercise"])
                        
                        with tabs[0]:
                            if len(desc) > 0:
                                st.info(desc[0])
                            else:
                                st.info("No description available for this condition.")
                        
                        with tabs[1]:
                            if len(pre) > 0:
                                st.markdown("**Recommended precautions:**")
                                for p in pre[0]:
                                    if pd.notna(p):
                                        st.markdown(f"• {p}")
                            else:
                                st.write("No precautions data available.")
                            
                        with tabs[2]:
                            if len(med) > 0:
                                st.markdown("**Suggested medications:**")
                                try:
                                    meds = ast.literal_eval(med[0]) if "[" in str(med[0]) else [med[0]]
                                    for m in meds:
                                        st.markdown(f"• {m}")
                                except:
                                    st.markdown(f"• {med[0]}")
                            else:
                                st.write("No medication information available.")
                            
                        with tabs[3]:
                            if len(diet) > 0:
                                st.markdown("**Dietary recommendations:**")
                                try:
                                    diets = ast.literal_eval(diet[0]) if "[" in str(diet[0]) else [diet[0]]
                                    for d in diets:
                                        st.markdown(f"• {d}")
                                except:
                                    st.markdown(f"• {diet[0]}")
                            else:
                                st.write("No dietary information available.")
                            
                        with tabs[4]:
                            if len(work) > 0:
                                st.markdown("**Recommended exercises:**")
                                for w in work:
                                    st.markdown(f"• {w}")
                            else:
                                st.write("No exercise recommendations available.")
                        
                        st.markdown("---")
                        st.markdown("""
                            <div class="glass-card" style="background: rgba(59, 130, 246, 0.05); border-color: rgba(59, 130, 246, 0.2);">
                                <h4 style="color: #60a5fa; margin-bottom: 1rem;">💬 Want to learn more?</h4>
                                <p>Discuss this condition with our AI Medical Assistant for detailed information and personalized guidance.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("💬 Chat with AI Assistant", use_container_width=True):
                            set_page("Chatbot")
                            st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page: Chatbot
# ---------------------------------------------------------------------------

elif st.session_state.page == "Chatbot":
    st.markdown("<h1 style='text-align:center; background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900;'>💬 Medical AI Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#94a3b8; font-size: 1.2rem; margin-bottom: 2rem;'>Ask anything about medical conditions, treatments, and health management</p>", unsafe_allow_html=True)
    
    rag_chain = get_rag_chain()
    
    if not rag_chain:
        st.markdown("""
            <div class="glass-card" style="background: rgba(239, 68, 68, 0.1); border-color: rgba(239, 68, 68, 0.3);">
                <h3 style="color: #f87171;">⚠️ Chatbot Unavailable</h3>
                <p>The Medical Assistant requires API keys to function. Please configure your <code>GROQ_API_KEY</code> and <code>PINECONE_API_KEY</code> in the <code>.env</code> file.</p>
                <p style="margin-top: 1rem;">You can still use the <strong>Disease Prediction</strong> tool without these configurations.</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔬 Go to Disease Prediction", use_container_width=True):
            set_page("Prediction")
            st.rerun()
    else:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
            # If coming from prediction, add context
            if st.session_state.predicted_disease:
                initial_msg = f"I've been predicted to have **{st.session_state.predicted_disease}**. Can you provide more information about this condition, its causes, and long-term management strategies?"
                st.session_state.messages.append({"role": "user", "content": initial_msg})
                
                with st.spinner("🤔 Consulting medical knowledge base..."):
                    response = rag_chain.invoke({"input": initial_msg})
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("💭 Ask me anything about health and medical conditions..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching medical database..."):
                    response = rag_chain.invoke({"input": prompt})
                    full_response = response["answer"]
                    st.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------

elif st.session_state.page == "About":
    st.markdown("<h1 style='text-align:center; background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900;'>ℹ️ About MediBot AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#94a3b8; font-size: 1.2rem; margin-bottom: 3rem;'>Intelligent medical assistance powered by advanced AI</p>", unsafe_allow_html=True)
    
    # Project Overview Section
    st.markdown("""
    <div class="glass-card">
    <h3 style="color: #60a5fa; margin-bottom: 1.5rem;">🚀 Project Overview</h3>

    <p style="font-size: 1.1rem; line-height: 1.9; color: #cbd5e1;">
        MediBot AI is a cutting-edge medical assistance platform that combines Machine Learning and Large Language Models 
        to democratize healthcare information. Our mission is to make preliminary health insights accessible to everyone, 
        bridging the gap between complex medical data and user-friendly technology.
    </p>

    <p style="margin-top: 1.5rem; font-size: 1.05rem; line-height: 1.9; color: #cbd5e1;">
        Built with state-of-the-art AI technologies, MediBot AI provides:
    </p>

    <ul style="color: #cbd5e1; line-height: 2.2; font-size: 1.05rem; margin-top: 1rem;">
        <li>🎯 <strong style="color: #60a5fa;">Accurate disease prediction</strong> based on symptoms</li>
        <li>💊 <strong style="color: #60a5fa;">Comprehensive treatment recommendations</strong> including medications, diet, and exercise</li>
        <li>💬 <strong style="color: #60a5fa;">Intelligent conversational AI assistant</strong> powered by RAG technology</li>
        <li>📚 <strong style="color: #60a5fa;">Evidence-based medical information</strong> from trusted sources</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    
    # Features Section
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; margin: 2rem 0;'>✨ Key Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown("""
            <div class="glass-card" style="text-align: center; height: 100%;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔬</div>
                <h3 style="color: #60a5fa; margin-bottom: 1rem;">SVM Classification</h3>
                <p style="color: #cbd5e1; line-height: 1.8;">Advanced machine learning model trained on extensive medical datasets for accurate symptom-based disease prediction</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="glass-card" style="text-align: center; height: 100%;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">💬</div>
                <h3 style="color: #a78bfa; margin-bottom: 1rem;">RAG Pipeline</h3>
                <p style="color: #cbd5e1; line-height: 1.8;">Retrieval-Augmented Generation technology that provides accurate, context-aware answers from medical knowledge base</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="glass-card" style="text-align: center; height: 100%;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
                <h3 style="color: #ec4899; margin-bottom: 1rem;">Real-time Processing</h3>
                <p style="color: #cbd5e1; line-height: 1.8;">Instant predictions and responses powered by optimized algorithms and high-performance computing</p>
            </div>
        """, unsafe_allow_html=True)
    
    # How It Works Section
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; margin: 2rem 0;'>🔄 How It Works</h2>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2, gap="large")
    
    with col_a:
        st.markdown("""
            <div class="glass-card">
                <h3 style="color: #60a5fa; margin-bottom: 1.5rem;">🔬 Disease Prediction System</h3>
                <ol style="color: #cbd5e1; line-height: 2.2; font-size: 1.05rem;">
                    <li><strong style="color: #60a5fa;">Symptom Input:</strong> Select your symptoms from a comprehensive list of 132 medical indicators</li>
                    <li><strong style="color: #60a5fa;">ML Analysis:</strong> Our SVM model analyzes the symptom pattern against trained medical data</li>
                    <li><strong style="color: #60a5fa;">Prediction:</strong> Get instant disease prediction with confidence scores</li>
                    <li><strong style="color: #60a5fa;">Recommendations:</strong> Receive detailed information on medications, diet, precautions, and exercises</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        st.markdown("""
            <div class="glass-card">
                <h3 style="color: #a78bfa; margin-bottom: 1.5rem;">💬 Medical Chatbot</h3>
                <ol style="color: #cbd5e1; line-height: 2.2; font-size: 1.05rem;">
                    <li><strong style="color: #a78bfa;">Ask Question:</strong> Type any medical question or concern</li>
                    <li><strong style="color: #a78bfa;">Vector Search:</strong> Your query is embedded and searched across our medical knowledge base</li>
                    <li><strong style="color: #a78bfa;">Context Retrieval:</strong> Most relevant medical documents are retrieved</li>
                    <li><strong style="color: #a78bfa;">AI Response:</strong> Groq-powered LLM generates accurate, contextual answers with citations</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
    
    # Important Disclaimer
    st.markdown("---")
    st.markdown("""
        <div class="glass-card" style="background: rgba(251, 191, 36, 0.05); border: 2px solid rgba(251, 191, 36, 0.3); margin-top: 2rem;">
            <div style="text-align: center; margin-bottom: 1rem;">
                <span style="font-size: 3rem;">⚠️</span>
            </div>
            <h2 style="color: #fbbf24; margin-bottom: 1.5rem; text-align: center;">Important Disclaimer</h2>
            <p style="color: #cbd5e1; font-size: 1.1rem; line-height: 2; text-align: center; max-width: 900px; margin: 0 auto;">
                This application is designed for <strong style="color: #fbbf24;">educational and informational purposes only</strong>. 
                The predictions and medical information provided are generated by AI models and should 
                <strong style="color: #fbbf24;">NOT</strong> be considered as professional medical advice, diagnosis, or treatment.
            </p>
            <p style="color: #cbd5e1; font-size: 1.1rem; line-height: 2; text-align: center; max-width: 900px; margin: 1.5rem auto 0;">
                ⚕️ Always consult with a qualified healthcare provider for any health concerns or before making 
                any decisions related to your health or treatment.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Developer Info
    st.markdown("---")
    st.markdown("""
        <div class="glass-card" style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%); text-align: center;">
            <h3 style="color: #60a5fa; margin-bottom: 1rem;">👨‍💻 Developer</h3>
            <p style="color: #cbd5e1; font-size: 1.1rem; line-height: 1.8;">
                Created with ❤️ by <strong style="color: #60a5fa; font-size: 1.2rem;">Aman Kar</strong>
            </p>
            <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 1rem;">
                Powered by Machine Learning, NLP, and Retrieval-Augmented Generation
            </p>
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
    <div class="footer">
        <p style="margin-bottom: 0.5rem; font-size: 1rem;">Made with ❤️ by <strong>Aman Kar</strong></p>
        <p style="color: #475569; font-size: 0.85rem;">© 2026 MediBot AI | Powered by Machine Learning & AI</p>
    </div>
""", unsafe_allow_html=True)