# ============================================================
# Multi-Agent Generative AI System for Personalized Marketing
# Campaign Optimization
# ============================================================
# Run with: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
import random
import json
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MarketMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Background */
.stApp {
    background: #0a0e1a;
    color: #e0e6f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1120 0%, #111827 100%);
    border-right: 1px solid #1e2d47;
}
[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 0.9rem;
    padding: 4px 0;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #38bdf8 !important;
}

/* Cards */
.agent-card {
    background: linear-gradient(135deg, #111827 0%, #1a2332 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 24px rgba(56, 189, 248, 0.05);
}

.metric-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
}

.metric-label {
    color: #64748b;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Hero */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}

.hero-sub {
    color: #64748b;
    font-size: 1.1rem;
    margin-top: 8px;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.5px;
}
.badge-blue  { background: rgba(56,189,248,0.15); color: #38bdf8; border: 1px solid rgba(56,189,248,0.3); }
.badge-purple{ background: rgba(129,140,248,0.15);color: #818cf8; border: 1px solid rgba(129,140,248,0.3);}
.badge-green { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.badge-amber { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }

/* Section header */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    color: #e2e8f0;
    margin-bottom: 4px;
}
.section-desc {
    color: #64748b;
    font-size: 0.9rem;
    margin-bottom: 20px;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #1e3a5f;
    margin: 24px 0;
}

/* Output box */
.output-box {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-left: 4px solid #38bdf8;
    border-radius: 8px;
    padding: 16px 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #94a3b8;
    white-space: pre-wrap;
    line-height: 1.7;
}

/* Strategy box */
.strategy-box {
    background: linear-gradient(135deg, #0f172a, #1a1040);
    border: 1px solid #4c1d95;
    border-left: 4px solid #818cf8;
    border-radius: 8px;
    padding: 20px;
    color: #c4b5fd;
    font-size: 0.9rem;
    line-height: 1.8;
}

/* Streamlit overrides */
div[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px;
}
div[data-testid="stMetric"] label {
    color: #64748b !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-family: 'Space Mono', monospace;
}

.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(99,102,241,0.4);
}

.stSelectbox > div > div, .stNumberInput > div > div > input, .stTextInput > div > div > input {
    background: #111827 !important;
    border: 1px solid #1e3a5f !important;
    color: #e0e6f0 !important;
    border-radius: 8px !important;
}

.stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# GEMINI API HELPER
# ─────────────────────────────────────────────
def call_gemini(prompt: str, api_key: str) -> str:
    """Call Google Gemini API and return the text response."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except ImportError:
        return None  # Library not installed
    except Exception as e:
        return None  # API error – caller will use fallback


# ─────────────────────────────────────────────
# AGENT 1 – DATA ANALYSIS
# ─────────────────────────────────────────────
def generate_synthetic_data(n: int = 200) -> pd.DataFrame:
    """Generate a realistic synthetic marketing dataset."""
    random.seed(42)
    np.random.seed(42)

    ages        = np.random.randint(18, 65, n)
    genders     = np.random.choice(["Male", "Female", "Non-binary"], n, p=[0.48, 0.48, 0.04])
    locations   = np.random.choice(["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune"], n)
    interests_pool = ["Sports", "Fashion", "Technology", "Travel", "Food", "Music", "Gaming", "Fitness", "Books", "Art"]
    interests   = [random.choice(interests_pool) for _ in range(n)]
    purchase_freq = np.random.choice(["Low", "Medium", "High"], n, p=[0.35, 0.40, 0.25])
    platforms   = np.random.choice(["Instagram", "Email", "WhatsApp", "YouTube", "Twitter"], n)

    df = pd.DataFrame({
        "User_ID": [f"U{str(i).zfill(4)}" for i in range(1, n+1)],
        "Age": ages,
        "Gender": genders,
        "Location": locations,
        "Interests": interests,
        "Purchase_Frequency": purchase_freq,
        "Preferred_Platform": platforms,
    })
    return df


def clean_and_analyze(df: pd.DataFrame):
    """Clean dataset and return stats."""
    # Remove duplicates and missing values
    df = df.drop_duplicates().dropna().reset_index(drop=True)

    # Ensure correct types
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(df["Age"].median()).astype(int)

    # Encode Purchase Frequency
    freq_map = {"Low": 1, "Medium": 2, "High": 3}
    df["Purchase_Frequency_Encoded"] = df["Purchase_Frequency"].map(freq_map).fillna(1).astype(int)

    stats = {
        "total_users": len(df),
        "age_mean": round(df["Age"].mean(), 1),
        "age_std": round(df["Age"].std(), 1),
        "gender_dist": df["Gender"].value_counts().to_dict(),
        "top_interests": df["Interests"].value_counts().head(5).to_dict(),
        "platform_dist": df["Preferred_Platform"].value_counts().to_dict(),
        "purchase_dist": df["Purchase_Frequency"].value_counts().to_dict(),
    }
    return df, stats


# ─────────────────────────────────────────────
# AGENT 2 – SEGMENTATION
# ─────────────────────────────────────────────
def segment_customers(df: pd.DataFrame):
    """K-Means clustering into 3 segments."""
    features = df[["Age", "Purchase_Frequency_Encoded"]].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # Assign meaningful labels based on cluster centroids
    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                           columns=["Age_c", "PF_c"])
    
    label_map = {}
    for idx, row in centers.iterrows():
        if row["PF_c"] <= 1.4:
            label_map[idx] = "Students / Low Spenders"
        elif row["PF_c"] <= 2.4:
            label_map[idx] = "Budget Buyers"
        else:
            label_map[idx] = "Premium Users"

    # If duplicate labels, differentiate by age
    if len(set(label_map.values())) < 3:
        sorted_by_age = centers.sort_values("Age_c").index.tolist()
        fallback = {sorted_by_age[0]: "Students / Low Spenders",
                    sorted_by_age[1]: "Budget Buyers",
                    sorted_by_age[2]: "Premium Users"}
        label_map = fallback

    df["Segment"] = df["Cluster"].map(label_map)
    return df, kmeans, scaler


# ─────────────────────────────────────────────
# AGENT 3 – CONTENT GENERATION
# ─────────────────────────────────────────────
FALLBACK_TEMPLATES = {
    "Students / Low Spenders": [
        "🎓 Hey {name}! Score big with our student discount — up to 40% off on {interest} gear. Limited time!",
        "📚 We know student life is tough. That's why we're giving you exclusive {interest} deals — just for you!",
        "💸 Budget smart! Our special student offer on {interest} products is live. Don't miss it!",
    ],
    "Budget Buyers": [
        "💡 Smart shopping starts here! Get the best value on {interest} — quality you can afford.",
        "🛒 Great deals for savvy buyers! Check out our curated {interest} collection under your budget.",
        "🔖 Value-packed offers on {interest} are waiting for you. Shop more, spend less!",
    ],
    "Premium Users": [
        "✨ Exclusively for you — our luxury {interest} collection. Experience the finest, because you deserve it.",
        "👑 Elevate your lifestyle. Discover our premium {interest} picks curated for discerning tastes.",
        "🥂 Your premium membership unlocks exclusive {interest} experiences. Indulge today.",
    ],
}


def generate_content(segment: str, interest: str, platform: str, api_key: str = None) -> str:
    """Generate personalized marketing message via Gemini or fallback templates."""
    if api_key:
        prompt = (
            f"Write a short, engaging personalized marketing message (2-3 sentences) for a customer.\n"
            f"Segment: {segment}\nInterest: {interest}\nPreferred Platform: {platform}\n"
            f"Rules:\n"
            f"- Students / Low Spenders: focus on discounts and student deals\n"
            f"- Budget Buyers: focus on value and affordability\n"
            f"- Premium Users: focus on luxury, exclusivity and quality\n"
            f"Keep it punchy, platform-appropriate, and persuasive. No hashtags unless Instagram."
        )
        result = call_gemini(prompt, api_key)
        if result:
            return result

    # Fallback rule-based templates
    templates = FALLBACK_TEMPLATES.get(segment, FALLBACK_TEMPLATES["Budget Buyers"])
    msg = random.choice(templates).format(name="Valued Customer", interest=interest)
    return msg


# ─────────────────────────────────────────────
# AGENT 4 – CAMPAIGN SCHEDULING
# ─────────────────────────────────────────────
SCHEDULE_RULES = {
    "Students / Low Spenders": {"platform": "Instagram", "time": "Evening (6–9 PM)",  "frequency": "3× / week"},
    "Budget Buyers":           {"platform": "Email",     "time": "Afternoon (1–4 PM)", "frequency": "2× / week"},
    "Premium Users":           {"platform": "WhatsApp",  "time": "Morning (8–11 AM)",  "frequency": "4× / week"},
}


def get_schedule(segment: str) -> dict:
    """Return campaign schedule for a segment."""
    return SCHEDULE_RULES.get(segment, SCHEDULE_RULES["Budget Buyers"])


# ─────────────────────────────────────────────
# AGENT 5 – PERFORMANCE MONITORING
# ─────────────────────────────────────────────
def generate_performance_data(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate before/after performance metrics per segment."""
    np.random.seed(42)
    segments = df["Segment"].unique()
    records = []
    for seg in segments:
        records.append({
            "Segment": seg,
            "Phase": "Before",
            "CTR (%)":        round(np.random.uniform(1.0, 3.5), 2),
            "Open Rate (%)":  round(np.random.uniform(10.0, 25.0), 2),
            "Conversion (%)": round(np.random.uniform(0.5, 2.5), 2),
        })
        records.append({
            "Segment": seg,
            "Phase": "After",
            "CTR (%)":        round(np.random.uniform(4.0, 9.0), 2),
            "Open Rate (%)":  round(np.random.uniform(28.0, 55.0), 2),
            "Conversion (%)": round(np.random.uniform(3.0, 8.0), 2),
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# AGENT 7 – AI STRATEGY OPTIMIZER
# ─────────────────────────────────────────────
def optimize_strategy(segment: str, interest: str, schedule: dict,
                      perf: dict, api_key: str = None) -> str:
    """Use Gemini (or fallback) to suggest improved marketing strategy."""
    if api_key:
        prompt = (
            f"You are a senior digital marketing strategist. Analyze the following campaign data and provide actionable improvements.\n\n"
            f"Customer Segment: {segment}\n"
            f"Primary Interest: {interest}\n"
            f"Current Campaign Plan:\n"
            f"  - Platform: {schedule['platform']}\n"
            f"  - Timing: {schedule['time']}\n"
            f"  - Frequency: {schedule['frequency']}\n"
            f"Performance Metrics:\n"
            f"  - CTR: {perf.get('CTR (%)', 'N/A')}%\n"
            f"  - Open Rate: {perf.get('Open Rate (%)', 'N/A')}%\n"
            f"  - Conversion Rate: {perf.get('Conversion (%)', 'N/A')}%\n\n"
            f"Provide a structured strategy optimization with:\n"
            f"1. Content improvements (tone, format, CTA)\n"
            f"2. Platform adjustments (add/change channels)\n"
            f"3. Timing & frequency refinements\n"
            f"4. One innovative tactic specific to this segment\n\n"
            f"Be specific and data-driven. Format with clear numbered points."
        )
        result = call_gemini(prompt, api_key)
        if result:
            return result

    # Fallback rule-based strategy
    strategies = {
        "Students / Low Spenders": (
            "📋 Strategy Optimization (Rule-Based Fallback)\n\n"
            "1. CONTENT: Use meme-style visuals and short video reels. Add urgency with countdown timers. "
            "CTA: 'Grab your student deal now — expires in 24hrs!'\n\n"
            "2. PLATFORM: Expand to TikTok alongside Instagram. Reels drive 3× higher engagement for under-25 cohort.\n\n"
            "3. TIMING: Push notifications at 7:30 PM for maximum reach during peak scroll time. "
            "Increase to 4× per week during semester starts.\n\n"
            "4. INNOVATION: Partner with student influencers for micro-campaigns. "
            "Offer referral credits — students trust peer recommendations over brand ads."
        ),
        "Budget Buyers": (
            "📋 Strategy Optimization (Rule-Based Fallback)\n\n"
            "1. CONTENT: Lead with price-comparison visuals ('Save ₹500 vs competitors'). "
            "Use testimonials emphasizing value. CTA: 'Best price guaranteed.'\n\n"
            "2. PLATFORM: A/B test SMS alongside Email. SMS has 98% open rate for time-sensitive deals.\n\n"
            "3. TIMING: Send Tuesday–Thursday between 1–3 PM for highest email open rates. "
            "Avoid weekends — low purchase intent.\n\n"
            "4. INNOVATION: Introduce a 'Price Drop Alert' subscription — notify when wishlist items go on sale. "
            "Builds habitual engagement."
        ),
        "Premium Users": (
            "📋 Strategy Optimization (Rule-Based Fallback)\n\n"
            "1. CONTENT: Use long-form storytelling: 'Crafted for those who appreciate the finest.' "
            "Avoid discount language — focus on experience, craftsmanship, exclusivity.\n\n"
            "2. PLATFORM: Add LinkedIn and curated newsletter alongside WhatsApp. "
            "Premium buyers respond to thought-leadership content.\n\n"
            "3. TIMING: Early morning (7–8 AM) messages see best response from high-income segments. "
            "Limit to 3× per week — avoid oversaturation.\n\n"
            "4. INNOVATION: Invite-only early access campaigns. 'You're among the first 50 to see this.' "
            "Exclusivity drives desire and conversion."
        ),
    }
    return strategies.get(segment, strategies["Budget Buyers"])


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "df_segmented" not in st.session_state:
    st.session_state.df_segmented = None
if "stats" not in st.session_state:
    st.session_state.stats = None
if "perf_df" not in st.session_state:
    st.session_state.perf_df = None


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.markdown("""
<div style='padding: 16px 0 8px 0;'>
  <div style='font-family: Space Mono, monospace; font-size: 1.1rem; color: #38bdf8; font-weight: 700;'>
    🧠 MarketMind AI
  </div>
  <div style='color: #475569; font-size: 0.75rem; margin-top: 4px;'>Multi-Agent Campaign System</div>
</div>
<hr style='border-color: #1e3a5f; margin: 8px 0 16px 0;'>
""", unsafe_allow_html=True)

pages = [
    "🏠  Home / Overview",
    "📂  Upload Dataset",
    "📊  Data Analysis",
    "🔵  Segmentation",
    "✍️  Content Generation",
    "📅  Campaign Scheduling",
    "📈  Performance Monitoring",
    "👤  Add Customer",
    "🤖  AI Strategy Optimizer",
]
page = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")

# API Key input in sidebar
st.sidebar.markdown("<hr style='border-color: #1e3a5f; margin: 16px 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;'>Gemini API Key</div>", unsafe_allow_html=True)
api_key_input = st.sidebar.text_input("API Key", type="password", placeholder="AIza...", label_visibility="collapsed")
if api_key_input:
    st.sidebar.markdown("<div style='color: #34d399; font-size: 0.75rem;'>✅ API Key set</div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<div style='color: #f59e0b; font-size: 0.75rem;'>⚠️ Using rule-based fallback</div>", unsafe_allow_html=True)

API_KEY = api_key_input if api_key_input else None


# ════════════════════════════════════════════
# PAGE: HOME / OVERVIEW
# ════════════════════════════════════════════
if page == "🏠  Home / Overview":
    st.markdown("""
    <div style='padding: 40px 0 20px 0;'>
      <div class='hero-title'>Multi-Agent Marketing<br>Campaign Optimizer</div>
      <div class='hero-sub'>Powered by Generative AI · K-Means Segmentation · Gemini LLM</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("7", "AI Agents"),
        ("K=3", "Segments"),
        ("3+", "Metrics Tracked"),
        ("Real-Time", "Predictions"),
    ]
    for col, (val, label) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-value'>{val}</div>
              <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    agents = [
        ("🧠", "Data Analysis Agent",       "blue",   "Loads, cleans and analyzes customer data. Encodes purchase frequency and generates descriptive statistics."),
        ("🔵", "Segmentation Agent",         "purple", "Applies K-Means clustering to divide customers into Students, Budget Buyers, and Premium Users."),
        ("✍️", "Content Generation Agent",   "green",  "Uses Google Gemini API (with rule-based fallback) to craft personalized marketing messages per segment."),
        ("📅", "Campaign Scheduling Agent",  "amber",  "Assigns optimal platform, timing, and frequency for each customer segment based on behavioral rules."),
        ("📈", "Performance Monitoring Agent","blue",  "Simulates and visualizes CTR, Open Rate, and Conversion Rate — before vs after personalization."),
        ("👤", "Customer Input Agent",       "purple", "Accepts real-time customer input and runs the full pipeline to predict segment, message, and schedule."),
        ("🤖", "AI Strategy Optimizer",      "green",  "Uses Gemini LLM reasoning to suggest data-driven improvements to content, platform, timing, and tactics."),
    ]

    for icon, name, color, desc in agents:
        st.markdown(f"""
        <div class='agent-card'>
          <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
            <span style='font-size: 1.5rem;'>{icon}</span>
            <span style='font-family: Space Mono, monospace; font-size: 1rem; color: #e2e8f0; font-weight: 700;'>{name}</span>
            <span class='badge badge-{color}'>AGENT</span>
          </div>
          <div style='color: #64748b; font-size: 0.9rem;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════
# PAGE: UPLOAD DATASET
# ════════════════════════════════════════════
elif page == "📂  Upload Dataset":
    st.markdown("<div class='section-header'>📂 Upload Dataset</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Upload your customer CSV or generate a synthetic dataset to get started.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='agent-card'>", unsafe_allow_html=True)
        st.markdown("**Upload CSV File**")
        st.markdown("<div style='color:#64748b; font-size:0.85rem; margin-bottom:12px;'>Required columns: User_ID, Age, Gender, Location, Interests, Purchase_Frequency, Preferred_Platform</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            try:
                df_raw = pd.read_csv(uploaded)
                df_clean, stats = clean_and_analyze(df_raw)
                st.session_state.df_clean = df_clean
                st.session_state.stats = stats
                st.session_state.df_segmented = None
                st.session_state.perf_df = None
                st.success(f"✅ Loaded {len(df_clean)} records from your file.")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='agent-card'>", unsafe_allow_html=True)
        st.markdown("**Generate Synthetic Dataset**")
        st.markdown("<div style='color:#64748b; font-size:0.85rem; margin-bottom:12px;'>No data? No problem. Generate a realistic customer dataset instantly.</div>", unsafe_allow_html=True)
        n_rows = st.slider("Number of customers", 50, 1000, 200, 50)
        if st.button("⚡ Generate Dataset"):
            df_raw = generate_synthetic_data(n_rows)
            df_clean, stats = clean_and_analyze(df_raw)
            st.session_state.df_clean = df_clean
            st.session_state.stats = stats
            st.session_state.df_segmented = None
            st.session_state.perf_df = None
            st.success(f"✅ Generated {len(df_clean)} synthetic customer records.")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.df_clean is not None:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("**Preview (first 10 rows)**")
        st.dataframe(st.session_state.df_clean.head(10), use_container_width=True)


# ════════════════════════════════════════════
# PAGE: DATA ANALYSIS
# ════════════════════════════════════════════
elif page == "📊  Data Analysis":
    st.markdown("<div class='section-header'>📊 Data Analysis Agent</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Analyzes your dataset: demographics, interests, platform preferences, and purchase behavior.</div>", unsafe_allow_html=True)

    if st.session_state.df_clean is None:
        st.warning("⚠️ No dataset loaded. Go to **Upload Dataset** first.")
        st.stop()

    df = st.session_state.df_clean
    stats = st.session_state.stats

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Users", stats["total_users"])
    c2.metric("Mean Age", stats["age_mean"])
    c3.metric("Age Std Dev", stats["age_std"])
    c4.metric("Unique Interests", df["Interests"].nunique())

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Age distribution
        fig_age = px.histogram(df, x="Age", nbins=20,
                               title="Age Distribution",
                               color_discrete_sequence=["#38bdf8"])
        fig_age.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", title_font_color="#e2e8f0",
            xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f")
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        # Gender distribution
        gender_df = pd.DataFrame(list(stats["gender_dist"].items()), columns=["Gender", "Count"])
        fig_gender = px.pie(gender_df, names="Gender", values="Count",
                            title="Gender Distribution",
                            color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399"])
        fig_gender.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8", title_font_color="#e2e8f0"
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Interests
        int_df = df["Interests"].value_counts().reset_index()
        int_df.columns = ["Interest", "Count"]
        fig_int = px.bar(int_df, x="Interest", y="Count",
                         title="Top Interests",
                         color="Count", color_continuous_scale="Blues")
        fig_int.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", title_font_color="#e2e8f0",
            xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f")
        )
        st.plotly_chart(fig_int, use_container_width=True)

    with col4:
        # Platform distribution
        plat_df = df["Preferred_Platform"].value_counts().reset_index()
        plat_df.columns = ["Platform", "Count"]
        fig_plat = px.bar(plat_df, x="Platform", y="Count",
                          title="Preferred Platforms",
                          color="Count", color_continuous_scale="Purples")
        fig_plat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", title_font_color="#e2e8f0",
            xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f")
        )
        st.plotly_chart(fig_plat, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**Cleaned Dataset**")
    st.dataframe(df, use_container_width=True)


# ════════════════════════════════════════════
# PAGE: SEGMENTATION
# ════════════════════════════════════════════
elif page == "🔵  Segmentation":
    st.markdown("<div class='section-header'>🔵 Segmentation Agent</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>K-Means clustering (K=3) groups customers into actionable segments based on age and purchase frequency.</div>", unsafe_allow_html=True)

    if st.session_state.df_clean is None:
        st.warning("⚠️ No dataset loaded. Go to **Upload Dataset** first.")
        st.stop()

    if st.button("🚀 Run Segmentation"):
        with st.spinner("Clustering customers..."):
            df_seg, _, _ = segment_customers(st.session_state.df_clean.copy())
            st.session_state.df_segmented = df_seg
            st.session_state.perf_df = None
        st.success("✅ Segmentation complete!")

    if st.session_state.df_segmented is not None:
        df_seg = st.session_state.df_segmented

        # Segment counts
        seg_counts = df_seg["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]

        c1, c2, c3 = st.columns(3)
        colors_map = {
            "Students / Low Spenders": "#38bdf8",
            "Budget Buyers": "#818cf8",
            "Premium Users": "#34d399",
        }
        for col, (_, row) in zip([c1, c2, c3], seg_counts.iterrows()):
            color = colors_map.get(row["Segment"], "#94a3b8")
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                  <div class='metric-value' style='color:{color};'>{row['Count']}</div>
                  <div class='metric-label'>{row['Segment']}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            fig_scatter = px.scatter(
                df_seg, x="Age", y="Purchase_Frequency_Encoded",
                color="Segment", title="Customer Clusters",
                color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399"],
                labels={"Purchase_Frequency_Encoded": "Purchase Frequency"},
            )
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8", title_font_color="#e2e8f0",
                xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f")
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            fig_pie = px.pie(seg_counts, names="Segment", values="Count",
                             title="Segment Distribution",
                             color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399"])
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8", title_font_color="#e2e8f0"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("**Segmented Dataset (first 20)**")
        st.dataframe(df_seg[["User_ID", "Age", "Gender", "Interests", "Purchase_Frequency", "Segment"]].head(20),
                     use_container_width=True)


# ════════════════════════════════════════════
# PAGE: CONTENT GENERATION
# ════════════════════════════════════════════
elif page == "✍️  Content Generation":
    st.markdown("<div class='section-header'>✍️ Content Generation Agent</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Generates personalized marketing messages using Google Gemini API or rule-based templates.</div>", unsafe_allow_html=True)

    if st.session_state.df_segmented is None:
        st.warning("⚠️ Run Segmentation first.")
        st.stop()

    df_seg = st.session_state.df_segmented

    col1, col2 = st.columns([1, 1])
    with col1:
        segment_options = df_seg["Segment"].unique().tolist()
        selected_seg = st.selectbox("Select Segment", segment_options)
    with col2:
        interest_options = df_seg["Interests"].unique().tolist()
        selected_interest = st.selectbox("Select Interest", sorted(interest_options))

    platform = get_schedule(selected_seg)["platform"]

    if st.button("✨ Generate Message"):
        with st.spinner("Generating personalized content..."):
            msg = generate_content(selected_seg, selected_interest, platform, API_KEY)
        
        source = "Gemini API" if API_KEY else "Rule-Based Fallback"
        st.markdown(f"""
        <div class='agent-card' style='margin-top:16px;'>
          <div style='display:flex; align-items:center; gap:8px; margin-bottom:12px;'>
            <span class='badge badge-{"green" if API_KEY else "amber"}'>{source}</span>
            <span class='badge badge-blue'>{selected_seg}</span>
            <span class='badge badge-purple'>{selected_interest}</span>
          </div>
          <div style='font-size:1.05rem; color:#e2e8f0; line-height:1.7;'>{msg}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**Bulk Content Preview (5 samples per segment)**")
    samples = []
    for seg in df_seg["Segment"].unique():
        seg_df = df_seg[df_seg["Segment"] == seg].head(5)
        for _, row in seg_df.iterrows():
            msg = generate_content(seg, row["Interests"], platform, None)  # Always use fallback for bulk
            samples.append({"User_ID": row["User_ID"], "Segment": seg,
                             "Interest": row["Interests"], "Message": msg})
    st.dataframe(pd.DataFrame(samples), use_container_width=True)


# ════════════════════════════════════════════
# PAGE: CAMPAIGN SCHEDULING
# ════════════════════════════════════════════
elif page == "📅  Campaign Scheduling":
    st.markdown("<div class='section-header'>📅 Campaign Scheduling Agent</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Assigns optimal platform, time slot, and frequency for each customer segment.</div>", unsafe_allow_html=True)

    if st.session_state.df_segmented is None:
        st.warning("⚠️ Run Segmentation first.")
        st.stop()

    df_seg = st.session_state.df_segmented

    # Display rules
    col1, col2, col3 = st.columns(3)
    seg_styles = [
        ("Students / Low Spenders", "blue",   "🎓"),
        ("Budget Buyers",           "purple", "💡"),
        ("Premium Users",           "green",  "👑"),
    ]
    for col, (seg, color, icon) in zip([col1, col2, col3], seg_styles):
        sched = get_schedule(seg)
        with col:
            st.markdown(f"""
            <div class='agent-card' style='text-align:center;'>
              <div style='font-size:2rem; margin-bottom:8px;'>{icon}</div>
              <div class='badge badge-{color}' style='margin-bottom:12px;'>{seg}</div>
              <hr class='divider' style='margin:12px 0;'>
              <div style='margin-bottom:8px;'>
                <div class='metric-label'>Platform</div>
                <div style='font-family: Space Mono; color:#e2e8f0; font-size:0.9rem;'>{sched['platform']}</div>
              </div>
              <div style='margin-bottom:8px;'>
                <div class='metric-label'>Send Time</div>
                <div style='font-family: Space Mono; color:#e2e8f0; font-size:0.9rem;'>{sched['time']}</div>
              </div>
              <div>
                <div class='metric-label'>Frequency</div>
                <div style='font-family: Space Mono; color:#e2e8f0; font-size:0.9rem;'>{sched['frequency']}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Add schedule columns to dataset
    df_sched = df_seg.copy()
    df_sched["Campaign_Platform"] = df_sched["Segment"].map(lambda s: get_schedule(s)["platform"])
    df_sched["Campaign_Time"]     = df_sched["Segment"].map(lambda s: get_schedule(s)["time"])
    df_sched["Campaign_Frequency"]= df_sched["Segment"].map(lambda s: get_schedule(s)["frequency"])

    st.markdown("**Full Campaign Schedule**")
    st.dataframe(
        df_sched[["User_ID", "Segment", "Campaign_Platform", "Campaign_Time", "Campaign_Frequency"]],
        use_container_width=True
    )


# ════════════════════════════════════════════
# PAGE: PERFORMANCE MONITORING
# ════════════════════════════════════════════
elif page == "📈  Performance Monitoring":
    st.markdown("<div class='section-header'>📈 Performance Monitoring Agent</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Tracks CTR, Open Rate, and Conversion Rate — compares before vs after personalization and identifies best/low performers.</div>", unsafe_allow_html=True)

    if st.session_state.df_segmented is None:
        st.warning("⚠️ Run Segmentation first.")
        st.stop()

    if st.session_state.perf_df is None:
        st.session_state.perf_df = generate_performance_data(st.session_state.df_segmented)

    perf_df = st.session_state.perf_df

    # Before vs After bar charts
    metrics = ["CTR (%)", "Open Rate (%)", "Conversion (%)"]
    colors_before = "#334155"
    colors_after  = "#38bdf8"

    for metric in metrics:
        fig = px.bar(
            perf_df, x="Segment", y=metric, color="Phase",
            barmode="group", title=f"{metric} — Before vs After Personalization",
            color_discrete_map={"Before": colors_before, "After": colors_after},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", title_font_color="#e2e8f0",
            xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#94a3b8")
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Best / Low performers
    after_df = perf_df[perf_df["Phase"] == "After"].copy()
    after_df["Score"] = after_df["CTR (%)"] + after_df["Conversion (%)"]
    best_seg = after_df.sort_values("Score", ascending=False).iloc[0]
    low_seg  = after_df.sort_values("Score").iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='agent-card'>
          <div style='color:#34d399; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;'>🏆 Best Performer</div>
          <div style='font-family: Space Mono; font-size:1.1rem; color:#e2e8f0;'>{best_seg['Segment']}</div>
          <div style='color:#64748b; font-size:0.85rem; margin-top:8px;'>
            CTR: {best_seg['CTR (%)']:.2f}% · Conversion: {best_seg['Conversion (%)']:.2f}%
          </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='agent-card'>
          <div style='color:#f59e0b; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;'>⚠️ Needs Improvement</div>
          <div style='font-family: Space Mono; font-size:1.1rem; color:#e2e8f0;'>{low_seg['Segment']}</div>
          <div style='color:#64748b; font-size:0.85rem; margin-top:8px;'>
            CTR: {low_seg['CTR (%)']:.2f}% · Conversion: {low_seg['Conversion (%)']:.2f}%
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**Raw Performance Table**")
    st.dataframe(perf_df, use_container_width=True)


# ════════════════════════════════════════════
# PAGE: ADD CUSTOMER (REAL-TIME PREDICTION)
# ════════════════════════════════════════════
elif page == "👤  Add Customer":
    st.markdown("<div class='section-header'>👤 Add Customer / Predict Strategy</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Enter a new customer's details and instantly receive their segment, personalized message, and campaign plan.</div>", unsafe_allow_html=True)

    if st.session_state.df_segmented is None:
        st.warning("⚠️ Run Segmentation first (the model needs to be trained on existing data).")
        st.stop()

    with st.container():
        st.markdown("<div class='agent-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            age_input    = st.number_input("Age", min_value=10, max_value=100, value=25)
            gender_input = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
            location_input = st.text_input("Location", value="Bangalore")
        with col2:
            interests_input = st.selectbox("Primary Interest",
                sorted(["Sports", "Fashion", "Technology", "Travel", "Food",
                        "Music", "Gaming", "Fitness", "Books", "Art"]))
            pf_input = st.selectbox("Purchase Frequency", ["Low", "Medium", "High"])

        predict_clicked = st.button("🔮 Predict Strategy")
        st.markdown("</div>", unsafe_allow_html=True)

    if predict_clicked:
        # Encode purchase frequency
        pf_encoded = {"Low": 1, "Medium": 2, "High": 3}[pf_input]

        # Re-train scaler on existing segmented data
        df_seg = st.session_state.df_segmented
        features = df_seg[["Age", "Purchase_Frequency_Encoded"]].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X_scaled)

        # Assign labels to clusters
        centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                                columns=["Age_c", "PF_c"])
        label_map = {}
        for idx, row in centers.iterrows():
            if row["PF_c"] <= 1.4:
                label_map[idx] = "Students / Low Spenders"
            elif row["PF_c"] <= 2.4:
                label_map[idx] = "Budget Buyers"
            else:
                label_map[idx] = "Premium Users"
        if len(set(label_map.values())) < 3:
            sorted_by_age = centers.sort_values("Age_c").index.tolist()
            label_map = {sorted_by_age[0]: "Students / Low Spenders",
                         sorted_by_age[1]: "Budget Buyers",
                         sorted_by_age[2]: "Premium Users"}

        # Predict cluster for new customer
        new_point = scaler.transform([[age_input, pf_encoded]])
        cluster_id = kmeans.predict(new_point)[0]
        pred_segment = label_map[cluster_id]

        # Generate content and schedule
        pred_schedule = get_schedule(pred_segment)
        pred_message  = generate_content(pred_segment, interests_input,
                                          pred_schedule["platform"], API_KEY)

        # Display results
        badge_colors = {
            "Students / Low Spenders": "blue",
            "Budget Buyers": "purple",
            "Premium Users": "green",
        }
        bcolor = badge_colors.get(pred_segment, "blue")

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("**📊 Prediction Results**")

        c1, c2, c3, c4 = st.columns(4)
        results = [
            ("Segment",   pred_segment),
            ("Platform",  pred_schedule["platform"]),
            ("Send Time", pred_schedule["time"]),
            ("Frequency", pred_schedule["frequency"]),
        ]
        for col, (label, val) in zip([c1, c2, c3, c4], results):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                  <div class='metric-label'>{label}</div>
                  <div style='color:#e2e8f0; font-family:Space Mono; font-size:0.85rem; margin-top:6px;'>{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='agent-card'>
          <div style='display:flex; align-items:center; gap:8px; margin-bottom:12px;'>
            <span class='badge badge-{bcolor}'>{pred_segment}</span>
            <span class='badge badge-blue'>{interests_input}</span>
            <span style='color:#64748b; font-size:0.8rem;'>Personalized Message</span>
          </div>
          <div style='font-size:1.05rem; color:#e2e8f0; line-height:1.7;'>{pred_message}</div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════
# PAGE: AI STRATEGY OPTIMIZER
# ════════════════════════════════════════════
elif page == "🤖  AI Strategy Optimizer":
    st.markdown("<div class='section-header'>🤖 AI Strategy Optimizer</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Uses Google Gemini LLM to generate intelligent, data-driven marketing strategy improvements — not just templates.</div>", unsafe_allow_html=True)

    if st.session_state.df_segmented is None:
        st.warning("⚠️ Run Segmentation first.")
        st.stop()

    df_seg = st.session_state.df_segmented

    if st.session_state.perf_df is None:
        st.session_state.perf_df = generate_performance_data(df_seg)
    perf_df = st.session_state.perf_df

    st.markdown("<div class='agent-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        seg_options = df_seg["Segment"].unique().tolist()
        opt_segment  = st.selectbox("Segment to Optimize", seg_options)
    with col2:
        opt_interest = st.selectbox("Customer Interest",
            sorted(df_seg["Interests"].unique().tolist()))

    # Get performance data for chosen segment
    seg_perf_after = perf_df[(perf_df["Segment"] == opt_segment) & (perf_df["Phase"] == "After")]
    if not seg_perf_after.empty:
        perf_row = seg_perf_after.iloc[0].to_dict()
    else:
        perf_row = {"CTR (%)": "N/A", "Open Rate (%)": "N/A", "Conversion (%)": "N/A"}

    # Show current metrics
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, key in zip([c1, c2, c3], ["CTR (%)", "Open Rate (%)", "Conversion (%)"]):
        val = perf_row.get(key, "N/A")
        display = f"{val:.2f}%" if isinstance(val, float) else val
        with col:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>{key}</div>
              <div style='color:#38bdf8; font-family:Space Mono; font-size:1.2rem; margin-top:4px;'>{display}</div>
            </div>""", unsafe_allow_html=True)

    optimize_clicked = st.button("🧠 Generate AI Strategy")
    st.markdown("</div>", unsafe_allow_html=True)

    if optimize_clicked:
        schedule = get_schedule(opt_segment)
        with st.spinner("AI is analyzing and crafting your strategy..."):
            strategy = optimize_strategy(opt_segment, opt_interest, schedule, perf_row, API_KEY)

        source_label = "Gemini AI" if API_KEY else "Rule-Based Fallback"
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:8px; margin-bottom:12px;'>
          <span class='badge badge-purple'>{opt_segment}</span>
          <span class='badge badge-{"green" if API_KEY else "amber"}'>{source_label}</span>
          <span style='color:#64748b; font-size:0.8rem;'>Optimized Marketing Strategy</span>
        </div>
        <div class='strategy-box'>{strategy}</div>
        """, unsafe_allow_html=True)

        # Recommended schedule comparison
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Current vs Recommended Plan**")
        comparison = pd.DataFrame({
            "Attribute": ["Platform", "Time", "Frequency"],
            "Current":   [schedule["platform"], schedule["time"], schedule["frequency"]],
            "Recommended": ["See strategy above", "See strategy above", "See strategy above"],
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)