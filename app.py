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

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="MarketMind AI",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

.stApp {
    background: #0a0e1a;
    color: #e0e6f0;
}

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

.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.5px;
}
.badge-blue   { background: rgba(56,189,248,0.15);  color: #38bdf8; border: 1px solid rgba(56,189,248,0.3); }
.badge-purple { background: rgba(129,140,248,0.15); color: #818cf8; border: 1px solid rgba(129,140,248,0.3); }
.badge-green  { background: rgba(52,211,153,0.15);  color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.badge-amber  { background: rgba(251,191,36,0.15);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }

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

.divider {
    border: none;
    border-top: 1px solid #1e3a5f;
    margin: 24px 0;
}

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

.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: #111827 !important;
    border: 1px solid #1e3a5f !important;
    color: #e0e6f0 !important;
    border-radius: 8px !important;
}

.stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# GEMINI API HELPER
# -------------------------------------------------
def call_gemini(prompt: str, api_key: str) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except ImportError:
        return None
    except Exception:
        return None


# -------------------------------------------------
# AGENT 1 - DATA ANALYSIS
# -------------------------------------------------
def generate_synthetic_data(n: int = 200) -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)

    ages          = np.random.randint(10, 65, n)
    genders       = np.random.choice(["Male", "Female", "Non-binary"], n, p=[0.48, 0.48, 0.04])
    locations     = np.random.choice(["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune"], n)
    interests_pool = ["Sports", "Fashion", "Technology", "Travel", "Food",
                      "Music", "Gaming", "Fitness", "Books", "Art"]
    interests     = [random.choice(interests_pool) for _ in range(n)]
    purchase_freq = np.random.choice(["Low", "Medium", "High"], n, p=[0.35, 0.40, 0.25])
    platforms     = np.random.choice(["Instagram", "Email", "WhatsApp", "YouTube", "Twitter"], n)

    df = pd.DataFrame({
        "User_ID":            [f"U{str(i).zfill(4)}" for i in range(1, n + 1)],
        "Age":                ages,
        "Gender":             genders,
        "Location":           locations,
        "Interests":          interests,
        "Purchase_Frequency": purchase_freq,
        "Preferred_Platform": platforms,
    })
    return df


def clean_and_analyze(df: pd.DataFrame):
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(df["Age"].median()).astype(int)
    freq_map = {"Low": 1, "Medium": 2, "High": 3}
    df["Purchase_Frequency_Encoded"] = df["Purchase_Frequency"].map(freq_map).fillna(1).astype(int)

    stats = {
        "total_users":   len(df),
        "age_mean":      round(df["Age"].mean(), 1),
        "age_std":       round(df["Age"].std(), 1),
        "gender_dist":   df["Gender"].value_counts().to_dict(),
        "top_interests": df["Interests"].value_counts().head(5).to_dict(),
        "platform_dist": df["Preferred_Platform"].value_counts().to_dict(),
        "purchase_dist": df["Purchase_Frequency"].value_counts().to_dict(),
    }
    return df, stats


# -------------------------------------------------
# SEGMENTATION LOGIC
# Age 10-24                          -> Student / Low Spender
# Age 25-40, Low/Medium frequency    -> Budget Buyer
# Age 25-40, High frequency          -> Premium User
# Age 41+,   Low/Medium frequency    -> Budget Buyer
# Age 41+,   High frequency          -> Premium User
# -------------------------------------------------
def assign_segment_rule(age: int, pf_encoded: int) -> str:
    """Rule-based segmentation — no ML clustering."""
    if age <= 24:
        return "Students / Low Spenders"
    elif pf_encoded == 3:          # High purchase frequency
        return "Premium Users"
    else:                          # Low or Medium purchase frequency
        return "Budget Buyers"


def segment_customers(df: pd.DataFrame):
    """Apply rule-based segmentation to the entire dataframe."""
    df = df.copy()
    df["Segment"] = df.apply(
        lambda row: assign_segment_rule(row["Age"], row["Purchase_Frequency_Encoded"]),
        axis=1,
    )
    return df


# -------------------------------------------------
# AGENT 3 - CONTENT GENERATION
# -------------------------------------------------
FALLBACK_TEMPLATES = {
    "Students / Low Spenders": [
        "Hey there! Score big with our student discount — up to 40% off on {interest} gear. Limited time!",
        "We know student life is tough. That is why we are giving you exclusive {interest} deals — just for you!",
        "Budget smart! Our special student offer on {interest} products is live. Do not miss it!",
    ],
    "Budget Buyers": [
        "Smart shopping starts here! Get the best value on {interest} — quality you can afford.",
        "Great deals for savvy buyers! Check out our curated {interest} collection under your budget.",
        "Value-packed offers on {interest} are waiting for you. Shop more, spend less!",
    ],
    "Premium Users": [
        "Exclusively for you — our luxury {interest} collection. Experience the finest, because you deserve it.",
        "Elevate your lifestyle. Discover our premium {interest} picks curated for discerning tastes.",
        "Your premium membership unlocks exclusive {interest} experiences. Indulge today.",
    ],
}


def generate_content(segment: str, interest: str, platform: str, api_key: str = None) -> str:
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

    templates = FALLBACK_TEMPLATES.get(segment, FALLBACK_TEMPLATES["Budget Buyers"])
    msg = random.choice(templates).format(name="Valued Customer", interest=interest)
    return msg


# -------------------------------------------------
# AGENT 4 - CAMPAIGN SCHEDULING
# -------------------------------------------------
SCHEDULE_RULES = {
    "Students / Low Spenders": {"platform": "Instagram", "time": "Evening (6-9 PM)",   "frequency": "3x / week"},
    "Budget Buyers":           {"platform": "Email",     "time": "Afternoon (1-4 PM)", "frequency": "2x / week"},
    "Premium Users":           {"platform": "WhatsApp",  "time": "Morning (8-11 AM)",  "frequency": "4x / week"},
}


def get_schedule(segment: str) -> dict:
    return SCHEDULE_RULES.get(segment, SCHEDULE_RULES["Budget Buyers"])


# -------------------------------------------------
# AGENT 5 - PERFORMANCE MONITORING
# -------------------------------------------------
def generate_performance_data(df: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(42)
    segments = df["Segment"].unique()
    records = []
    for seg in segments:
        records.append({
            "Segment":        seg,
            "Phase":          "Before",
            "CTR (%)":        round(np.random.uniform(1.0, 3.5), 2),
            "Open Rate (%)":  round(np.random.uniform(10.0, 25.0), 2),
            "Conversion (%)": round(np.random.uniform(0.5, 2.5), 2),
        })
        records.append({
            "Segment":        seg,
            "Phase":          "After",
            "CTR (%)":        round(np.random.uniform(4.0, 9.0), 2),
            "Open Rate (%)":  round(np.random.uniform(28.0, 55.0), 2),
            "Conversion (%)": round(np.random.uniform(3.0, 8.0), 2),
        })
    return pd.DataFrame(records)


# -------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "df_segmented" not in st.session_state:
    st.session_state.df_segmented = None
if "stats" not in st.session_state:
    st.session_state.stats = None
if "perf_df" not in st.session_state:
    st.session_state.perf_df = None


# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
st.sidebar.markdown("""
<div style='padding: 16px 0 8px 0;'>
  <div style='font-family: Space Mono, monospace; font-size: 1.1rem; color: #38bdf8; font-weight: 700;'>
    MarketMind AI
  </div>
  <div style='color: #475569; font-size: 0.75rem; margin-top: 4px;'>Multi-Agent Campaign System</div>
</div>
<hr style='border-color: #1e3a5f; margin: 8px 0 16px 0;'>
""", unsafe_allow_html=True)

pages = [
    "Home / Overview",
    "Upload Dataset",
    "Data Analysis",
    "Segmentation",
    "Content Generation",
    "Campaign Scheduling",
    "Performance Monitoring",
    "Add Customer",
]
page = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")

st.sidebar.markdown("<hr style='border-color: #1e3a5f; margin: 16px 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;'>Gemini API Key</div>", unsafe_allow_html=True)
api_key_input = st.sidebar.text_input("API Key", type="password", placeholder="AIza...", label_visibility="collapsed")
if api_key_input:
    st.sidebar.markdown("<div style='color: #34d399; font-size: 0.75rem;'>API Key set</div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<div style='color: #f59e0b; font-size: 0.75rem;'>Using rule-based fallback</div>", unsafe_allow_html=True)

API_KEY = api_key_input if api_key_input else None


# ================================================
# PAGE: HOME / OVERVIEW
# ================================================
if page == "Home / Overview":
    st.markdown("""
    <div style='padding: 40px 0 20px 0;'>
      <div class='hero-title'>Multi-Agent Marketing<br>Campaign Optimizer</div>
      <div class='hero-sub'>Powered by Generative AI · Rule-Based Segmentation · Gemini LLM</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("6", "AI Agents"),
        ("3",  "Segments"),
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
        ("Data Analysis Agent",
         "blue",
         "Loads, cleans and analyzes customer data. Encodes purchase frequency and generates descriptive statistics."),
        ("Segmentation Agent",
         "purple",
         "Applies age and purchase frequency rules to divide customers into Students, Budget Buyers, and Premium Users."),
        ("Content Generation Agent",
         "green",
         "Uses Google Gemini API (with rule-based fallback) to craft personalized marketing messages per segment."),
        ("Campaign Scheduling Agent",
         "amber",
         "Assigns optimal platform, timing, and frequency for each customer segment based on behavioral rules."),
        ("Performance Monitoring Agent",
         "blue",
         "Simulates and visualizes CTR, Open Rate, and Conversion Rate — before vs after personalization."),
        ("Customer Input Agent",
         "purple",
         "Accepts real-time customer input and runs the full pipeline to predict segment, message, and campaign plan."),
    ]

    for name, color, desc in agents:
        st.markdown(f"""
        <div class='agent-card'>
          <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
            <span style='font-family: Space Mono, monospace; font-size: 1rem; color: #e2e8f0; font-weight: 700;'>{name}</span>
            <span class='badge badge-{color}'>AGENT</span>
          </div>
          <div style='color: #64748b; font-size: 0.9rem;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # Segmentation rules reference card
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Segmentation Rules</div>", unsafe_allow_html=True)
    rules = [
        ("Age 10 – 24",   "Any purchase frequency",          "Students / Low Spenders"),
        ("Age 25 – 40",   "Low or Medium purchase frequency", "Budget Buyers"),
        ("Age 25 – 40",   "High purchase frequency",          "Premium Users"),
        ("Age 41 and above", "Low or Medium purchase frequency", "Budget Buyers"),
        ("Age 41 and above", "High purchase frequency",          "Premium Users"),
    ]
    rules_df = pd.DataFrame(rules, columns=["Age Range", "Purchase Frequency", "Segment"])
    st.dataframe(rules_df, use_container_width=True, hide_index=True)


# ================================================
# PAGE: UPLOAD DATASET
# ================================================
elif page == "Upload Dataset":
    st.markdown("<div class='section-header'>Upload Dataset</div>", unsafe_allow_html=True)
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
                st.success(f"Loaded {len(df_clean)} records from your file.")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='agent-card'>", unsafe_allow_html=True)
        st.markdown("**Generate Synthetic Dataset**")
        st.markdown("<div style='color:#64748b; font-size:0.85rem; margin-bottom:12px;'>No data? No problem. Generate a realistic customer dataset instantly.</div>", unsafe_allow_html=True)
        n_rows = st.slider("Number of customers", 50, 1000, 200, 50)
        if st.button("Generate Dataset"):
            df_raw = generate_synthetic_data(n_rows)
            df_clean, stats = clean_and_analyze(df_raw)
            st.session_state.df_clean = df_clean
            st.session_state.stats = stats
            st.session_state.df_segmented = None
            st.session_state.perf_df = None
            st.success(f"Generated {len(df_clean)} synthetic customer records.")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.df_clean is not None:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("**Preview (first 10 rows)**")
        st.dataframe(st.session_state.df_clean.head(10), use_container_width=True)


# ================================================
# PAGE: DATA ANALYSIS
# ================================================
elif page == "Data Analysis":
    st.markdown("<div class='section-header'>Data Analysis Agent</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Analyzes your dataset: demographics, interests, platform preferences, and purchase behavior.</div>", unsafe_allow_html=True)

    if st.session_state.df_clean is None:
        st.warning("No dataset loaded. Go to Upload Dataset first.")
        st.stop()

    df    = st.session_state.df_clean
    stats = st.session_state.stats

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Users",       stats["total_users"])
    c2.metric("Mean Age",          stats["age_mean"])
    c3.metric("Age Std Dev",       stats["age_std"])
    c4.metric("Unique Interests",  df["Interests"].nunique())

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
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


# ================================================
# PAGE: SEGMENTATION
# ================================================
elif page == "Segmentation":
    st.markdown("<div class='section-header'>Segmentation Agent</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Rule-based segmentation groups customers by age and purchase frequency into three actionable segments.</div>", unsafe_allow_html=True)

    if st.session_state.df_clean is None:
        st.warning("No dataset loaded. Go to Upload Dataset first.")
        st.stop()

    # Segmentation rules reference
    st.markdown("<div class='agent-card'>", unsafe_allow_html=True)
    st.markdown("**Segmentation Rules Applied**")
    rules = [
        ("10 – 24",       "Any",              "Students / Low Spenders"),
        ("25 – 40",       "Low / Medium",     "Budget Buyers"),
        ("25 – 40",       "High",             "Premium Users"),
        ("41 and above",  "Low / Medium",     "Budget Buyers"),
        ("41 and above",  "High",             "Premium Users"),
    ]
    rules_df = pd.DataFrame(rules, columns=["Age Range", "Purchase Frequency", "Segment"])
    st.dataframe(rules_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Run Segmentation"):
        with st.spinner("Segmenting customers..."):
            df_seg = segment_customers(st.session_state.df_clean.copy())
            st.session_state.df_segmented = df_seg
            st.session_state.perf_df = None
        st.success("Segmentation complete!")

    if st.session_state.df_segmented is not None:
        df_seg = st.session_state.df_segmented

        seg_counts = df_seg["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]

        c1, c2, c3 = st.columns(3)
        colors_map = {
            "Students / Low Spenders": "#38bdf8",
            "Budget Buyers":           "#818cf8",
            "Premium Users":           "#34d399",
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
                color="Segment", title="Customer Segments",
                color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399"],
                labels={"Purchase_Frequency_Encoded": "Purchase Frequency (encoded)"},
            )
            # Vertical reference lines
            fig_scatter.add_vline(x=24.5, line_dash="dash", line_color="#475569",
                                  annotation_text="Age 24/25", annotation_font_color="#64748b")
            fig_scatter.add_vline(x=40.5, line_dash="dash", line_color="#475569",
                                  annotation_text="Age 40/41", annotation_font_color="#64748b")
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
        st.dataframe(
            df_seg[["User_ID", "Age", "Gender", "Interests", "Purchase_Frequency", "Segment"]].head(20),
            use_container_width=True,
        )


# ================================================
# PAGE: CONTENT GENERATION
# ================================================
elif page == "Content Generation":
    st.markdown("<div class='section-header'>Content Generation Agent</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Generates personalized marketing messages using Google Gemini API or rule-based templates.</div>", unsafe_allow_html=True)

    if st.session_state.df_segmented is None:
        st.warning("Run Segmentation first.")
        st.stop()

    df_seg = st.session_state.df_segmented

    col1, col2 = st.columns([1, 1])
    with col1:
        segment_options    = df_seg["Segment"].unique().tolist()
        selected_seg       = st.selectbox("Select Segment", segment_options)
    with col2:
        interest_options   = df_seg["Interests"].unique().tolist()
        selected_interest  = st.selectbox("Select Interest", sorted(interest_options))

    platform = get_schedule(selected_seg)["platform"]

    if st.button("Generate Message"):
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
            msg = generate_content(seg, row["Interests"], platform, None)
            samples.append({"User_ID": row["User_ID"], "Segment": seg,
                             "Interest": row["Interests"], "Message": msg})
    st.dataframe(pd.DataFrame(samples), use_container_width=True)


# ================================================
# PAGE: CAMPAIGN SCHEDULING
# ================================================
elif page == "Campaign Scheduling":
    st.markdown("<div class='section-header'>Campaign Scheduling Agent</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Assigns optimal platform, time slot, and frequency for each customer segment.</div>", unsafe_allow_html=True)

    if st.session_state.df_segmented is None:
        st.warning("Run Segmentation first.")
        st.stop()

    df_seg = st.session_state.df_segmented

    col1, col2, col3 = st.columns(3)
    seg_styles = [
        ("Students / Low Spenders", "blue"),
        ("Budget Buyers",           "purple"),
        ("Premium Users",           "green"),
    ]
    for col, (seg, color) in zip([col1, col2, col3], seg_styles):
        sched = get_schedule(seg)
        with col:
            st.markdown(f"""
            <div class='agent-card' style='text-align:center;'>
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

    df_sched = df_seg.copy()
    df_sched["Campaign_Platform"]  = df_sched["Segment"].map(lambda s: get_schedule(s)["platform"])
    df_sched["Campaign_Time"]      = df_sched["Segment"].map(lambda s: get_schedule(s)["time"])
    df_sched["Campaign_Frequency"] = df_sched["Segment"].map(lambda s: get_schedule(s)["frequency"])

    st.markdown("**Full Campaign Schedule**")
    st.dataframe(
        df_sched[["User_ID", "Segment", "Campaign_Platform", "Campaign_Time", "Campaign_Frequency"]],
        use_container_width=True,
    )


# ================================================
# PAGE: PERFORMANCE MONITORING
# ================================================
elif page == "Performance Monitoring":
    st.markdown("<div class='section-header'>Performance Monitoring Agent</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Tracks CTR, Open Rate, and Conversion Rate — compares before vs after personalization and identifies best and low performers.</div>", unsafe_allow_html=True)

    if st.session_state.df_segmented is None:
        st.warning("Run Segmentation first.")
        st.stop()

    if st.session_state.perf_df is None:
        st.session_state.perf_df = generate_performance_data(st.session_state.df_segmented)

    perf_df = st.session_state.perf_df

    metrics         = ["CTR (%)", "Open Rate (%)", "Conversion (%)"]
    colors_before   = "#334155"
    colors_after    = "#38bdf8"

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

    after_df          = perf_df[perf_df["Phase"] == "After"].copy()
    after_df["Score"] = after_df["CTR (%)"] + after_df["Conversion (%)"]
    best_seg          = after_df.sort_values("Score", ascending=False).iloc[0]
    low_seg           = after_df.sort_values("Score").iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='agent-card'>
          <div style='color:#34d399; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;'>Best Performer</div>
          <div style='font-family: Space Mono; font-size:1.1rem; color:#e2e8f0;'>{best_seg['Segment']}</div>
          <div style='color:#64748b; font-size:0.85rem; margin-top:8px;'>
            CTR: {best_seg['CTR (%)']:.2f}% &middot; Conversion: {best_seg['Conversion (%)']:.2f}%
          </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='agent-card'>
          <div style='color:#f59e0b; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;'>Needs Improvement</div>
          <div style='font-family: Space Mono; font-size:1.1rem; color:#e2e8f0;'>{low_seg['Segment']}</div>
          <div style='color:#64748b; font-size:0.85rem; margin-top:8px;'>
            CTR: {low_seg['CTR (%)']:.2f}% &middot; Conversion: {low_seg['Conversion (%)']:.2f}%
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**Raw Performance Table**")
    st.dataframe(perf_df, use_container_width=True)


# ================================================
# PAGE: ADD CUSTOMER (REAL-TIME PREDICTION)
# ================================================
elif page == "Add Customer":
    st.markdown("<div class='section-header'>Add Customer / Predict Strategy</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Enter a new customer's details and instantly receive their segment, personalized message, and campaign plan.</div>", unsafe_allow_html=True)

    # Show segmentation rules for reference
    

    with st.container():
        st.markdown("<div class='agent-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            age_input      = st.number_input("Age", min_value=10, max_value=100, value=25)
            gender_input   = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
            location_input = st.text_input("Location", value="Bangalore")
        with col2:
            interests_input = st.selectbox("Primary Interest",
                sorted(["Sports", "Fashion", "Technology", "Travel", "Food",
                        "Music", "Gaming", "Fitness", "Books", "Art"]))
            pf_input = st.selectbox("Purchase Frequency", ["Low", "Medium", "High"])

        predict_clicked = st.button("Predict Strategy")
        st.markdown("</div>", unsafe_allow_html=True)

    if predict_clicked:
        pf_encoded    = {"Low": 1, "Medium": 2, "High": 3}[pf_input]
        pred_segment  = assign_segment_rule(age_input, pf_encoded)
        pred_schedule = get_schedule(pred_segment)
        pred_message  = generate_content(pred_segment, interests_input,
                                          pred_schedule["platform"], API_KEY)

        badge_colors = {
            "Students / Low Spenders": "blue",
            "Budget Buyers":           "purple",
            "Premium Users":           "green",
        }
        bcolor = badge_colors.get(pred_segment, "blue")

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("**Prediction Results**")

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

        # Explanation of why this segment was assigned
        st.markdown("<br>", unsafe_allow_html=True)
        if age_input <= 24:
            reason = f"Age {age_input} falls in the 10-24 range, so this customer is classified as a Student / Low Spender regardless of purchase frequency."
        elif pf_encoded == 3:
            reason = f"Age {age_input} is in the {('25-40' if age_input <= 40 else '41+') } range with High purchase frequency, qualifying as a Premium User."
        else:
            reason = f"Age {age_input} is in the {('25-40' if age_input <= 40 else '41+') } range with {pf_input} purchase frequency, qualifying as a Budget Buyer."

        st.markdown(f"""
        <div class='output-box'>Segmentation reason: {reason}</div>
        """, unsafe_allow_html=True)