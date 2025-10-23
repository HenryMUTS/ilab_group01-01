# st_theme.py
import streamlit as st
import base64
import os

# Get the absolute path to the logo from the root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(ROOT_DIR, "assets", "logo.png")

def set_page_theme(title="Nosevision AI", icon="dashboard", spin_logo=False):
    """Apply custom theme, branding, and CSS for the Segmenta app."""
    st.set_page_config(page_title=title, page_icon="", layout="wide")

    # --- Load logo ---
    try:
        logo_b64 = base64.b64encode(open(LOGO_PATH, "rb").read()).decode()
    except Exception:
        logo_b64 = ""

    st.markdown("""
    <link rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
    <style>
    .material-symbol {
      font-family: 'Material Symbols Outlined';
      font-weight: normal;
      font-style: normal;
      font-size: 24px; /* Adjust size */
      line-height: 1;
      display: inline-block;
      vertical-align: middle;
      color: #96C7B6; /* Updated to mint green color */
      font-variation-settings:
        'FILL' 0,
        'wght' 400,
        'GRAD' 0,
        'opsz' 24;
      -webkit-font-feature-settings: 'liga';
      font-feature-settings: 'liga';
    }
    .material-symbols-outlined {
      font-family: 'Material Symbols Outlined';
      font-weight: normal;
      font-style: normal;
      font-size: 24px;
      line-height: 1;
      display: inline-block;
      vertical-align: middle;
      color: #96C7B6;
      font-variation-settings:
        'FILL' 0,
        'wght' 400,
        'GRAD' 0,
        'opsz' 24;
      -webkit-font-feature-settings: 'liga';
      font-feature-settings: 'liga';
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Branding block ---
    st.markdown(f"""
    <style>
    .brand {{
      display:flex;align-items:center;gap:0px;line-height:1;
    }}
    .brand .icon {{
      font-family: 'Material Symbols Outlined';
      font-size: 48px;
      color: #96C7B6;
      text-shadow: 0 2px 8px rgba(150, 199, 182, 0.3);
      font-feature-settings: 'liga';
      -webkit-font-feature-settings: 'liga';
      -webkit-font-smoothing: antialiased;
    }}
    .brand img.logo {{
      height:116px;width:auto;display:block;margin:0;
      transform-origin:50% 50%;
      filter: drop-shadow(0 6px 12px rgba(0,0,0,0.25));
    }}
   
           
    .brand img.logo.spin {{
      animation:spin 1s linear infinite;
    }}
    @keyframes spin {{from{{transform:rotate(0)}}to{{transform:rotate(360deg)}}}}
    .brand h1{{
      margin:0;
      margin-left:0;
      font-family:'Calibri', 'Calibri Light', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      font-weight:300;
      font-size:clamp(28px,5vw,56px);
      color:#0F172A;
      text-shadow:0 1px 0 rgba(255,255,255,.6),0 8px 24px rgba(15,23,42,.15);
    }}
    </style>
    <div class="brand">
      {"<img class='logo " + ("spin" if spin_logo else "") + "' src='data:image/png;base64," + logo_b64 + "' />" if logo_b64 else "<span class='material-symbol icon'>" + icon + "</span>"}
      <h1>{title}</h1>
    </div>
    """, unsafe_allow_html=True)

    # --- Global card-like shadow for blocks (tables, charts, figures) ---
    st.markdown("""
    <style>
    /* Tables */
    .stDataFrame, .stTable {
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
        border-radius: 12px;
        padding: 10px;
        background: white;
    }

    /* Matplotlib/Plotly figures */
    [data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"], [data-testid="stAltairChart"], [data-testid="stDeckGlJsonChart"], .stImage {
        box-shadow: 0 8px 22px rgba(0,0,0,0.15);
        border-radius: 12px;
        background: white;
        padding: 8px;
    }

    /* Expander + Metrics (already styled but add consistency) */
    div[data-testid="stMetric"], div[data-testid="stExpander"] {
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
        border-radius: 12px;
        background: white;
    }

    /* Callout (info box) */
    .callout {
      font-family: 'Calibri', ui-sans-serif, system-ui, -apple-system,
                   'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      font-weight: 300;
      background: #f0f6f7;
      border: 1px solid #f0f6f7;
      color: #082F49;
      border-radius: 12px;
      padding: 14px 16px;
      margin: 12px 0;
      box-shadow: 0 10px 28px rgba(8, 47, 73, 0.16);
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Neon Green Primary Colors ---
    st.markdown("""
    <style>
    /* Override Streamlit's primary color */
    :root {
        --primary-color: #00b3b3;
        --primary-color-dark: #009999;
        --secondary-background-color: #F0F2F6;
        --background-color: #FFFFFF;
        --text-color: #262730;
    }
    
    /* Primary buttons and interactive elements */
    .stButton > button {
        background: linear-gradient(90deg, #39FF14, #2ECC11) !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 12px rgba(57, 255, 20, 0.25) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #2ECC11, #28B50F) !important;
        box-shadow: 0 6px 16px rgba(57, 255, 20, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Links and interactive text */
    a, .stMarkdown a {
        color: #39FF14 !important;
    }
    a:hover, .stMarkdown a:hover {
        color: #2ECC11 !important;
    }
    
    /* Slider, selectbox, and other input accents */
    .stSlider > div > div > div > div {
        background-color: #39FF14 !important;
    }
    
    /* Checkbox and radio button accents */
    .stCheckbox > label > span[data-baseweb="checkbox"] > div,
    .stRadio > label > span[data-baseweb="radio"] > div {
        background-color: #39FF14 !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background-color: #39FF14 !important;
    }
    
    /* Tabs active state */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #39FF14 !important;
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Gradient button (as before) ---
    st.markdown("""
    <style>
    .stButton > button {
        background: linear-gradient(90deg, #6366F1, #3B82F6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 15px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #4F46E5, #2563EB);
        box-shadow: 0 6px 16px rgba(0,0,0,0.25);
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Background & sidebar (frosted glass aesthetic) ---
    st.markdown("""
    <style>
/* Global Calibri font */
* {
  font-family: 'Calibri', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
}

/* 🌅 Light blue-green gradient background */
.stApp {
  background:
    radial-gradient(50rem 35rem at 10% 20%, rgba(90, 180, 255, 0.12) 0%, transparent 70%),
    radial-gradient(45rem 30rem at 85% 15%, rgba(100, 200, 150, 0.15) 0%, transparent 70%),
    radial-gradient(55rem 40rem at 50% 80%, rgba(70, 160, 200, 0.10) 0%, transparent 75%),
    radial-gradient(40rem 35rem at 25% 60%, rgba(120, 220, 180, 0.08) 0%, transparent 80%),
    radial-gradient(60rem 25rem at 75% 40%, rgba(80, 170, 220, 0.12) 0%, transparent 75%),
    linear-gradient(120deg, #fafcff 0%, #f5faff 50%, #f8fbff 100%);
  background-attachment: fixed;
  color: #1a1a1a;
  font-family: 'Calibri', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
}

    /* 🧊 Enhanced frosted glass sidebar */
    [data-testid="stSidebar"] > div:first-child {
      background: rgba(255, 255, 255, 0.15);
      backdrop-filter: blur(20px) saturate(180%);
      -webkit-backdrop-filter: blur(20px) saturate(180%);
      border-right: 1px solid rgba(255, 255, 255, 0.2);
      border-left: 1px solid rgba(255, 255, 255, 0.1);
      box-shadow: 
        0 8px 32px rgba(150, 199, 182, 0.1),
        inset 0 1px 0 rgba(255, 255, 255, 0.4),
        inset 0 -1px 0 rgba(255, 255, 255, 0.1);
    }

    /* 🪟 Enhanced frosted glass containers */
    .stContainer, [data-testid="stExpander"] {
      background: rgba(255, 255, 255, 0.1) !important;
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: 20px;
      padding: 24px;
      box-shadow: 
        0 8px 32px rgba(150, 199, 182, 0.15),
        inset 0 1px 0 rgba(255, 255, 255, 0.3),
        inset 0 -1px 0 rgba(255, 255, 255, 0.1);
      backdrop-filter: blur(16px) saturate(180%);
      -webkit-backdrop-filter: blur(16px) saturate(180%);
      transition: all 0.3s ease;
    }

    /* Solid white forms with shadows */
    [data-testid="stForm"] {
      background: rgba(255, 255, 255, 0.95) !important;
      border: 1px solid rgba(255, 255, 255, 0.3);
      border-radius: 20px;
      padding: 24px;
      box-shadow: 
        0 8px 32px rgba(150, 199, 182, 0.15),
        0 4px 16px rgba(150, 199, 182, 0.1),
        inset 0 1px 0 rgba(255, 255, 255, 0.4);
      transition: all 0.3s ease;
    }

    /* Enhanced glass cards */
    .glass {
      background: rgba(255, 255, 255, 0.12);
      border: 1px solid rgba(255, 255, 255, 0.25);
      border-radius: 20px;
      padding: 24px;
      box-shadow: 
        0 8px 32px rgba(150, 199, 182, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.4),
        inset 0 -1px 0 rgba(255, 255, 255, 0.1);
      backdrop-filter: blur(18px) saturate(180%);
      -webkit-backdrop-filter: blur(18px) saturate(180%);
      transition: all 0.3s ease;
    }

    /* Enhanced hover effects */
    .glass:hover, .stContainer:hover {
      transform: translateY(-2px);
      box-shadow: 
        0 12px 40px rgba(150, 199, 182, 0.25),
        inset 0 1px 0 rgba(255, 255, 255, 0.5),
        inset 0 -1px 0 rgba(255, 255, 255, 0.2);
      border-color: rgba(255, 255, 255, 0.35);
    }

    /* Solid white form hover effects */
    [data-testid="stForm"]:hover {
      transform: translateY(-2px);
      box-shadow: 
        0 12px 40px rgba(150, 199, 182, 0.25),
        0 6px 20px rgba(150, 199, 182, 0.15),
        inset 0 1px 0 rgba(255, 255, 255, 0.6);
      border-color: rgba(255, 255, 255, 0.4);
    }

    /* Frosted glass for input elements */
    .stTextInput > div > div > input, .stSelectbox > div > div > div, 
    .stDateInput > div > div > input, .stTextArea > div > div > textarea {
      background: rgba(255, 255, 255, 0.08) !important;
      border: 1px solid rgba(255, 255, 255, 0.2) !important;
      border-radius: 12px !important;
      backdrop-filter: blur(10px) !important;
      -webkit-backdrop-filter: blur(10px) !important;
      color: #1a1a1a !important;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }

    /* Frosted glass for buttons */
    .stButton > button {
      background: rgba(255, 255, 255, 0.3) !important;
      border: 1px solid rgba(255, 255, 255, 0.4) !important;
      border-radius: 12px !important;
      backdrop-filter: blur(12px) saturate(180%) !important;
      -webkit-backdrop-filter: blur(12px) saturate(180%) !important;
      color: #1a1a1a !important;
      font-weight: 600 !important;
      box-shadow: 
        0 4px 16px rgba(150, 199, 182, 0.2),
        0 2px 8px rgba(150, 199, 182, 0.1),
        inset 0 1px 0 rgba(255, 255, 255, 0.5) !important;
      transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
      background: rgba(255, 255, 255, 0.5) !important;
      transform: translateY(-2px) !important;
      box-shadow: 
        0 8px 24px rgba(150, 199, 182, 0.25),
        0 4px 12px rgba(150, 199, 182, 0.15),
        inset 0 1px 0 rgba(255, 255, 255, 0.7) !important;
    }

    /* Enhanced frosted dataframes/tables */
    [data-testid="stDataFrame"], .stTable {
      background: rgba(255, 255, 255, 0.08) !important;
      border: 1px solid rgba(255, 255, 255, 0.2) !important;
      border-radius: 16px !important;
      backdrop-filter: blur(14px) saturate(180%) !important;
      -webkit-backdrop-filter: blur(14px) saturate(180%) !important;
      padding: 12px !important;
      box-shadow: 
        0 8px 24px rgba(150, 199, 182, 0.15),
        inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)


def show_callout(message: str):
    """Render a message inside a styled callout box."""
    st.markdown(f"<div class='callout'>{message}</div>", unsafe_allow_html=True)


def material_icon(icon_name: str, size: int = 24, color: str = "#96C7B6") -> str:
    """Generate HTML for a Material Symbol icon."""
    return f'<span class="material-symbol" style="font-size: {size}px; color: {color};">{icon_name}</span>'


def show_icon_text(icon_name: str, text: str, size: int = 24, color: str = "#96C7B6"):
    """Display an icon with text using Material Symbols."""
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin: 8px 0;">
        {material_icon(icon_name, size, color)}
        <span style="font-family: 'Calibri', sans-serif; color: #1a1a1a;">{text}</span>
    </div>
    """, unsafe_allow_html=True)

