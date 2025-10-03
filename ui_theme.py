# ui_theme.py
from __future__ import annotations
from pathlib import Path
import streamlit as st
from base64 import b64encode
from mimetypes import guess_type

def inject_theme() -> None:
    # Klein beetje tegen font-flicker
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    /* ===== FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    /* ===== LAYOUT ===== */
    [data-testid="stAppViewContainer"] { background:#FFFFFF; }
    [data-testid="stSidebar"]         { background:#F5F7FB; border-right:1px solid #E6E7E8; }

    /* ===== SIDEBAR TYPE ===== */
    [data-testid="stSidebar"] *{
      font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
      font-size:15px; font-weight:600; letter-spacing:0; line-height:1.2em; color:#6E6B6B;
      -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
    }

    /* ===== BODY TYPE ===== */
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"],
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stAppViewContainer"] .stText,
    [data-testid="stAppViewContainer"] .stCaption,
    [data-testid="stAppViewContainer"] .stAlert{
      font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,sans-serif !important;
      font-size:15px !important; font-weight:400 !important; letter-spacing:0 !important;
      color:#6E6B6B !important;
    }
    [data-testid="stAppViewContainer"] h1{     
    font-size: 3rem !important;   /* standaard ±2rem */
    font-weight: 800;
}
    [data-testid="stAppViewContainer"] h2{     
    font-size: 1.5rem !important;   
    font-weight: 600;
}
    [data-testid="stAppViewContainer"] h3{
      font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,sans-serif !important;
      font-size:15px !important; font-weight:600 !important; letter-spacing:0 !important;
      line-height:1.2em !important; color:#6E6B6B !important; margin:.25rem 0 .5rem 0 !important;
    }

    /* ===== BODY BUTTONS ===== */
    :root{ --btn-bg:#333232; --btn-fg:#FFFFFF; --btn-bg-hover:#4A4949; --btn-bg-active:#2A2A2A; }
    [data-testid="stAppViewContainer"] .stButton > button,
    [data-testid="stAppViewContainer"] .stDownloadButton > button,
    [data-testid="stAppViewContainer"] [data-testid="stFormSubmitButton"] button,
    [data-testid="stAppViewContainer"] [data-testid^="baseButton"] button{
      background-color:var(--btn-bg) !important; color:var(--btn-fg) !important;
      border:1px solid var(--btn-bg) !important; border-radius:10px !important;
      font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,sans-serif !important;
    }
    [data-testid="stAppViewContainer"] .stButton > button :is(span,p,div,svg,path),
    [data-testid="stAppViewContainer"] .stDownloadButton > button :is(span,p,div,svg,path),
    [data-testid="stAppViewContainer"] [data-testid="stFormSubmitButton"] button :is(span,p,div,svg,path),
    [data-testid="stAppViewContainer"] [data-testid^="baseButton"] button :is(span,p,div,svg,path){
      color:var(--btn-fg) !important; fill:var(--btn-fg) !important; stroke:var(--btn-fg) !important;
      -webkit-text-fill-color:var(--btn-fg) !important; opacity:1 !important;
    }
    [data-testid="stAppViewContainer"] .stButton > button:hover,
    [data-testid="stAppViewContainer"] .stDownloadButton > button:hover,
    [data-testid="stAppViewContainer"] [data-testid="stFormSubmitButton"] button:hover,
    [data-testid="stAppViewContainer"] [data-testid^="baseButton"] button:hover{
      background-color:var(--btn-bg-hover) !important; border-color:var(--btn-bg) !important;
    }
    [data-testid="stAppViewContainer"] .stButton > button:active,
    [data-testid="stAppViewContainer"] .stDownloadButton > button:active,
    [data-testid="stAppViewContainer"] [data-testid="stFormSubmitButton"] button:active,
    [data-testid="stAppViewContainer"] [data-testid^="baseButton"] button:active{
      background-color:var(--btn-bg-active) !important; border-color:var(--btn-bg-active) !important;
    }
    [data-testid="stAppViewContainer"] .stButton > button:focus,
    [data-testid="stAppViewContainer"] .stDownloadButton > button:focus,
    [data-testid="stAppViewContainer"] [data-testid="stFormSubmitButton"] button:focus,
    [data-testid="stAppViewContainer"] [data-testid^="baseButton"] button:focus{
      outline:none !important; box-shadow:0 0 0 2px rgba(51,50,50,0.25) !important;
    }

    /* ===== BRAND (logo links) ===== */
    [data-testid="stSidebar"] .brand{
      display:flex; align-items:center; justify-content:flex-start;   /* links uitlijnen */
      padding:12px 10px 10px;
      margin:0 0 12px 0;
      border-bottom:1px solid #E6E7E8;
    }
    [data-testid="stSidebar"] .brand img{
      max-width:160px; width:100%; height:auto; display:block;
    }

    /* =========================================================
       NAV — SIMPEL (alleen tekstlinks) + ANTI-JUMP FIX
       ========================================================= */

    /* 0) Forceer de Streamlit wrappers op full width (overrulet inline width: 7.96px) */
    [data-testid="stSidebar"] .nav-zone [data-testid="stPageLink"]{ width:100% !important; display:block !important; }
    [data-testid="stSidebar"] .nav-zone [data-testid="element-container"]{ width:100% !important; }

    /* 1) De eigenlijke link als strakke “row” */
    [data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"]{
      display:flex !important; align-items:center !important;
      width:100% !important; padding:6px 8px !important; border-radius:8px !important;
      gap:8px !important; text-decoration:none !important;
      background:transparent !important; border:0 !important; box-shadow:none !important; outline:none !important;
      margin:0 !important; line-height:1.05 !important;
      /* lock typografie in alle states → geen “vet” flicker */
      font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,sans-serif !important;
      font-weight:600 !important; letter-spacing:0 !important; color:#6E6B6B !important;
      transition:none !important; -webkit-tap-highlight-color: transparent;
    }

    /* 2) Reset ALLE binnenwrappers (span/div/markdown) zodat niets layout wijzigt */
    [data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"] > span,
    [data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"] > span > div,
    [data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"] *[data-testid="stMarkdownContainer"]{
      display:block !important; padding:0 !important; margin:0 !important; background:transparent !important; box-shadow:none !important; border:0 !important;
    }
    [data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"] *[data-testid="stMarkdownContainer"] p{
      margin:0 !important; line-height:1.05 !important; font-weight:600 !important; letter-spacing:0 !important;
    }

    /* 3) Exact dezelfde typografie in ALLE anchor states + alleen kleur veranderen */
    [data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"]:link,
    [data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"]:visited,
    [data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"]:hover,
    [data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"]:active,
    [data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"][aria-current="page"]{
      font-weight:600 !important; letter-spacing:0 !important; line-height:1.05 !important;
      color:var(--nav-fg, #6E6B6B) !important; background:transparent !important; box-shadow:none !important; outline:none !important; border:0 !important;
    }
    [data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"]:hover,
    [data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"][aria-current="page"]{
      --nav-fg:#333232;
    }

    /* 4) Focus (keyboard) zonder layout-shift */
    [data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"]:is(:focus, :focus-visible){
      outline:none !important; box-shadow:inset 0 0 0 2px rgba(51,50,50,.18) !important;
    }

    /* ===== TOGGLE (collapse/expand): neutraal houden) ===== */
    [data-testid="stSidebar"] button[aria-label*="sidebar" i],
    [data-testid="stAppViewContainer"] button[aria-label*="sidebar" i],
    [data-testid="stSidebar"] button[title*="sidebar" i],
    [data-testid="stAppViewContainer"] button[title*="sidebar" i]{
      background:transparent !important; border:0 !important; box-shadow:none !important;
      display:inline-flex !important; width:auto !important; padding:4px !important; margin:0 !important; min-height:0 !important;
    }
    button[aria-label*="sidebar" i]:hover, button[title*="sidebar" i]:hover{
      background:transparent !important; filter:brightness(0.92);
    }
    /* === ANTI-JUMP (hard) — paste at the very bottom of <style> === */

/* Force the whole st.page_link chain to 100% width, beating inline width */
[data-testid="stSidebar"] .nav-zone [data-testid="element-container"],
[data-testid="stSidebar"] .nav-zone [data-testid="stPageLink"],
[data-testid="stSidebar"] .nav-zone [data-testid="stPageLink"] > div,
[data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"]{
  width: 100% !important;
  min-width: 100% !important;
  max-width: 100% !important;
  display: block !important;
  box-sizing: border-box !important;
}

/* Make the link itself a stable row */
[data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"]{
  display: flex !important;
  align-items: center !important;
  gap: 0 !important;
  padding: 6px 8px !important;
  margin: 0 !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  outline: none !important;
}

/* Nuke any spacing the inner wrappers add */
[data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"] > span,
[data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"] > span > div,
[data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"] [data-testid="stMarkdownContainer"]{
  margin: 0 !important;
  padding: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  border: 0 !important;
}
[data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"] [data-testid="stMarkdownContainer"] p{
  margin: 0 !important;
  line-height: 1.05 !important;
  font-weight: 600 !important;
  letter-spacing: 0 !important;
}

/* Lock identical typography in all states so nothing “fattens” on click */
[data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"]:link,
[data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"]:visited,
[data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"]:hover,
[data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"]:active,
[data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"][aria-current="page"]{
  font: 600 15px/1.05 Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif !important;
  color: var(--nav-fg, #6E6B6B) !important;
  letter-spacing: 0 !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  outline: none !important;
}

/* Only change color on hover/active, never box-model */
[data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"]:hover,
[data-testid="stSidebar"] .nav-zone a[data-testid="stPageLink-NavLink"][aria-current="page"]{
  --nav-fg: #333232;
}

    </style>
    """, unsafe_allow_html=True)

def render_brand(logo_path: str = "assets/img/The_Studio_Logo.png", max_width: int = 160) -> None:
    p = Path(logo_path)
    if not p.exists():
        with st.sidebar:
            st.markdown('<div class="brand"><strong>The Studio</strong></div>', unsafe_allow_html=True)
        return
    mime, _ = guess_type(str(p))
    if mime is None: mime = "image/png"
    if p.suffix.lower() == ".svg":
        html = f'<div class="brand">{p.read_text(encoding="utf-8")}</div>'
    else:
        b64 = b64encode(p.read_bytes()).decode("ascii")
        html = f'<div class="brand"><img src="data:{mime};base64,{b64}" style="max-width:{max_width}px" alt="The Studio logo"/></div>'
    with st.sidebar:
        st.markdown(html, unsafe_allow_html=True)

def render_nav_with_icons(active: str | None = None) -> None:
    """Simpel: alleen tekstlinks, stabiel."""
    home_candidates = ["Home.py", "home.py", "app.py", "main.py"]
    home_target = next((p for p in home_candidates if Path(p).exists()), "Home.py")
    with st.sidebar:
        render_brand("assets/img/The_Studio_Logo.png")
        st.markdown('<div class="nav-zone">', unsafe_allow_html=True)
        if hasattr(st, "page_link"):
            st.page_link(home_target, label="The Studio Suite")
            st.page_link("pages/1_Value_Optimizer.py", label="Value Optimizer")
            st.page_link("pages/2_Workspace_Optimizer.py", label="Workspace Optimizer")
        else:
            if st.button("The Studio Suite", key="nav_home"):
                if hasattr(st, "switch_page"): st.switch_page(home_target)
            if st.button("Value Optimizer", key="nav_quote"):
                if hasattr(st, "switch_page"): st.switch_page("pages/1_Value_Optimizer.py")
            if st.button("Workspace Optimizer", key="nav_m2"):
                if hasattr(st, "switch_page"): st.switch_page("pages/2_Workspace_Optimizer.py")
        st.markdown('</div>', unsafe_allow_html=True)
