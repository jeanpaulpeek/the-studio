# ===============================
# Home.py — The Studio Suite (Intro + Handleidingen)
# (Sidebar blijft zoals nu; styling via ui_theme.py)
# ===============================
# --- must be the very first Streamlit call on the page ---
import streamlit as st
from streamlit.errors import StreamlitAPIException

try:
    st.set_page_config(
        page_title="The Studio Suite — Home",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except StreamlitAPIException:
    # Page config is likely already set by ui_theme.py or another module.
    pass

# -------- Thema / Navigatie: respecteer bestaande ui_theme.py --------
try:
    from ui_theme import inject_theme, render_nav_with_icons
    inject_theme()
    try:
        # Als je een nav-balk hebt in ui_theme (icons/tabs), render 'm hier:
        render_nav_with_icons(active="Home")
    except Exception:
        pass
except Exception:
    # Geen fallback-CSS: we laten styling aan jouw app en sidebar met rust.
    pass

# -------- (Optioneel) Logo boven de pagina — géén sidebar-mutaties --------
LOGO_URL = ""  # bv: "https://raw.githubusercontent.com/jeanpaulpeek/quote-optimizer/refs/heads/main/The_Studio_Logo.png"
if LOGO_URL:
    st.image(LOGO_URL, width=140)


#------- Big Quote CSS ---------
st.markdown("""
<style>
.big-quote {
  font-family: Georgia, "Times New Roman", Times, serif;
  font-style: italic;
  font-size: 22px;
  line-height: 1.6;
  border-left: 6px solid #000; /* zwarte balk links */
  padding: 14px 18px;
  margin: 0 0 18px 0;
  border-radius: 4px;
}
.big-quote b { font-style: normal; }
.step-label {
  font-size: 16px; letter-spacing: .06em; text-transform: uppercase;
  font-weight:700; margin: 18px 0 6px 0;
}
.small-muted { color:#64748b; font-size:12px; }
hr.soft { border:0; border-top:1px solid #e5e7eb; margin:1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# -------- Intro / Hero Quote --------
st.markdown("""
<div class="big-quote">
  <em>"Eén suite. Twee apps. Voor meer inzicht vooraf en betere onderbouwing van elk zakelijk inrichtingsproject.”</em>
</div>
""", unsafe_allow_html=True)

# -------- Uitleg The Studio Suite --------
st.markdown("""
The Studio Suite bestaat uit twee applicaties:

- **FURNITURE VALUE OPTIMIZER**  
- **WORKSPACE OPTIMIZER**

The Studio Suite is gemaakt voor interieurprofessionals en facilitair managers die **keuzes** moeten onderbouwen: producten, materialen, vloerindeling, budgetten en alternatieven.  
Minder getover met spreadsheets, meer dialoog met je klant.  
Op deze manier houd je regie over het budget.
""")

# -------- Furniture Value Optimizer --------
st.markdown('<div class="step-label">Furniture Value Optimizer</div>', unsafe_allow_html=True)
st.markdown("""
Met deze app kan je kiezen tussen twee doelen:

- De **laagste prijs** binnen een gestelde kwaliteitseis  
- De **beste kwaliteit** binnen een gesteld budget  

De basis van de resultaten is een gestandaardiseerde prijsoverzicht in `.xlsx` format, met daarin prijs- en kwaliteitsniveau van producten.  
De app gaat uit van **5 klasseniveau’s**: van basis/instap tot en met hi-end en iconisch design.
""")

st.markdown('<div class="step-label">Gebruik preset of eigen gegevens</div>', unsafe_allow_html=True)
st.markdown("""
Je kan gebruik maken van de demo-prijslijst, en je kunt een eigen prijsoverzicht uploaden.  
Belangrijk is dat het Excel-bestand de volgende kolommen bevat:

- **Artikel** (bijv. Eames Lounge Chair of AAC-22)  
- **Soort** (bijv. sofa, vergadertafel, bureaustoel)  
- **Merk** (bijv. Vitra, Narbutas, Hay)  
- **Prijs** (= adviesprijs exclusief BTW)  
- **Klasse** (verdeling 1-5)  
- **Korting Leverancier** (= % — optioneel)  
""")

st.markdown('<div class="step-label">Producten en aantallen</div>', unsafe_allow_html=True)
st.markdown("""
De app groepeert automatisch alle producten per soort.  
In de invoer-sheet vink je aan:  

- welke productgroepen je wil meenemen in het interieurvoorstel  
- welke aantallen  
- aan welke minimum en maximum waarde voldaan moet worden (of laat leeg)  
- of er binnen een productsoort meerdere verschillende producten toegepast mogen worden (vink uit als je dat niet wil)  

Bij minimale prijs binnen een kwaliteitseis: stel met de slider in welk gewenst niveau de productselectie gemiddeld moet voldoen.  
Bij werken vanuit een budget: de app stelt automatisch een selectie voor met de hoogst haalbare gemiddelde kwaliteit.
""")

st.markdown('<div class="step-label">Optimaliseer</div>', unsafe_allow_html=True)
st.markdown("""
De **optimaliseer-functie** genereert het beste aanbod inclusief 2 alternatieven die voldoen aan het gekozen doel:  

- laagste prijs binnen kwaliteitseis  
- beste kwaliteit binnen budget  
""")

# -------- Workspace Optimizer --------
st.markdown('<div class="step-label">Workspace Optimizer</div>', unsafe_allow_html=True)
st.markdown("""
Deze app maakt bij (her)huisvesting en (her)inrichtingsplannen inzichtelijk:  

- hoeveel vloeroppervlak heb ik nodig  
- welke initiële investering moet ik doen (CapEx)  
- wat de jaarlijkse operationele facilitaire kosten zijn (OpEx)  
- wat de totale kosten over de gebruikstermijn zijn (TotEx)  

Je kan hiervoor gebruikmaken van de **NFC index** of van eigen gegevens en uitgangspunten.
""")

st.markdown("""
Het gebruik van de app werkt intuïtief en geeft als resultaat een realistische indicatie van investeringen en budgetten.
""")
