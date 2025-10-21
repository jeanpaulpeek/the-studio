# ============================================
# pages/1_Value_Optimizer.py — G/B/B + vergelijking + export
# ============================================

from __future__ import annotations
import io
import math
import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import pulp
from streamlit.errors import StreamlitAPIException

# -- Veilige page_config (faalt stil als er al een pagina is geconfigureerd)
try:
    st.set_page_config(
        page_title="The Studio Suite — Value Optimizer",
        page_icon="🎛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except StreamlitAPIException:
    pass

# -- (Optioneel) thema / navigatie; faalt stil als ui_theme niet bestaat
try:
    from ui_theme import inject_theme, render_nav_with_icons
except Exception:  # pragma: no cover
    def inject_theme(): ...
    def render_nav_with_icons(*args, **kwargs): ...

# ==== PDF HELPERS ============================================================
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader, PdfWriter
try:
    from pypdf import Transformation  # pypdf >= 3.x
except Exception:
    Transformation = None

from pulp import HiGHS_CMD, PULP_CBC_CMD

def get_solver():
    try:
        return HiGHS_CMD(msg=False)
    except Exception:
        return PULP_CBC_CMD(msg=False)

PAGE_SIZE = A4
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 20*mm, 20*mm, 100*mm, 18*mm

def _on_page(letterhead_img_reader=None, footer_text=""):
    from reportlab.pdfgen import canvas as rl_canvas
    def _cb(c: rl_canvas.Canvas, _doc):
        if letterhead_img_reader is not None:
            w, h = PAGE_SIZE
            c.drawImage(letterhead_img_reader, 0, 0, width=w, height=h, preserveAspectRatio=False, mask="auto")
        c.setFont("Helvetica", 8)
        c.setFillColorRGB(0.45, 0.48, 0.55)
        c.drawRightString(PAGE_SIZE[0]-MARGIN_R, 10*mm, f"{footer_text}  •  pagina {c.getPageNumber()}")
    return _cb

def build_offer_pdf(
    project_meta: dict,
    result_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    show_prices: bool = True,
    letterhead_image_bytes: bytes | None = None,
    internal: bool = False,
) -> bytes:
    buf = io.BytesIO()
    top_margin_value = 150 if letterhead_image_bytes else MARGIN_T

    doc = SimpleDocTemplate(
        buf, pagesize=PAGE_SIZE,
        leftMargin=MARGIN_L, rightMargin=MARGIN_R,
        topMargin=top_margin_value, bottomMargin=MARGIN_B
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, leading=22, spaceAfter=6))
    styles.add(ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, leading=16, spaceAfter=4))
    styles.add(ParagraphStyle("Meta", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#475569")))
    styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14))

    story = []
    titel = project_meta.get("title") or ("Rapport (intern)" if internal else "Offerte")
    story.append(Paragraph(titel, styles["H1"]))

    bits = []
    for key, label in [
        ("client", "Klant"),
        ("reference", "Referentie"),
        ("date_str", "Datum"),
        ("mode_str", "Doel"),
        ("discount_client_str", "Klantkorting"),
        ("budget_str", "Budget"),
    ]:
        val = project_meta.get(key)
        if val:
            bits.append(f"<b>{label}</b>: {val}")
    if bits:
        story.append(Paragraph(" &nbsp; • &nbsp; ".join(bits), styles["Meta"]))
    story.append(Spacer(1, 10))

    # Samenvatting per soort
    story.append(Paragraph("Samenvatting per soort", styles["H2"]))
    sum_df = summary_df.copy()
    if internal:
        sum_cols = ["soort","totaal_stuks","omzet","inkoop","marge","marge_pct","gem_klasse_per_soort"]
        sum_headers = ["Soort","Stuks","Omzet","Inkoop","Marge","Marge %","Gem. klasse"]
    else:
        sum_cols = ["soort","totaal_stuks","omzet","gem_klasse_per_soort"]
        sum_headers = ["Soort","Stuks","Omzet","Gem. klasse"]

    for c in sum_cols:
        if c not in sum_df.columns:
            sum_df[c] = ""
    sum_tbl = [sum_headers] + sum_df[sum_cols].astype(str).values.tolist()
    t = Table(sum_tbl, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONT",(0,0),(-1,0),"Helvetica-Bold",10),
        ("FONT",(0,1),(-1,-1),"Helvetica",9),
        ("TEXTCOLOR",(0,0),(-1,0),colors.HexColor("#0F172A")),
        ("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#E2E8F0")),
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#F1F5F9")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#FCFDFF")]),
        ("ALIGN",(1,1),(-1,-1),"RIGHT"),
        ("ALIGN",(0,0),(0,-1),"LEFT"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("TOPPADDING",(0,0),(-1,-1),4),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Details
    story.append(Paragraph("Details", styles["H2"]))
    det = result_df.copy()

    def _eur(v):
        try:
            return f"€ {float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return str(v)

    if "Brutoprijs" not in det.columns and "prijs" in det.columns:
        det["Brutoprijs"] = det["prijs"].apply(_eur)
    if "Verkoop/klant" not in det.columns and "price_effective" in det.columns:
        det["Verkoop/klant"] = det["price_effective"].apply(_eur)
    if "Inkoop/netto" not in det.columns and "price_buy_unit" in det.columns:
        det["Inkoop/netto"] = det["price_buy_unit"].apply(_eur)
    if "Subtotaal klant" not in det.columns and {"aantal","price_effective"}.issubset(det.columns):
        det["Subtotaal klant"] = (det["aantal"]*det["price_effective"]).apply(_eur)
    if "Subtotaal inkoop" not in det.columns and {"aantal","price_buy_unit"}.issubset(det.columns):
        det["Subtotaal inkoop"] = (det["aantal"]*det["price_buy_unit"]).apply(_eur)

    if internal:
        det_cols = ["soort","artikel","merk","klasse","aantal","Brutoprijs","Verkoop/klant","Inkoop/netto","Subtotaal klant","Subtotaal inkoop"]
        det_headers = ["Soort","Artikel","Merk","Kl.","Aantal","Bruto","Verkoop","Inkoop","Subtot. klant","Subtot. inkoop"]
    else:
        det_cols = ["soort","artikel","merk","klasse","aantal","Verkoop/klant","Subtotaal klant"]
        det_headers = ["Soort","Artikel","Merk","Kl.","Aantal","Prijs/stuk","Subtotaal"]

    for c in det_cols:
        if c not in det.columns:
            det[c] = ""
    det_tbl = [det_headers] + det[det_cols].astype(str).values.tolist()
    td = Table(det_tbl, hAlign="LEFT", repeatRows=1)
    td.setStyle(TableStyle([
        ("FONT",(0,0),(-1,0),"Helvetica-Bold",9),
        ("FONT",(0,1),(-1,-1),"Helvetica",8),
        ("ALIGN",(len(det_headers)-2,1),(-1,-1),"RIGHT"),
        ("ALIGN",(0,0),(2,-1),"LEFT"),
        ("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#E2E8F0")),
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#F1F5F9")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#FCFDFF")]),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),3),
    ]))
    story.append(td)

    footer = project_meta.get("footer", "© jouw-merk – gegenereerd met Value Optimizer")
    img_reader = ImageReader(io.BytesIO(letterhead_image_bytes)) if letterhead_image_bytes else None
    doc.build(story, onFirstPage=_on_page(img_reader, footer), onLaterPages=_on_page(img_reader, footer))
    return buf.getvalue()

def overlay_pdf_letterhead(base_pdf_bytes: bytes, letterhead_pdf_bytes: bytes, first_page_only=True) -> bytes:
    base_reader = PdfReader(io.BytesIO(base_pdf_bytes))
    letter_reader = PdfReader(io.BytesIO(letterhead_pdf_bytes))
    out = PdfWriter()

    overlay_page0 = letter_reader.pages[0]
    ow, oh = float(overlay_page0.mediabox.width), float(overlay_page0.mediabox.height)

    for i, page in enumerate(base_reader.pages):
        bw, bh = float(page.mediabox.width), float(page.mediabox.height)
        new_page = page
        if (not first_page_only) or (i == 0):
            if Transformation is not None:
                sx, sy = (bw / ow if ow else 1.0), (bh / oh if oh else 1.0)
                overlay_page = PdfReader(io.BytesIO(letterhead_pdf_bytes)).pages[0]
                new_page.merge_transformed_page(overlay_page, Transformation().scale(sx, sy))
            else:
                overlay_page = PdfReader(io.BytesIO(letterhead_pdf_bytes)).pages[0]
                new_page.merge_page(overlay_page)
        out.add_page(new_page)

    out_buf = io.BytesIO()
    out.write(out_buf)
    return out_buf.getvalue()

# ==== Thema/Navigatie =========================================================
inject_theme()
render_nav_with_icons()

# ==== EU-notatie helpers ======================================================
try:
    from babel.numbers import format_currency
    def euro(x, nd=0):
        return format_currency(round(float(x or 0), nd), "EUR", locale="nl_NL")
except Exception:
    def _swap(s): return s.replace(",", "§").replace(".", ",").replace("§", ".")
    def euro(x, nd=0):
        try: v = float(x or 0)
        except Exception: v = 0.0
        return "€ " + _swap(f"{v:,.{nd}f}")

def int_to_eu(n: int) -> str:
    try:
        return f"{int(n):,}".replace(",", ".")
    except Exception:
        return "0"

def eu_to_int(s: str) -> int:
    if s is None: return 0
    s = str(s).strip()
    if s == "": return 0
    s = s.replace(".", "").replace(" ", "").replace(",", ".")
    try:
        return int(float(s))
    except Exception:
        return 0

def pct_to_float(v):
    if v is None or (isinstance(v, float) and pd.isna(v)): return 0.0
    s = str(v).strip().replace("%", "").replace(",", ".")
    try:
        val = float(s)
        return val / 100.0 if val > 1.0 else val
    except Exception:
        return 0.0

def format_date(d=None):
    if d is None: d = datetime.date.today()
    return d.strftime("%d-%m-%Y")

def _euro0(x):
    try:
        return f"€ {float(x):,.0f}".replace(",", ".")
    except Exception:
        return "€ 0"

# ==== Demo/Upload helpers =====================================================
DEMO_XLSX_PATH = Path(__file__).with_name("Demo_prijslijst.xlsx")

def _normalize_pricelist(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()
    needed = {"artikel", "soort", "merk", "prijs", "klasse"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError("Ontbrekende kolommen: " + ", ".join(sorted(missing)))
    df["prijs"] = pd.to_numeric(df["prijs"], errors="coerce")
    df["klasse"] = pd.to_numeric(df["klasse"], errors="coerce")
    if "korting_leverancier" in df.columns:
        df["korting_leverancier"] = pd.to_numeric(df["korting_leverancier"], errors="coerce").fillna(0.0)
        if df["korting_leverancier"].dropna().gt(1).any():
            df["korting_leverancier"] /= 100.0
    else:
        df["korting_leverancier"] = 0.0
    return df

@st.cache_data(show_spinner=False)
def _load_demo_df(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_excel(path)
    else:
        # compacte fallback demo
        data = [
            ["Bureau 160","Bureau","BrandA",550,2.0,0.10],
            ["Bureau 180","Bureau","BrandB",690,3.0,0.15],
            ["Ergostoel","Stoel","BrandA",420,3.5,0.05],
            ["Stoel Basic","Stoel","BrandC",220,2.0,0.00],
            ["Kast 2-deurs","Opbergen","BrandD",380,2.5,0.12],
            ["Kast hoog","Opbergen","BrandD",520,3.0,0.12],
            ["Lamp vloei","Verlichting","BrandE",140,1.5,0.00],
            ["Lamp taak","Verlichting","BrandE",240,2.5,0.00],
            ["Tafel 4p","Vergader","BrandF",980,3.2,0.10],
            ["Tafel 8p","Vergader","BrandF",1650,4.0,0.10],
        ]
        df = pd.DataFrame(data, columns=["artikel","soort","merk","prijs","klasse","korting_leverancier"])
    return _normalize_pricelist(df)

# ==== UI: Intro & stijl =======================================================
st.markdown("""
<style>
.big-quote { font-family: Georgia, "Times New Roman", Times, serif;
  font-style: italic; font-size: 22px; line-height: 1.6;
  border-left: 6px solid #000; padding: 14px 18px; margin: 0 0 18px 0; border-radius: 4px; }
.big-quote b { font-style: normal; }
.step-label { font-size: 13px; letter-spacing: .06em; text-transform: uppercase;
  color:#0f172a; font-weight:700; margin: 18px 0 6px 0; }
.callout { background:#f5f5f5; border:1px solid #e5e7eb; border-radius:6px;
  padding:12px 14px; margin:10px 0 6px 0; color:#334155; }
.callout b { color:#0f172a; }
hr.soft { border:0; border-top:1px solid #e5e7eb; margin:1.2rem 0; }
.card h4 { margin:0 0 4px 0; font-size:15px; letter-spacing:.02em }
.card { background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:12px 14px;
  min-height: 170px; display:flex; flex-direction:column; gap:6px; }
.card-foot { margin-top:auto; color:#64748b; font-size:12px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="big-quote">
  "<b><i>Furniture Value Optimizer</i></b> bouwt automatisch de beste productmix:
  óf de hoogste kwaliteit binnen je budget, óf de laagste prijs die aan je
  kwaliteitsdrempel voldoet."
</div>
<hr class="soft">
""", unsafe_allow_html=True)

# ==== STAP 1: bron ============================================================
st.markdown('<div class="step-label">STAP 1. KIES PRIJSLIJST</div>', unsafe_allow_html=True)
source_choice = st.radio("Bron", ["Demo-prijslijst gebruiken", "Upload eigen prijslijst"], index=0, horizontal=True)

if source_choice == "Upload eigen prijslijst":
    st.markdown("""
    <style>@media (max-width: 800px){ .callout.two-col{ grid-template-columns: 1fr !important; }}</style>
    <div class="callout two-col" style="display:grid; grid-template-columns: 1.4fr 1fr; gap:12px; align-items:start;">
      <div>
        <b>Formaat prijslijst (.xlsx): gebruik deze kolomnamen</b><br>
        • <b>artikel</b> · <b>soort</b> · <b>merk</b> · <b>prijs</b> · <b>klasse</b><br>
        • <b>korting_leverancier</b> = % of fractie (optioneel)
      </div>
      <div>
        <b>Uitleg klassen (1–5)</b>
        <ul style="margin:6px 0 0 18px;">
          <li><b>1</b> – Instap / budget</li>
          <li><b>2</b> – Degelijk</li>
          <li><b>3</b> – Comfortabel / merk</li>
          <li><b>4</b> – Design / premium</li>
          <li><b>5</b> – Topsegment</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload prijslijst (.xlsx)", type=["xlsx"])
    if not uploaded:
        st.stop()
    try:
        df_raw = pd.read_excel(uploaded)
        df_raw = _normalize_pricelist(df_raw)
    except Exception as e:
        st.error(f"Kon Excel niet verwerken: {e}")
        st.stop()
else:
    try:
        df_raw = _load_demo_df(DEMO_XLSX_PATH)
    except Exception as e:
        st.error(f"Kon demo-dataset niet laden: {e}")
        st.stop()

# ==== STAP 2: doel ============================================================
st.markdown('<div class="step-label">STAP 2. KIES OPTIMALISATIE-DOEL</div>', unsafe_allow_html=True)
DOEL_KLASSE = "Minimale prijs bij de gekozen kwaliteitsdrempel"
DOEL_BUDGET = "Maximale kwaliteit binnen budget"

doel = st.radio(" ", [DOEL_KLASSE, DOEL_BUDGET], index=0, label_visibility="collapsed")
st.session_state["opt_goal"] = ("BUDGET" if doel == DOEL_BUDGET else "KLASSE")

if doel == DOEL_KLASSE:
    st.caption("Je stelt een kwaliteitsdrempel; de optimizer zoekt de laagste prijs die daaraan voldoet.")
else:
    st.caption("Je geeft een budget; de optimizer maximaliseert de gemiddelde klasse binnen dat budget.")

# ==== STAP 3: korting/budget ==================================================
st.markdown('<div class="step-label">STAP 3. KLANTKORTING EN (EVENTUEEL) BUDGET</div>', unsafe_allow_html=True)
colK, colB = st.columns([1, 2])

with colK:
    klant_korting_pct = st.number_input("Klantkorting (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)

def _format_budget_cb():
    raw = st.session_state.get("budget_input_str", "")
    n = eu_to_int(raw)
    st.session_state["budget_input_str"] = int_to_eu(n)
    st.session_state["budget_value"] = n

budget_net = None
if doel == DOEL_BUDGET:
    if "budget_input_str" not in st.session_state:
        st.session_state["budget_input_str"] = int_to_eu(100_000)
        st.session_state["budget_value"] = 100_000
    with colB:
        st.text_input(
            "Budget (netto, na klantkorting)",
            key="budget_input_str",
            on_change=_format_budget_cb,
            help="Gebruik punten voor duizendtallen, bijv. 100.000",
        )
        budget_net = st.session_state.get("budget_value", 100_000)

st.markdown('<hr class="soft">', unsafe_allow_html=True)

# ==== STAP 4: invoer per soort ===============================================
st.markdown('<div class="step-label">STAP 4. INVULLEN PER SOORT (AANTAL / KLASSEN / NIET MIXEN)</div>', unsafe_allow_html=True)

df_work = df_raw.copy()
df_work["korting_leverancier"] = df_work.get("korting_leverancier", 0.0).apply(pct_to_float)
klant_korting = pct_to_float(klant_korting_pct)
df_work["price_buy_unit"] = df_work["prijs"] * (1 - df_work["korting_leverancier"])
df_work["price_effective"] = df_work["prijs"] * (1 - klant_korting)

types = sorted(df_work["soort"].dropna().astype(str).unique().tolist())
init_rows = [{"soort": s, "actief": False, "aantal": 0, "min_klasse": None, "max_klasse": None, "niet_mixen": False} for s in types]

types_key = tuple(types)
if "editor_types" not in st.session_state or st.session_state["editor_types"] != types_key:
    st.session_state["editor_types"] = types_key
    st.session_state["editor_df"] = pd.DataFrame(init_rows)

with st.form("opt_form", clear_on_submit=False):
    edited = st.data_editor(
        st.session_state["editor_df"],
        key="editor_df_widget",
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "soort": st.column_config.TextColumn("SOORT", disabled=True),
            "actief": st.column_config.CheckboxColumn("ACTIEF"),
            "aantal": st.column_config.NumberColumn("AANTAL", min_value=None, step=1),
            "min_klasse": st.column_config.NumberColumn("MIN. KLASSE", min_value=None, step=0.1),
            "max_klasse": st.column_config.NumberColumn("MAX. KLASSE (0=GEEN)", min_value=None, step=0.1),
            "niet_mixen": st.column_config.CheckboxColumn("NIET MIXEN"),
        },
    )
    target_avg = None
    if doel == DOEL_KLASSE:
        st.markdown("**DOELKLASSE (gemiddeld)**")
        target_avg = st.slider(" ", min_value=1.0, max_value=5.0, value=2.7, step=0.05, label_visibility="collapsed")
    submitted = st.form_submit_button("OPTIMALISEREN", type="primary")

# ==== Solver helpers ==========================================================
def _normalize_editor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for k, v in {"soort":"","actief":False,"aantal":0,"min_klasse":0.0,"max_klasse":0.0,"niet_mixen":False}.items():
        if k not in df.columns: df[k] = v
    df["soort"] = df["soort"].astype(str)
    df["actief"] = df["actief"].astype(bool)
    for c in ["aantal","min_klasse","max_klasse"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["niet_mixen"] = df["niet_mixen"].astype(bool)
    df["actief"] = df["actief"] | (df["aantal"] > 0)
    return df

def _build_common(df, needs):
    df = df.copy()
    df.columns = df.columns.str.lower()
    price_col = "price_effective" if "price_effective" in df.columns else "prijs"
    types = sorted(df["soort"].astype(str).unique().tolist())
    idx_by_type = {s: [i for i in range(len(df)) if str(df.loc[i,"soort"]) == s] for s in types}
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in range(len(df))]
    y = [None] * len(df)
    price_sum = pulp.lpSum(x[i] * float(df.loc[i, price_col]) for i in range(len(df)))
    class_sum = pulp.lpSum(x[i] * float(df.loc[i, "klasse"]) for i in range(len(df)))
    return df, price_col, types, idx_by_type, x, y, price_sum, class_sum

def _add_core_constraints(model, df, needs, idx_by_type, x, y, min_class_by_type, max_class_by_type, no_mix_by_type):
    for s, qty in needs.items():
        if int(qty) <= 0: continue
        idx = idx_by_type.get(s, [])
        model += pulp.lpSum(x[i] for i in idx) == int(qty)
    min_class_by_type = min_class_by_type or {}
    max_class_by_type = max_class_by_type or {}
    for s, qty in needs.items():
        if int(qty) <= 0: continue
        idx = idx_by_type.get(s, [])
        mn = float(min_class_by_type.get(s, 0.0) or 0.0)
        mx = float(max_class_by_type.get(s, 0.0) or 0.0)
        if mn > 0:
            model += pulp.lpSum(x[i] * float(df.loc[i,"klasse"]) for i in idx) >= mn * int(qty)
        if mx > 0:
            model += pulp.lpSum(x[i] * float(df.loc[i,"klasse"]) for i in idx) <= mx * int(qty)
    no_mix_by_type = no_mix_by_type or {}
    for s, qty in needs.items():
        if int(qty) <= 0: continue
        if no_mix_by_type.get(s, False):
            idx = idx_by_type.get(s, [])
            for i in idx:
                if y[i] is None:
                    y[i] = pulp.LpVariable(f"y_{i}", lowBound=0, upBound=1, cat="Binary")
            model += pulp.lpSum(y[i] for i in idx) == 1
            for i in idx:
                model += x[i] <= int(qty) * y[i]

def solve_min_cost_with_quality(
    df, needs, min_avg, min_class_by_type=None, max_class_by_type=None, no_mix_by_type=None, budget_cap=None
):
    df = df.copy()
    df.columns = df.columns.str.lower()
    required = {"artikel","soort","merk","prijs","klasse"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Ontbrekende kolommen: {required - set(df.columns)}")
    active = [s for s, q in needs.items() if int(q) > 0]
    if not active: raise ValueError("Geen aantallen ingevuld (>0).")
    df = df[df["soort"].isin(active)].reset_index(drop=True)
    N = int(sum(needs.values()))
    df, price_col, types, idx_by_type, x, y, price_sum, class_sum = _build_common(df, needs)
    model = pulp.LpProblem("min_cost_quality", pulp.LpMinimize)
    model += price_sum
    _add_core_constraints(model, df, needs, idx_by_type, x, y, min_class_by_type, max_class_by_type, no_mix_by_type)
    if min_avg is not None and min_avg > 0:
        model += class_sum >= float(min_avg) * N
    if budget_cap is not None and budget_cap > 0:
        model += price_sum <= float(budget_cap)
    res = model.solve(get_solver())
    if pulp.LpStatus[res] != "Optimal":
        raise RuntimeError("Geen optimale oplossing (min_cost_with_quality).")
    qtys = [int(v.value()) for v in x]
    df["aantal"] = qtys
    df = df[df["aantal"] > 0].copy()
    df["omzet_subtotaal"] = df["aantal"] * (df.get("price_effective", df["prijs"]))
    df["inkoop_subtotaal"] = df["aantal"] * (df.get("price_buy_unit", 0.0))
    total_price = float(df["omzet_subtotaal"].sum())
    avg_class = float((df["aantal"] * df["klasse"]).sum() / N)
    return df, total_price, avg_class

def solve_max_quality_under_budget(df, needs, budget, min_class_by_type=None, max_class_by_type=None, no_mix_by_type=None):
    df = df.copy()
    df.columns = df.columns.str.lower()
    required = {"artikel","soort","merk","prijs","klasse"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Ontbrekende kolommen: {required - set(df.columns)}")
    active = [s for s, q in needs.items() if int(q) > 0]
    if not active: raise ValueError("Geen aantallen ingevuld (>0).")
    df = df[df["soort"].isin(active)].reset_index(drop=True)
    N = int(sum(needs.values()))
    df, price_col, types, idx_by_type, x, y, price_sum, class_sum = _build_common(df, needs)
    model = pulp.LpProblem("max_quality_budget", pulp.LpMaximize)
    model += class_sum
    _add_core_constraints(model, df, needs, idx_by_type, x, y, min_class_by_type, max_class_by_type, no_mix_by_type)
    model += price_sum <= float(budget)
    res = model.solve(get_solver())
    if pulp.LpStatus[res] != "Optimal":
        raise RuntimeError("Geen optimale oplossing (max_quality_budget).")
    qtys = [int(v.value()) for v in x]
    df["aantal"] = qtys
    df = df[df["aantal"] > 0].copy()
    df["omzet_subtotaal"] = df["aantal"] * (df.get("price_effective", df["prijs"]))
    df["inkoop_subtotaal"] = df["aantal"] * (df.get("price_buy_unit", 0.0))
    total_price = float(df["omzet_subtotaal"].sum())
    avg_class = float((df["aantal"] * df["klasse"]).sum() / N)
    return df, total_price, avg_class

# ==== Optimaliseren ===========================================================
if submitted:
    ed = _normalize_editor(edited)
    st.session_state["editor_df"] = ed.copy()

    needs = {row["soort"]: int(max(0, row["aantal"])) for _, row in ed.iterrows() if (bool(row["actief"]) or int(row.get("aantal", 0)) > 0)}
    per_type_min = {row["soort"]: float(row.get("min_klasse", 0.0) or 0.0) for _, row in ed.iterrows()}
    per_type_max = {row["soort"]: float(row.get("max_klasse", 0.0) or 0.0) for _, row in ed.iterrows()}
    no_mix_by_type = {row["soort"]: bool(row.get("niet_mixen", False)) for _, row in ed.iterrows()}

    problems = []
    if sum(needs.values()) <= 0:
        problems.append("Geen aantallen ingevuld (>0).")
    if doel == DOEL_BUDGET and (not budget_net or budget_net <= 0):
        problems.append("Geen (geldig) budget ingevuld.")
    if problems:
        st.error("Corrigeer de volgende punten:")
        for p in problems: st.write("• " + p)
        st.stop()

    try:
        options = []
        if doel == DOEL_BUDGET:
            # BEST (referentie) = max kwaliteit binnen budget
            best_df, best_total, best_avg = solve_max_quality_under_budget(
                df_work, needs, budget_net,
                min_class_by_type=per_type_min, max_class_by_type=per_type_max, no_mix_by_type=no_mix_by_type
            )
            # GOOD / ALT_A = min kost met kwaliteit-steps onder BEST (budget-cap blijft gelden)
            thr_better = max(1.0, best_avg - 0.05)
            thr_good   = max(1.0, best_avg - 0.10)

            b_df, b_total, b_avg = solve_min_cost_with_quality(
                df_work, needs, thr_better,
                min_class_by_type=per_type_min, max_class_by_type=per_type_max, no_mix_by_type=no_mix_by_type,
                budget_cap=budget_net
            )
            g_df, g_total, g_avg = solve_min_cost_with_quality(
                df_work, needs, thr_good,
                min_class_by_type=per_type_min, max_class_by_type=per_type_max, no_mix_by_type=no_mix_by_type,
                budget_cap=budget_net
            )

            options = [
                {"role":"GOOD","name":"GOOD • BUDGETVRIENDELIJK (−0,10 t.o.v. beste)",
                 "result":g_df,"total":g_total,"avg":g_avg, "delta_class": g_avg - best_avg, "delta_ref":"beste"},
                {"role":"ALT_A","name":"ALTERNATIEF A • (−0,05 t.o.v. beste)",
                 "result":b_df,"total":b_total,"avg":b_avg, "delta_class": b_avg - best_avg, "delta_ref":"beste"},
                {"role":"ALT_B","name":"BEST • MAX KWALITEIT BINNEN BUDGET",
                 "result":best_df,"total":best_total,"avg":best_avg, "delta_class": 0.0, "delta_ref":"beste"},
            ]
        else:
            # GOED = minimale prijs die aan target_avg voldoet
            g_df, g_total, g_avg = solve_min_cost_with_quality(
                df_work, needs, target_avg,
                min_class_by_type=per_type_min, max_class_by_type=per_type_max, no_mix_by_type=no_mix_by_type
            )
            # ALT_A, ALT_B = duurdere varianten met hogere klasse
            b_df, b_total, b_avg = solve_min_cost_with_quality(
                df_work, needs, target_avg + 0.05,
                min_class_by_type=per_type_min, max_class_by_type=per_type_max, no_mix_by_type=no_mix_by_type
            )
            best_df, best_total, best_avg = solve_min_cost_with_quality(
                df_work, needs, target_avg + 0.10,
                min_class_by_type=per_type_min, max_class_by_type=per_type_max, no_mix_by_type=no_mix_by_type
            )
            options = [
                {"role":"GOOD","name":"GOED • MINIMALE PRIJS (voldoet aan eis)",
                 "result":g_df,"total":g_total,"avg":g_avg, "delta_class": g_avg - float(target_avg), "delta_ref":"eis"},
                {"role":"ALT_A","name":"ALTERNATIEF A • KLASSE +0,05",
                 "result":b_df,"total":b_total,"avg":b_avg, "delta_class": b_avg - float(target_avg), "delta_ref":"eis"},
                {"role":"ALT_B","name":"ALTERNATIEF B • KLASSE +0,10",
                 "result":best_df,"total":best_total,"avg":best_avg, "delta_class": best_avg - float(target_avg), "delta_ref":"eis"},
            ]

        st.session_state["gbb_options"] = options
        st.session_state["badge"] = (
            "Kwaliteit binnen budget (" + euro(budget_net, 0) + ")"
            if doel == DOEL_BUDGET
            else f"Min. prijs bij klasse ≥ {str(target_avg).replace('.',',')}"
        )
        st.session_state["df_work"] = df_work
        st.session_state["klant_korting_value"] = float(klant_korting_pct)
        if doel == DOEL_BUDGET:
            st.session_state["last_budget_shown"] = budget_net

        st.success("Opties berekend. Zie RESULTATEN hieronder.")
        st.rerun()

    except Exception as e:
        st.error(f"Optimalisatie mislukte: {e}")

# ==== RESULTATEN ==============================================================
if "gbb_options" in st.session_state and st.session_state["gbb_options"]:
    opts = st.session_state["gbb_options"]
    st.markdown("## RESULTATEN")
    st.caption(st.session_state.get("badge", ""))

    by_role = {o["role"]: o for o in opts}
    good = by_role.get("GOOD", opts[0])
    altA = by_role.get("ALT_A")
    altB = by_role.get("ALT_B")

    base_price = good["total"]

    def _col_card(title, total, avg, diff_vs_good=None, delta_class=None, delta_ref_label="eis/beste"):
        """Render één kaart met prijs, gemiddelde klasse en Δ-kwaliteit t.o.v. referentie."""
        price_html = ""
        if diff_vs_good is not None:
            sign = "+" if diff_vs_good > 0 else ("±" if abs(diff_vs_good) < 1e-6 else "")
            price_html = f'<div class="card-foot">Prijsverschil t.o.v. GOED: {sign}{euro(diff_vs_good,0)}</div>'

        delta_html = ""
        if delta_class is not None:
            dc = f"{delta_class:+.2f}".replace(".", ",")
            delta_html = f'<div>Δ kwaliteit t.o.v. <b>{delta_ref_label}</b>: {dc}</div>'

        st.markdown(
            f"""
<div class="card">
  <h4>{title}</h4>
  <div><b>Prijs = {euro(total, 0)}</b></div>
  <div><b>Gemiddelde klasse = {str(round(avg,2)).replace('.', ',')}</b></div>
  {delta_html}
  {price_html}
</div>
""",
            unsafe_allow_html=True,
        )

    goal_mode = st.session_state.get("opt_goal", "KLASSE")  # "BUDGET" of "KLASSE"
    ref_lbl = "beste" if goal_mode == "BUDGET" else "eis"

    c1, c2, c3 = st.columns(3)
    with c1:
        _col_card(
            "GOED — MINIMALE PRIJS" if goal_mode == "KLASSE" else "GOED — BUDGETVRIENDELIJK",
            good["total"], good["avg"],
            diff_vs_good=None,
            delta_class=good.get("delta_class"),
            delta_ref_label=ref_lbl
        )
    with c2:
        if altA:
            _col_card(
                "ALTERNATIEF A",
                altA["total"], altA["avg"],
                diff_vs_good=altA["total"] - base_price,
                delta_class=altA.get("delta_class"),
                delta_ref_label=ref_lbl
            )
        else:
            st.info("Geen alternatief A beschikbaar voor deze invoer.")
    with c3:
        if altB:
            _col_card(
                "ALTERNATIEF B" if goal_mode == "KLASSE" else "BEST — MAX KWALITEIT BINNEN BUDGET",
                altB["total"], altB["avg"],
                diff_vs_good=altB["total"] - base_price,
                delta_class=altB.get("delta_class"),
                delta_ref_label=ref_lbl
            )
        else:
            st.info("Geen alternatief B beschikbaar voor deze invoer.")

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

# ==== GEKOZEN PRODUCTEN & AANTALLEN ===========================================
st.markdown("### GEKOZEN PRODUCTEN & AANTALLEN")

def _build_grp_from_selection(sel_df: pd.DataFrame) -> pd.DataFrame:
    if sel_df is None or sel_df.empty:
        return pd.DataFrame(columns=["soort","totaal_stuks","omzet","inkoop","marge","marge_pct","gem_klasse_per_soort"])
    df = sel_df.copy()
    for col in ["aantal","klasse","price_effective","price_buy_unit"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "omzet_subtotaal" not in df.columns:
        df["omzet_subtotaal"] = df["aantal"] * df.get("price_effective", 0.0)
    if "inkoop_subtotaal" not in df.columns:
        df["inkoop_subtotaal"] = df["aantal"] * df.get("price_buy_unit", 0.0)
    if "gew_klasse" not in df.columns:
        df["gew_klasse"] = df["aantal"] * df.get("klasse", 0.0)
    grp = df.groupby("soort", as_index=False).agg(
        totaal_stuks=("aantal","sum"),
        omzet=("omzet_subtotaal","sum"),
        inkoop=("inkoop_subtotaal","sum"),
        som_gew_klasse=("gew_klasse","sum"),
        som_aantal=("aantal","sum"),
    )
    grp["gem_klasse_per_soort"] = grp.apply(lambda r: (r["som_gew_klasse"]/r["som_aantal"]) if r["som_aantal"]>0 else 0.0, axis=1)
    grp["marge"] = grp["omzet"] - grp["inkoop"]
    grp["marge_pct"] = grp.apply(lambda r: (r["marge"]/r["omzet"]*100.0) if r["omzet"]>0 else 0.0, axis=1)
    return grp[["soort","totaal_stuks","omzet","inkoop","marge","marge_pct","gem_klasse_per_soort"]]

opts = st.session_state.get("gbb_options")
if not opts:
    st.info("Nog geen resultaten om te tonen. Klik eerst op **Optimaliseren**.")
else:
    labels = []
    for i, o in enumerate(opts):
        if i == 0: labels.append("GOED — MINIMALE PRIJS" if st.session_state.get("opt_goal")=="KLASSE" else "GOED — BUDGETVRIENDELIJK")
        elif i == 1: labels.append("ALTERNATIEF A")
        elif i == 2: labels.append("ALTERNATIEF B" if st.session_state.get("opt_goal")=="KLASSE" else "BEST — MAX KWALITEIT BINNEN BUDGET")
        else: labels.append(o.get("name", f"Optie {i+1}"))

    sel_label = st.selectbox("Toon details van", labels, index=0, key="lead_choice")
    sel_idx = labels.index(sel_label)
    sel = opts[sel_idx]
    st.session_state["selected_option_name"] = sel.get("name", labels[sel_idx])

    sel_df = sel["result"].copy()
    sel_df.columns = sel_df.columns.str.lower()
    for c in ["artikel","soort","merk","klasse","aantal"]:
        if c not in sel_df.columns: sel_df[c] = ""

    if "price_effective" not in sel_df.columns and "prijs" in sel_df.columns:
        sel_df["price_effective"] = sel_df["prijs"]
    if "price_buy_unit" not in sel_df.columns:
        sel_df["price_buy_unit"] = 0.0

    sel_df["aantal"] = pd.to_numeric(sel_df["aantal"], errors="coerce").fillna(0).astype(int)
    sel_df = sel_df[sel_df["aantal"] > 0].copy()
    sel_df["omzet_subtotaal"] = sel_df["aantal"].astype(float) * sel_df["price_effective"].astype(float)
    sel_df["inkoop_subtotaal"] = sel_df["aantal"].astype(float) * sel_df["price_buy_unit"].astype(float)

    show = sel_df[["soort","artikel","merk","aantal","price_effective","omzet_subtotaal"]].rename(
        columns={
            "soort":"Soort","artikel":"Artikel","merk":"Merk","aantal":"Aantal",
            "price_effective":"Prijs/stuk (klant)","omzet_subtotaal":"Subtotaal (klant)"
        }
    ).sort_values(["Soort","Artikel"])

    st.dataframe(
        show.style.format({"Prijs/stuk (klant)": _euro0, "Subtotaal (klant)": _euro0}),
        use_container_width=True, hide_index=True
    )

    total_omzet = float(show["Subtotaal (klant)"].sum()) if not show.empty else 0.0
    total_inkoop = float(sel_df["inkoop_subtotaal"].sum()) if not sel_df.empty else 0.0
    marge_eur = total_omzet - total_inkoop
    marge_pct = (marge_eur / total_omzet * 100.0) if total_omzet > 0 else 0.0
    avg_klasse = float(sel.get("avg", 0.0))

    cA, cB, cC = st.columns(3)
    with cA: st.metric("Totaal (klant)", _euro0(total_omzet))
    with cB: st.metric("Gemiddelde klasse", f"{avg_klasse:.2f}".replace(".", ","))
    with cC: st.metric("Marge (intern)", f"{_euro0(marge_eur)}  ·  {marge_pct:.1f}%")
    st.caption(f"**Totaal stuks:** {int(show['Aantal'].sum()) if not show.empty else 0}")

    grp = _build_grp_from_selection(sel_df)
    st.session_state["current_sel_df"] = sel_df.copy()
    st.session_state["current_grp"] = grp.copy()

    # ==== PDF EXPORT ==========================================================
    st.markdown("### PDF EXPORT")
    c1, c2 = st.columns([2, 1])
    with c1:
        project_title = st.text_input("Titel op de offerte/rapport", value=st.session_state.get("pdf_title", ""), key="pdf_title")
        project_client = st.text_input("Klantnaam (optioneel)", value=st.session_state.get("pdf_client", ""), key="pdf_client")
        project_ref = st.text_input("Referentie (optioneel)", value=st.session_state.get("pdf_ref", ""), key="pdf_ref")
    with c2:
        pdf_variant = st.radio("PDF-type", ["Offerte (klant)", "Rapport (intern)"], horizontal=True, index=0, key="pdf_variant_radio")
        only_first_page = st.checkbox("Alleen 1e pagina bij PDF-briefpapier", value=st.session_state.get("pdf_firstpage", True), key="pdf_firstpage")

    internal = pdf_variant == "Rapport (intern)"
    letterhead_file = st.file_uploader("Optioneel: upload briefpapier (PNG, JPG of PDF)", type=["png","jpg","jpeg","pdf"], key="pdf_letterhead")

    if st.button("📄 Genereer PDF", key="pdf_generate"):
        sel_df_pdf = st.session_state.get("current_sel_df", sel_df).copy()
        grp_pdf = st.session_state.get("current_grp", grp).copy()

        def _eur(v):
            try:
                return f"€ {float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            except Exception:
                return str(v)

        if "prijs" not in sel_df_pdf.columns and "price_effective" in sel_df_pdf.columns:
            sel_df_pdf["prijs"] = sel_df_pdf["price_effective"]
        for need in ["aantal","price_effective","price_buy_unit","prijs"]:
            if need not in sel_df_pdf.columns:
                sel_df_pdf[need] = 0.0

        sel_df_pdf["Brutoprijs"] = sel_df_pdf["prijs"].apply(_eur)
        sel_df_pdf["Verkoop/klant"] = sel_df_pdf["price_effective"].apply(_eur)
        sel_df_pdf["Inkoop/netto"] = sel_df_pdf["price_buy_unit"].apply(_eur)
        sel_df_pdf["Subtotaal klant"] = (sel_df_pdf["aantal"] * sel_df_pdf["price_effective"]).apply(_eur)
        sel_df_pdf["Subtotaal inkoop"] = (sel_df_pdf["aantal"] * sel_df_pdf["price_buy_unit"]).apply(_eur)

        pdf_grp = grp_pdf.copy()
        for c in ["omzet","inkoop","marge","marge_pct","gem_klasse_per_soort"]:
            if c in pdf_grp.columns: pdf_grp[c] = pdf_grp[c].round(2)

        meta = {
            "title": (project_title or ("Rapport (intern)" if internal else "Offerte")),
            "client": (project_client or "").strip(),
            "reference": (project_ref or "").strip(),
            "date_str": format_date(),
            "discount_client_str": f"{float(st.session_state.get('klant_korting_value', klant_korting_pct)):.0f}%",
            "budget_str": euro(st.session_state.get("last_budget_shown", 0), 0) if st.session_state.get("last_budget_shown") else "",
            "mode_str": st.session_state.get("badge", ""),
            "footer": "© jouw-merk – gegenereerd met Value Optimizer",
        }

        final_pdf = None
        letter_bytes = letterhead_file.read() if letterhead_file is not None else None
        name = (letterhead_file.name.lower() if letterhead_file is not None else "")
        mime = (letterhead_file.type if letterhead_file is not None else "")

        if letter_bytes and (mime.startswith("image/") or name.endswith((".png",".jpg",".jpeg"))):
            base_pdf = build_offer_pdf(
                project_meta=meta,
                result_df=sel_df_pdf,
                summary_df=(pdf_grp[["soort","totaal_stuks","omzet","inkoop","marge","marge_pct","gem_klasse_per_soort"]] if internal
                            else pdf_grp[["soort","totaal_stuks","omzet","gem_klasse_per_soort"]]),
                show_prices=True,
                letterhead_image_bytes=letter_bytes,
                internal=internal,
            )
            final_pdf = base_pdf
        else:
            base_pdf = build_offer_pdf(
                project_meta=meta,
                result_df=sel_df_pdf,
                summary_df=(pdf_grp[["soort","totaal_stuks","omzet","inkoop","marge","marge_pct","gem_klasse_per_soort"]] if internal
                            else pdf_grp[["soort","totaal_stuks","omzet","gem_klasse_per_soort"]]),
                show_prices=True,
                letterhead_image_bytes=None,
                internal=internal,
            )
            if letter_bytes and (mime == "application/pdf" or name.endswith(".pdf")):
                final_pdf = overlay_pdf_letterhead(base_pdf, letter_bytes, first_page_only=only_first_page)
            else:
                final_pdf = base_pdf

        st.success("PDF klaar om te downloaden.")
        st.download_button(
            "⬇️ Download PDF",
            data=final_pdf,
            file_name=f"{'Rapport' if internal else 'Offerte'}_{(project_ref or format_date()).replace(' ', '_')}.pdf",
            mime="application/pdf",
        )

# ==== Reset ===================================================================
st.markdown('<hr class="soft">', unsafe_allow_html=True)

def _soft_reset():
    for k in [
        "editor_types","editor_df","editor_df_widget","gbb_options","badge","klant_korting_pct_input",
        "klant_korting_value","budget_input_str","budget_value","last_budget_shown","lead_choice",
        "selected_option_name","current_sel_df","current_grp","pdf_title","pdf_client","pdf_ref",
        "pdf_variant_radio","pdf_firstpage","pdf_letterhead","opt_goal"
    ]:
        st.session_state.pop(k, None)
    st.rerun()

st.button("RESET INVOER", help="Zet aantallen/keuzes/resultaten terug; behoudt huidige prijslijst-keuze.", on_click=_soft_reset)
