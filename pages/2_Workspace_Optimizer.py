import math
import streamlit as st
import pandas as pd
from typing import Dict, Tuple
from streamlit.errors import StreamlitAPIException
try:
    st.set_page_config(page_title="The Studio Suite", page_icon="🎛️", layout="wide", initial_sidebar_state="expanded")
except StreamlitAPIException:
    pass

# (optioneel thema/navigatie – faalt stil als je module niet hebt)
try:
    from ui_theme import inject_theme, render_nav_with_icons
except Exception:
    def inject_theme(): pass
    def render_nav_with_icons(): pass

# ========== EU-notatie (hele euro’s) ==========
try:
    from babel.numbers import format_currency, format_decimal
    def euro(x): return format_currency(round(float(x or 0), 0), "EUR", locale="nl_NL")
    def dec(x, nd=0): return format_decimal(round(float(x or 0), nd), locale="nl_NL")
except Exception:
    def _swap(s): return s.replace(",", "§").replace(".", ",").replace("§", ".")
    def euro(x):
        try: v = float(x or 0)
        except: v = 0.0
        return "€ " + _swap(f"{round(v):,.0f}")
    def dec(x, nd=0):
        try: v = float(x or 0)
        except: v = 0.0
        return _swap(f"{v:,.{nd}f}")

from pulp import HiGHS_CMD, PULP_CBC_CMD

def get_solver():
    try:
        return HiGHS_CMD(msg=False)
    except Exception:
        return PULP_CBC_CMD(msg=False)  # fallback als CBC toch aanwezig is

# later:
# prob.solve(get_solver())


# ========== Setup ==========
inject_theme()
render_nav_with_icons()

# Grote quote-stijl
st.markdown("""
<style>
.step-label {
  font-size: 13px; letter-spacing: .06em; text-transform: uppercase;
  color:#0f172a; font-weight:700; margin: 18px 0 6px 0;
}
.big-quote {
  font-family: Georgia, "Times New Roman", Times, serif;
  font-style: italic;
  font-size: 22px;
  line-height: 1.6;
  border-left: 6px solid #000;
  padding: 14px 18px;
  margin: 0 0 18px 0;
  border-radius: 4px;
}
.big-quote b { font-style: normal; }
.note-soft {
  background: #f5f5f5; border: 1px solid #e5e7eb;
  border-radius: 6px; padding: 12px 14px; margin-top: 8px;
}
.note-soft h3 { margin: 0 0 6px 0; font-size: 16px; font-weight: 600; }
.note-soft ul { margin: 0; padding-left: 18px; }
</style>
""", unsafe_allow_html=True)
st.markdown(
    """
    <div class="big-quote">
      <b><i>Workspace Optimizer</b></i> laat in één oogopslag zien wat je kantoorinrichting kost.
      Speel met kwaliteit, installaties, begeleiding en onvoorzien; de app rekent transparant mee —
      van investering vandaag tot jaarlijkse kosten morgen.
    </div>
    <hr style="border:0; border-top:1px solid #e5e7eb; margin:1.5rem 0;">
    """,
    unsafe_allow_html=True
)

# ========== Baselines & factoren ==========
BASE_YEAR = 2024
BASELINES_P50 = {"office": 600.0, "hospitality": 900.0, "retail": 800.0, "residential": 700.0}
QUALITY_FACTOR    = {"basiskwaliteit": 0.85, "middensegment": 1.00, "high-end": 1.25}
REGION_FACTOR     = {"NL-Randstad": 1.05, "NL-overig": 0.97, "EU-noord": 1.00, "EU-zuid": 0.95}
COMPLEXITY_FACTOR = {"standard": 1.00, "maatwerk": 1.10, "monument": 1.20}
SCOPE_FACTOR      = {"afbouw + FF&E": 1.00, "FF&E": 0.60, "afbouw": 0.55}
CPI_INDEX = {2021: 1.027, 2022: 1.100, 2023: 1.038, 2024: 1.030, 2025: 1.026}

# NFC / OpEx defaults
NFC_OPEX_EUR_PER_M2 = 589.0
NFC_M2_PER_WORKPLACE_VVO = 22.3

# ========== Helpers ==========
def inflation_factor(year: int, base_year: int, cpi: Dict[int, float]) -> float:
    if year == base_year: return 1.0
    years = sorted(cpi.keys())
    if year not in cpi or base_year not in cpi: return 1.0
    def cum(to_year):
        f = 1.0
        for y in range(years[0], to_year + 1):
            f *= cpi.get(y, 1.0)
        return f
    return cum(year) / cum(base_year)

def compute_base_per_m2(project_type: str, scope: str, quality: str, region: str,
                        year: int, complexity: str, cpi_index: Dict[int, float]) -> float:
    """Basis €/m² excl. MEP & PM (incl. scope/kwaliteit/regio/complexiteit en CPI)."""
    val = BASELINES_P50.get(project_type, 650.0)
    val *= SCOPE_FACTOR.get(scope, 1.0)
    val *= QUALITY_FACTOR.get(quality, 1.0)
    val *= REGION_FACTOR.get(region, 1.0)
    val *= COMPLEXITY_FACTOR.get(complexity, 1.0)
    val *= inflation_factor(year, BASE_YEAR, cpi_index)
    return val

def compute_model_band(p50: float) -> Tuple[float, float]:
    """Heuristiek; band blijft ‘basis’ (excl. onvoorzien/add-ons)."""
    return p50 * 0.85, p50 * 1.25

# ========== Sidebar ==========
st.sidebar.header("⚙️ Config")
cpi_edit = st.sidebar.checkbox("CPI handmatig bewerken?", value=False)
cpi_state = CPI_INDEX.copy()
if cpi_edit:
    st.sidebar.caption("Index per jaar (1,03 = +3% vs vorig jaar).")
    for y in sorted(cpi_state.keys()):
        cpi_state[y] = st.sidebar.number_input(f"CPI {y}", value=float(cpi_state[y]), step=0.001, format="%.3f")

# ========== Main UI ==========
st.markdown('<div class="step-label">PROJECTGEGEVENS</div>', unsafe_allow_html=True)
st.caption("Voer hieronder de projectgegevens in")

col1, col2, col3 = st.columns(3)
with col1:
    project_type = st.selectbox("Projecttype", ["office", "hospitality", "retail", "residential"])
    scope_ui = st.selectbox("Projectkenmerken", ["FF&E + afbouw", "FF&E", "afbouw"],
                            help="FF&E = Furniture, Fixtures & Equipment (meubilair, inrichting, apparatuur).")
    quality = st.selectbox("Kwaliteit", ["basiskwaliteit", "middensegment", "high-end"])
with col2:
    region = st.selectbox("Regio", ["NL-Randstad", "NL-overig", "EU-noord", "EU-zuid"])
    year = st.number_input("Jaar (kostenpeil)", min_value=2021, max_value=2030, value=2025, step=1)
    complexity = st.selectbox("Complexiteit", ["standard", "maatwerk", "monument"])
with col3:
    m2 = st.number_input("Oppervlakte (m²)", min_value=10.0, max_value=50000.0, value=1000.0, step=10.0)

    MEP_OPTIONS = [
        "Geen installatiekosten (0%)",
        "Licht (+5%)", "Normaal (+15%)",
        "Zwaar (+30%)", "Ingrijpend (+50%)", "Zwaar ingrijpend (+70%)"
    ]
    if scope_ui == "FF&E":
        mep_choice = st.selectbox(
            "Gewicht installatiekosten",
            ["Geen installatiekosten (0%)"], index=0, disabled=True,
            help="Installaties horen niet bij FF&E — MEP staat hier vast op 0%."
        )
    else:
        mep_choice = st.selectbox("Gewicht installatiekosten", MEP_OPTIONS, index=2)

    pm_pct_input = st.number_input("PM & logistiek (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.5)

# NEN 2580 conversie
st.markdown('<div class="step-label">VVO, NVO of BVO</div>', unsafe_allow_html=True)
m2_input_type = st.radio("Is het aantal m² Verhuurbaar, Netto of Bruto Vloeroppervlak?",
                         ["VVO","NVO (netto)","BVO (bruto)"], horizontal=True)
nvo_to_vvo, bvo_to_vvo = 1.08, 0.80
m2_vvo = m2
if m2_input_type.startswith("NVO"):
    nvo_to_vvo = st.slider("VVO t.o.v. NVO", 1.00, 1.20, 1.08, 0.01)
    m2_vvo = m2 * nvo_to_vvo
elif m2_input_type.startswith("BVO"):
    bvo_to_vvo = st.slider("VVO t.o.v. BVO", 0.70, 0.90, 0.80, 0.01)
    m2_vvo = m2 * bvo_to_vvo
st.caption(f"Interne rekenbasis: **{dec(m2_vvo,0)} m² VVO** (input: {dec(m2,0)} m² {m2_input_type.split()[0]})")

# Mapping & factoren + P50/P20/P80
MEP_FACTOR = {
    "Geen installatiekosten (0%)": 1.00,
    "Licht (+5%)": 1.05, "Normaal (+15%)": 1.15,
    "Zwaar (+30%)": 1.30, "Ingrijpend (+50%)": 1.50, "Zwaar ingrijpend (+70%)": 1.70
}
mep_factor_selected = MEP_FACTOR.get(mep_choice, 1.15)
pm_frac = pm_pct_input / 100.0

scope_ui_to_model = {"FF&E + afbouw": "afbouw + FF&E", "FF&E": "FF&E", "afbouw": "afbouw"}
scope_model = scope_ui_to_model.get(scope_ui, scope_ui)
mep_factor_effective = 1.00 if scope_model == "FF&E" else mep_factor_selected  # FF&E => MEP 0%

base_per_m2 = compute_base_per_m2(
    project_type, scope_model, quality, region, year,
    complexity, (cpi_state if cpi_edit else CPI_INDEX)
)
mep_pct = max(0.0, (1.00 if scope_model == "FF&E" else mep_factor_effective) - 1.0)  # 1.15 -> 0.15
pm_pct  = max(0.0, pm_frac)                                                           # 0.10 -> 0.10
p50 = base_per_m2 * (1.0 + mep_pct + pm_pct)
p20, p80 = compute_model_band(p50)

# ========== Office-specifiek ==========
n_employees = 0
annual_opex_total = 0.0
capex_meeting_extra = 0.0
contingency_pct = 0
horizon = 0
opex_growth = 0.0

if project_type == "office":
    st.markdown('<div class="step-label">AANTAL MEDEWERKERS EN OPERATIONELE KOSTEN</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        n_employees = st.number_input("Aantal medewerkers", min_value=1, value=50, step=1)
        basis = st.radio("Hoe wil je de benodigde m² bepalen?",
                         ["Handmatig (boven)","NFC-norm (22,3 m² VVO per medewerker)","Eigen norm (netto m² per medewerker) — zelf invullen"], index=0)
    if basis.startswith("NFC"):
        m2_vvo = n_employees * NFC_M2_PER_WORKPLACE_VVO
        st.info(f"Benodigde m² (VVO) o.b.v. NFC: **{dec(m2_vvo,0)} m²**")
    elif basis.startswith("Eigen norm"):
        net_per_emp = st.slider("Eigen norm (netto m² per medewerker)", 8.0, 20.0, 12.0, 0.5)
        m2_vvo = n_employees * net_per_emp * (nvo_to_vvo if not m2_input_type.startswith("BVO") else 1.0)

    c1, c2 = st.columns(2)
    with c1:
        rent_per_m2_year = st.number_input("Huur €/m²/jaar (optioneel)", 0.0, 1_000_000.0, 0.0, 5.0)
        opex_growth = st.number_input("Verwachte groei operationele kosten p.j. (%)", 0.0, 50.0, 3.0, 0.5)
        use_nfc_opex = st.checkbox("Gebruik het NFC OpEx gemiddelde van €589/m² per jaar", value=True)
    annual_opex_facility = (NFC_OPEX_EUR_PER_M2 * m2_vvo) if use_nfc_opex else st.number_input("Eigen OpEx €/jaar (facilitair, excl. huur)", 0.0, 1_000_000_000.0, 0.0, 1000.0)
    annual_rent = rent_per_m2_year * m2_vvo
    annual_opex_total = annual_opex_facility + annual_rent

    horizon = st.slider("Horizon / aantal jaren waarover totale facilitaire kosten wordt vastgesteld", 1, 10, 5)

    st.metric("Totaal operationele kosten in 1e jaar (OpEx)", euro(annual_opex_total))
    st.metric("Operationele kosten per medewerker in 1e jaar (OpEx/aantal medewerkers)", euro(annual_opex_total/max(n_employees,1)))
    st.caption(f"(huur: {euro(annual_rent)} • facilitair: {euro(annual_opex_facility)})")

    # --- VLOERVERDELING (VVO %) — volledig binnen office-blok ---
    st.markdown('<div class="step-label">VLOERVERDELING (VVO %)</div>', unsafe_allow_html=True)
    KEY_WP, KEY_MTG, KEY_SUP = "prog_wp", "prog_mtg", "prog_sup"
    st.session_state.setdefault(KEY_WP, 60)
    st.session_state.setdefault(KEY_MTG, 25)
    st.session_state.setdefault(KEY_SUP, 15)

    def _normalize_program():
        wp  = int(st.session_state[KEY_WP])
        mtg = int(st.session_state[KEY_MTG])
        sup = int(st.session_state[KEY_SUP])
        total = wp + mtg + sup
        if total == 0:
            wp, mtg, sup = 34, 33, 33
        else:
            factor = 100.0 / total
            raw = [wp*factor, mtg*factor, sup*factor]
            floors = [math.floor(x) for x in raw]
            remainder = 100 - sum(floors)
            decs = sorted([(raw[i] - floors[i], i) for i in range(3)], reverse=True)
            for i in range(remainder): floors[decs[i][1]] += 1
            wp, mtg, sup = floors
        st.session_state[KEY_WP], st.session_state[KEY_MTG], st.session_state[KEY_SUP] = int(wp), int(mtg), int(sup)

    def _reset_program():
        st.session_state[KEY_WP], st.session_state[KEY_MTG], st.session_state[KEY_SUP] = 60, 25, 15

    w1, w2, w3 = st.columns(3)
    wp  = w1.slider("Werkplekken %",     0, 100, step=1, key=KEY_WP)
    mtg = w2.slider("Vergaderruimtes %", 0, 100, step=1, key=KEY_MTG)
    sup = w3.slider("Support %",         0, 100, step=1, key=KEY_SUP)
    st.caption(f"Som: {wp + mtg + sup}%")

    cbtn, creset = st.columns([1, 1])
    with cbtn:   st.button("Maak samen 100%", on_click=_normalize_program)
    with creset: st.button("Reset 60 / 25 / 15", on_click=_reset_program)

    # --- VERGADERRUIMTES (OPTIONEEL) + ONVOORZIEN (alleen UI/inputs; nog géén totals!) ---
    st.markdown('<div class="step-label">VERGADERRUIMTES (OPTIONEEL)</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        seats_small = st.number_input("Zitplaatsen: klein", 0, 10000, 0)
        seats_med   = st.number_input("Zitplaatsen: medium",0, 10000, 0)
        seats_large = st.number_input("Zitplaatsen: groot", 0, 10000, 0)
    with s2:
        eur_small = st.number_input("€ per zitplaats (klein)", 0.0, 1_000_000.0, 500.0, 50.0)
        eur_med   = st.number_input("€ per zitplaats (medium)",0.0, 1_000_000.0, 750.0, 50.0)
        eur_large = st.number_input("€ per zitplaats (groot)", 0.0, 1_000_000.0,1000.0, 50.0)
    capex_meeting_extra = seats_small*eur_small + seats_med*eur_med + seats_large*eur_large

    st.markdown('<div class="step-label">Post onvoorzien op investering (CapEx)</div>', unsafe_allow_html=True)
    contingency_pct = st.slider("Onvoorzien % op totale investering", 0, 30, 5, 1)

# ========== RESULTATEN ==========
st.markdown('<div class="step-label">RESULTATEN</div>', unsafe_allow_html=True)

# --- Begrotingsniveau hier ---
choice = st.radio(
    "Begrotingsniveau",
    ["Richtprijs (P50)", "Veilig (P80)"],
    index=0, horizontal=True,
    help="Kies welk niveau je wil hanteren voor de totaalbedragen."
)
sel_label  = "P50" if choice.startswith("Richtprijs") else "P80"
sel_per_m2 = p50 if sel_label == "P50" else p80

# --- CapEx en €/m² (alles dat sel_per_m2 nodig heeft) ---
capex_base = sel_per_m2 * m2_vvo
if project_type == "office":
    capex_total = (capex_base + capex_meeting_extra) * (1 + contingency_pct/100.0)
else:
    capex_total = capex_base

if project_type == "office" and m2_vvo > 0:
    addon_per_m2 = capex_meeting_extra / m2_vvo
    cont_amount  = (capex_base + capex_meeting_extra) * (contingency_pct / 100.0)
    cont_per_m2  = cont_amount / m2_vvo
    extra_per_m2 = addon_per_m2 + cont_per_m2
else:
    addon_per_m2 = cont_per_m2 = extra_per_m2 = 0.0

eff_sel_per_m2 = sel_per_m2 + extra_per_m2

# --- Tiles (2 rijen × 3) ---
r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    st.metric("P50-basis per m² (P50)", f"{dec(p50,0)} €/m²")
with r1c2:
    st.metric("Onvoorzien + add-ons per m²", f"{dec(extra_per_m2,0)} €/m²")
    if project_type == "office" and m2_vvo > 0:
        st.caption(f"Add-ons: {dec(addon_per_m2,0)} €/m² • Onvoorzien: {dec(cont_per_m2,0)} €/m²")
with r1c3:
    st.metric(f"€/m² incl. onvoorzien & add-ons — {sel_label}", f"{dec(eff_sel_per_m2,0)} €/m²")

r2c1, r2c2, r2c3 = st.columns(3)
with r2c1:
    st.metric("Bandbreedte P20–P80-basis per m²", f"{dec(p20,0)} – {dec(p80,0)} €/m²")
with r2c2:
    st.metric(f"Indicatieve investering (CapEx — {sel_label})", euro(capex_total))
with r2c3:
    horizon_val = horizon if (project_type == "office") else 0
    totex_val = (capex_total + sum(annual_opex_total * ((1 + opex_growth/100)**t) for t in range(horizon_val))) \
                if (project_type == "office" and horizon_val > 0) else None
    label_totex = f"Totale investering + kosten over {horizon_val} jaar (TotEx=CapEx+OpEx)"
    st.metric(label_totex, euro(totex_val) if totex_val is not None else "—")

# --- Uitlegblok ---
st.markdown("""
<div class="note-soft">
  <h3>Uitleg P20 · P50 · P80</h3>
  <ul>
    <li><strong>P50</strong> – middenwaarde / realistische richtprijs: ~50% kans dat het hoger of lager uitvalt.</li>
    <li><strong>P20</strong> – optimistisch (laag): ~20% kans nóg lager, ~80% kans hoger.</li>
    <li><strong>P80</strong> – voorzichtig (hoog): ~80% kans dat je eronder blijft, ~20% kans erboven.</li>
  </ul>
  <p><em>Tip:</em> gebruik <strong>P50</strong> als richtprijs en <strong>P80</strong> als veilige begroting (buffer).</p>
</div>
""", unsafe_allow_html=True)

# ========== PROGRAMMAOVERZICHT ==========
if project_type == "office":
    KEY_WP, KEY_MTG, KEY_SUP = "prog_wp", "prog_mtg", "prog_sup"
    wp_pct  = int(st.session_state.get(KEY_WP, 60))
    mtg_pct = int(st.session_state.get(KEY_MTG, 25))
    sup_pct = int(st.session_state.get(KEY_SUP, 15))
    if wp_pct + mtg_pct + sup_pct == 100:
        m2_wp  = m2_vvo * (wp_pct/100.0)
        m2_mtg = m2_vvo * (mtg_pct/100.0)
        m2_sup = m2_vvo * (sup_pct/100.0)

        capex_base_wp  = sel_per_m2 * m2_wp
        capex_base_mtg = sel_per_m2 * m2_mtg
        capex_base_sup = sel_per_m2 * m2_sup

        pre_wp  = capex_base_wp
        pre_mtg = capex_base_mtg + capex_meeting_extra
        pre_sup = capex_base_sup

        tot_wp  = pre_wp  * (1 + contingency_pct/100.0)
        tot_mtg = pre_mtg * (1 + contingency_pct/100.0)
        tot_sup = pre_sup * (1 + contingency_pct/100.0)

        opex_wp  = annual_opex_total * (wp_pct/100.0)
        opex_mtg = annual_opex_total * (mtg_pct/100.0)
        opex_sup = annual_opex_total * (sup_pct/100.0)

        prog_df = pd.DataFrame([
            ["Werkplekken",    wp_pct,  m2_wp,  capex_base_wp,  0.0,                 tot_wp,  opex_wp],
            ["Vergaderruimte", mtg_pct, m2_mtg, capex_base_mtg, capex_meeting_extra, tot_mtg, opex_mtg],
            ["Support",        sup_pct, m2_sup, capex_base_sup, 0.0,                 tot_sup, opex_sup],
        ], columns=["Categorie","% VVO","m² VVO","CapEx basis €","CapEx add-on €","CapEx + onvoorzien €","OpEx jaar 1 €"])
        st.markdown('<div class="step-label">PROGRAMMAOVERZICHT</div>', unsafe_allow_html=True)
        st.dataframe(
            prog_df.style.format({
                "% VVO": lambda v: f"{int(round(v))} %",
                "m² VVO": lambda v: dec(v,0),
                "CapEx basis €": euro,
                "CapEx add-on €": euro,
                "CapEx + onvoorzien €": euro,
                "OpEx jaar 1 €": euro,
            }),
            use_container_width=True
        )

# ========== Details & aannames ==========
with st.expander("Details & aannames"):
    base = BASELINES_P50.get(project_type, 650.0)
    f_scope = SCOPE_FACTOR.get(scope_model, 1.0)
    f_qual  = QUALITY_FACTOR.get(quality, 1.0)
    f_reg   = REGION_FACTOR.get(region, 1.0)
    f_cplx  = COMPLEXITY_FACTOR.get(complexity, 1.0)
    f_cpi   = inflation_factor(year, BASE_YEAR, cpi_state if cpi_edit else CPI_INDEX)
    base_calc = base * f_scope * f_qual * f_reg * f_cplx * f_cpi

    st.write(
        f"{dec(base,0)} (baseline) × {f_scope:.2f} (scope) × {f_qual:.2f} (kwaliteit) × "
        f"{f_reg:.2f} (regio) × {f_cplx:.2f} (complexiteit) × {f_cpi:.3f} (CPI) "
        f"= **{dec(base_calc,0)} €/m²** (basis excl. MEP & PM)"
    )
    st.write(
        f"+ {int(round(mep_pct*100))}% MEP + {int(round(pm_pct*100))}% PM ⇒ "
        f"**{dec(p50,0)} €/m² (P50-basis)**"
    )
    if project_type == "office":
        st.write(f"- Add-ons vergaderruimtes: {euro(capex_meeting_extra)} • Onvoorzien: {int(round(contingency_pct))}%")
        st.write(f"- OpEx jaar 1 totaal: {euro(annual_opex_total)} • Horizon: {horizon} jaar • Groei: {dec(opex_growth,1)}% p.j.")
