# ============================================
# pages/1_Value_Optimizer.py  —  schoon & stabiel
# ============================================

from __future__ import annotations

import datetime
from pathlib import Path
import math

import streamlit as st
import pandas as pd
import pulp
from streamlit.errors import StreamlitAPIException

# ---------- Basissetup ----------
try:
    st.set_page_config(
        page_title="The Studio — Value Optimizer",
        page_icon="🎛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except StreamlitAPIException:
    pass

# (optioneel) thema / nav – faalt stil als module niet bestaat
try:
    from ui_theme import inject_theme, render_nav_with_icons
    inject_theme()
    render_nav_with_icons()
except Exception:
    pass


# ---------- Helpers: EU-format ----------
def _swap(s: str) -> str:
    return s.replace(",", "§").replace(".", ",").replace("§", ".")

def euro(v: float, nd: int = 0) -> str:
    try:
        x = round(float(v or 0), nd)
    except Exception:
        x = 0.0
    return "€ " + _swap(f"{x:,.{nd}f}")

def int_to_eu(n: int) -> str:
    try:
        return f"{int(n):,}".replace(",", ".")
    except Exception:
        return "0"

def eu_to_int(s: str | None) -> int:
    if not s:
        return 0
    s = str(s).strip().replace(".", "").replace(" ", "").replace(",", ".")
    try:
        return int(float(s))
    except Exception:
        return 0

def pct_to_float(v) -> float:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 0.0
    s = str(v).strip().replace("%", "").replace(",", ".")
    try:
        val = float(s)
        return val / 100.0 if val > 1 else val
    except Exception:
        return 0.0


# ---------- Data normalisatie ----------
def _normalize_pricelist(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    required = {"artikel", "soort", "merk", "prijs", "klasse"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Ontbrekende kolommen: " + ", ".join(sorted(missing)))

    df["artikel"] = df["artikel"].astype(str)
    df["soort"]   = df["soort"].astype(str)
    df["merk"]    = df["merk"].astype(str)

    df["prijs"]   = pd.to_numeric(df["prijs"], errors="coerce")
    df["klasse"]  = pd.to_numeric(df["klasse"], errors="coerce")

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
        data = [
            ["Bureau 160", "Bureau", "BrandA", 550, 2.0, 0.10],
            ["Bureau 180", "Bureau", "BrandB", 690, 3.0, 0.15],
            ["Ergostoel", "Stoel", "BrandA", 420, 3.5, 0.05],
            ["Stoel Basic", "Stoel", "BrandC", 220, 2.0, 0.00],
            ["Kast 2-deurs", "Opbergen", "BrandD", 380, 2.5, 0.12],
            ["Kast hoog", "Opbergen", "BrandD", 520, 3.0, 0.12],
            ["Lamp vloei", "Verlichting", "BrandE", 140, 1.5, 0.00],
            ["Lamp taak", "Verlichting", "BrandE", 240, 2.5, 0.00],
            ["Tafel 4p", "Vergader", "BrandF", 980, 3.2, 0.10],
            ["Tafel 8p", "Vergader", "BrandF", 1650, 4.0, 0.10],
        ]
        df = pd.DataFrame(
            data,
            columns=["artikel", "soort", "merk", "prijs", "klasse", "korting_leverancier"],
        )
    return _normalize_pricelist(df)


# ---------- LP bouwstenen ----------
def _build_common(df, needs):
    df = df.copy()
    df.columns = df.columns.str.lower()
    price_col = "price_effective" if "price_effective" in df.columns else "prijs"

    types = sorted(df["soort"].astype(str).unique().tolist())
    idx_by_type = {s: [i for i in df.index if str(df.loc[i, "soort"]) == s] for s in types}

    x = {i: pulp.LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in df.index}
    y = {}  # voor no-mix

    price_sum = pulp.lpSum(x[i] * float(df.loc[i, price_col]) for i in df.index)
    class_sum = pulp.lpSum(x[i] * float(df.loc[i, "klasse"]) for i in df.index)

    return df, price_col, types, idx_by_type, x, y, price_sum, class_sum


def _add_core_constraints(
    model,
    df,
    needs: dict[str, int],
    idx_by_type: dict[str, list[int]],
    x: dict[int, pulp.LpVariable],
    y: dict[int, pulp.LpVariable],
    min_class_by_type: dict[str, float] | None,
    max_class_by_type: dict[str, float] | None,
    no_mix_by_type: dict[str, bool] | None,
):
    min_class_by_type = min_class_by_type or {}
    max_class_by_type = max_class_by_type or {}
    no_mix_by_type = no_mix_by_type or {}

    # Aantallen per soort
    for s, qty in needs.items():
        if int(qty) > 0:
            idx = idx_by_type.get(s, [])
            model += pulp.lpSum(x[i] for i in idx) == int(qty)

    # Min/max klasse per soort
    for s, qty in needs.items():
        if int(qty) <= 0:
            continue
        idx = idx_by_type.get(s, [])
        mn = float(min_class_by_type.get(s, 0.0) or 0.0)
        mx = float(max_class_by_type.get(s, 0.0) or 0.0)
        if mn > 0:
            model += pulp.lpSum(x[i] * float(df.loc[i, "klasse"]) for i in idx) >= mn * int(qty)
        if mx > 0:
            model += pulp.lpSum(x[i] * float(df.loc[i, "klasse"]) for i in idx) <= mx * int(qty)

    # Niet mixen: per soort max 1 artikel kiezen
    for s, qty in needs.items():
        if int(qty) <= 0:
            continue
        if no_mix_by_type.get(s, False):
            idx = idx_by_type.get(s, [])
            for i in idx:
                if i not in y:
                    y[i] = pulp.LpVariable(f"y_{i}", lowBound=0, upBound=1, cat="Binary")
            model += pulp.lpSum(y[i] for i in idx) == 1
            for i in idx:
                model += x[i] <= int(qty) * y[i]


def solve_min_cost_with_quality(
    df,
    needs,
    min_avg,
    min_class_by_type=None,
    max_class_by_type=None,
    no_mix_by_type=None,
    budget_cap=None,
):
    df = df.copy()
    df.columns = df.columns.str.lower()
    required = {"artikel", "soort", "merk", "prijs", "klasse"}
    if not required.issubset(df.columns):
        raise ValueError("Ontbrekende kolommen voor optimalisatie.")

    active = [s for s, q in needs.items() if int(q) > 0]
    if not active:
        raise ValueError("Geen aantallen ingevuld (>0).")

    df = df[df["soort"].isin(active)].reset_index(drop=True)
    N = int(sum(needs.values()))

    df, price_col, types, idx_by_type, x, y, price_sum, class_sum = _build_common(df, needs)
    model = pulp.LpProblem("min_cost_quality", pulp.LpMinimize)
    model += price_sum

    _add_core_constraints(model, df, needs, idx_by_type, x, y, min_class_by_type, max_class_by_type, no_mix_by_type)

    if (min_avg is not None) and (min_avg > 0):
        model += class_sum >= float(min_avg) * N
    if (budget_cap is not None) and (budget_cap > 0):
        model += price_sum <= float(budget_cap)

    res = model.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[res] != "Optimal":
        raise RuntimeError("Geen optimale oplossing (min_cost_with_quality).")

    qtys = {i: int(x[i].value()) for i in df.index}
    df["aantal"] = df.index.map(qtys).astype(int)
    df = df[df["aantal"] > 0].copy()

    df["omzet_subtotaal"] = df["aantal"] * df.get("price_effective", df["prijs"])
    df["inkoop_subtotaal"] = df["aantal"] * df.get("price_buy_unit", 0.0)

    total_price = float(df["omzet_subtotaal"].sum())
    avg_class = float((df["aantal"] * df["klasse"]).sum() / N)

    return df, total_price, avg_class


def solve_max_quality_under_budget(
    df, needs, budget, min_class_by_type=None, max_class_by_type=None, no_mix_by_type=None
):
    df = df.copy()
    df.columns = df.columns.str.lower()
    required = {"artikel", "soort", "merk", "prijs", "klasse"}
    if not required.issubset(df.columns):
        raise ValueError("Ontbrekende kolommen voor optimalisatie.")

    active = [s for s, q in needs.items() if int(q) > 0]
    if not active:
        raise ValueError("Geen aantallen ingevuld (>0).")

    df = df[df["soort"].isin(active)].reset_index(drop=True)
    N = int(sum(needs.values()))

    df, price_col, types, idx_by_type, x, y, price_sum, class_sum = _build_common(df, needs)
    model = pulp.LpProblem("max_quality_budget", pulp.LpMaximize)
    model += class_sum

    _add_core_constraints(model, df, needs, idx_by_type, x, y, min_class_by_type, max_class_by_type, no_mix_by_type)
    model += price_sum <= float(budget)

    res = model.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[res] != "Optimal":
        raise RuntimeError("Geen optimale oplossing (max_quality_budget).")

    qtys = {i: int(x[i].value()) for i in df.index}
    df["aantal"] = df.index.map(qtys).astype(int)
    df = df[df["aantal"] > 0].copy()

    df["omzet_subtotaal"] = df["aantal"] * df.get("price_effective", df["prijs"])
    df["inkoop_subtotaal"] = df["aantal"] * df.get("price_buy_unit", 0.0)

    total_price = float(df["omzet_subtotaal"].sum())
    avg_class = float((df["aantal"] * df["klasse"]).sum() / N)

    return df, total_price, avg_class


# ---------- UI ----------
st.title("Furniture Value Optimizer")

st.markdown(
    """
    Kies of je **zo goedkoop mogelijk** wilt zijn voor een **gegeven kwaliteitsdrempel**,  
    of juist binnen een **vast budget** de **hoogste gemiddelde klasse** wilt halen.
    """
)

# Stap 1: bron
st.subheader("Stap 1 — Kies prijslijst")
DEMO_XLSX_PATH = Path(__file__).with_name("Demo_prijslijst.xlsx")

source_choice = st.radio("Bron", ["Demo-prijslijst gebruiken", "Upload eigen prijslijst"], index=0, horizontal=True)

if source_choice == "Upload eigen prijslijst":
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

# Stap 2: doel
st.subheader("Stap 2 — Kies optimalisatie-doel")
DOEL_KLASSE = "Minimale prijs bij de gekozen kwaliteitsdrempel"
DOEL_BUDGET = "Maximale kwaliteit binnen budget"
doel = st.radio("Doel", [DOEL_KLASSE, DOEL_BUDGET], index=0)

# Bewaar doel (belangrijk voor resultatenlabels)
st.session_state["opt_goal"] = "BUDGET" if doel == DOEL_BUDGET else "KLASSE"
goal_mode = st.session_state.get("opt_goal", "KLASSE")  # fallback

# Stap 3: korting/budget
st.subheader("Stap 3 — Klantkorting en (eventueel) budget")

colK, colB = st.columns([1, 2], vertical_alignment="center")
with colK:
    klant_korting_pct = st.number_input("Klantkorting (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)

def _format_budget_cb():
    raw = st.session_state.get("budget_input_str", "")
    n = eu_to_int(raw)
    st.session_state["budget_input_str"] = int_to_eu(n)
    st.session_state["budget_value"] = n

budget_net = None
if goal_mode == "BUDGET":
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

st.divider()

# Stap 4: invoer per soort
st.subheader("Stap 4 — Aantallen / Min/Max klasse / Niet mixen")

df_work = df_raw.copy()
df_work["korting_leverancier"] = df_work.get("korting_leverancier", 0.0).apply(pct_to_float)
klant_korting = pct_to_float(klant_korting_pct)
df_work["price_buy_unit"] = df_work["prijs"] * (1 - df_work["korting_leverancier"])
df_work["price_effective"] = df_work["prijs"] * (1 - klant_korting)

types = sorted(df_work["soort"].dropna().astype(str).unique().tolist())
init_rows = [
    {"soort": s, "actief": False, "aantal": 0, "min_klasse": None, "max_klasse": None, "niet_mixen": False}
    for s in types
]

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
            "aantal": st.column_config.NumberColumn("AANTAL", min_value=0, step=1),
            "min_klasse": st.column_config.NumberColumn("MIN. KLASSE", step=0.1),
            "max_klasse": st.column_config.NumberColumn("MAX. KLASSE (0=GEEN)", step=0.1),
            "niet_mixen": st.column_config.CheckboxColumn("NIET MIXEN"),
        },
    )

    target_avg = None
    if goal_mode == "KLASSE":
        st.markdown("**Doelklasse (gemiddeld)**")
        target_avg = st.slider(" ", min_value=1.0, max_value=5.0, value=2.7, step=0.05, label_visibility="collapsed")

    submitted = st.form_submit_button("Optimaliseren", type="primary")


# ---------- Normalisatie invoer ----------
def _normalize_editor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for k, v in {
        "soort": "",
        "actief": False,
        "aantal": 0,
        "min_klasse": 0.0,
        "max_klasse": 0.0,
        "niet_mixen": False,
    }.items():
        if k not in df.columns:
            df[k] = v
    df["soort"] = df["soort"].astype(str)
    df["actief"] = df["actief"].astype(bool)
    for c in ["aantal", "min_klasse", "max_klasse"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["niet_mixen"] = df["niet_mixen"].astype(bool)
    df["actief"] = df["actief"] | (df["aantal"] > 0)
    return df


# ---------- Optimaliseren ----------
if submitted:
    ed = _normalize_editor(edited)
    st.session_state["editor_df"] = ed.copy()

    needs = {row["soort"]: int(max(0, row["aantal"])) for _, row in ed.iterrows() if (bool(row["actief"]) or int(row.get("aantal", 0)) > 0)}
    per_type_min = {row["soort"]: float(row.get("min_klasse", 0.0) or 0.0) for _, row in ed.iterrows()}
    per_type_max = {row["soort"]: float(row.get("max_klasse", 0.0) or 0.0) for _, row in ed.iterrows()}
    no_mix = {row["soort"]: bool(row.get("niet_mixen", False)) for _, row in ed.iterrows()}

    problems = []
    if sum(needs.values()) <= 0:
        problems.append("Geen aantallen ingevuld (>0).")
    if goal_mode == "BUDGET" and (not budget_net or budget_net <= 0):
        problems.append("Geen (geldig) budget ingevuld.")

    if problems:
        st.error("Corrigeer de volgende punten:")
        for p in problems:
            st.write("• " + p)
        st.stop()

    try:
        options = []

        if goal_mode == "BUDGET":
            # BEST = hoogste klasse binnen budget
            best_df, best_total, best_avg = solve_max_quality_under_budget(
                df_work, needs, budget_net,
                min_class_by_type=per_type_min, max_class_by_type=per_type_max, no_mix_by_type=no_mix
            )
            # Goedkopere varianten op lagere klasse-drempel (−0.05 / −0.10)
            thr_A = max(1.0, best_avg - 0.05)
            thr_G = max(1.0, best_avg - 0.10)

            a_df, a_total, a_avg = solve_min_cost_with_quality(
                df_work, needs, thr_A,
                min_class_by_type=per_type_min, max_class_by_type=per_type_max, no_mix_by_type=no_mix, budget_cap=budget_net
            )
            g_df, g_total, g_avg = solve_min_cost_with_quality(
                df_work, needs, thr_G,
                min_class_by_type=per_type_min, max_class_by_type=per_type_max, no_mix_by_type=no_mix, budget_cap=budget_net
            )
            # Volgorde: GOED (budgetvriendelijk), ALT_A, BEST
            options = [
                {"role": "GOOD",  "name": "GOOD • BUDGETVRIENDELIJK (−0,10 klasse)", "result": g_df,  "total": g_total,  "avg": g_avg},
                {"role": "ALT_A", "name": "ALTERNATIEF A • ZUINIG (−0,05 klasse)",    "result": a_df,  "total": a_total,  "avg": a_avg},
                {"role": "ALT_B", "name": "BEST • MAX KWALITEIT BINNEN BUDGET",       "result": best_df,"total": best_total,"avg": best_avg},
            ]
            st.session_state["badge"] = f"Doel: hoogste kwaliteit binnen budget ({euro(budget_net,0)})"

        else:
            # Min. prijs bij >= target_avg + varianten +0.05 / +0.10
            g_df, g_total, g_avg = solve_min_cost_with_quality(
                df_work, needs, target_avg,
                min_class_by_type=per_type_min, max_class_by_type=per_type_max, no_mix_by_type=no_mix
            )
            a_df, a_total, a_avg = solve_min_cost_with_quality(
                df_work, needs, target_avg + 0.05,
                min_class_by_type=per_type_min, max_class_by_type=per_type_max, no_mix_by_type=no_mix
            )
            b_df, b_total, b_avg = solve_min_cost_with_quality(
                df_work, needs, target_avg + 0.10,
                min_class_by_type=per_type_min, max_class_by_type=per_type_max, no_mix_by_type=no_mix
            )
            options = [
                {"role": "GOOD",  "name": "GOOD • MINIMALE PRIJS (voldoet)", "result": g_df, "total": g_total, "avg": g_avg},
                {"role": "ALT_A", "name": "ALTERNATIEF A • klasse +0,05",     "result": a_df, "total": a_total, "avg": a_avg},
                {"role": "ALT_B", "name": "ALTERNATIEF B • klasse +0,10",     "result": b_df, "total": b_total, "avg": b_avg},
            ]
            st.session_state["badge"] = f"Doel: minimale prijs bij klasse ≥ {str(round(target_avg,2)).replace('.',',')}"

        st.session_state["gbb_options"] = options
        st.session_state["df_work"] = df_work
        st.session_state["klant_korting_value"] = float(klant_korting_pct)
        if goal_mode == "BUDGET":
            st.session_state["last_budget_shown"] = budget_net

        st.success("Opties berekend. Zie resultaten hieronder.")
        st.rerun()

    except Exception as e:
        st.error(f"Optimalisatie mislukte: {e}")


# ---------- Resultaten ----------
if "gbb_options" in st.session_state and st.session_state["gbb_options"]:
    st.subheader("Resultaten")
    st.caption(st.session_state.get("badge", ""))

    opts = st.session_state["gbb_options"]
    by_role = {o["role"]: o for o in opts}
    good = by_role.get("GOOD", opts[0])
    altA = by_role.get("ALT_A")
    altB = by_role.get("ALT_B")

    base_price = good["total"]  # referentie voor prijsverschil

    def _card(title, subtitle, total, avg, diff_vs_good=None):
        diff_html = ""
        if diff_vs_good is not None:
            sign = "+" if diff_vs_good > 0 else ("±" if abs(diff_vs_good) < 1e-6 else "")
            diff_html = f"<div style='color:#64748b;font-size:12px;'>Prijsverschil t.o.v. GOOD: {sign}{euro(diff_vs_good,0)}</div>"

        st.markdown(
            f"""
            <div style="
                background:#fff;border:1px solid #e5e7eb;border-radius:8px;
                padding:12px 14px;min-height: 160px;display:flex;flex-direction:column;gap:6px;">
              <h4 style="margin:0 0 4px 0;font-size:15px;letter-spacing:.02em">{title}</h4>
              <div style="color:#475569">{subtitle}</div>
              <div><b>Prijs = {euro(total, 0)}</b></div>
              <div><b>Gemiddelde klasse = {str(round(avg,2)).replace('.', ',')}</b></div>
              <div style="margin-top:auto;">{diff_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        _card(
            "GOOD — referentie",
            "Baseline voor prijsvergelijking.",
            good["total"], good["avg"]
        )
    with c2:
        if altA:
            _card(
                "ALTERNATIEF A",
                "Variant op het referentiescenario.",
                altA["total"], altA["avg"],
                altA["total"] - base_price,
            )
        else:
            st.info("Geen alternatief A beschikbaar.")
    with c3:
        if altB:
            _card(
                "ALTERNATIEF B",
                "Variant op het referentiescenario.",
                altB["total"], altB["avg"],
                altB["total"] - base_price,
            )
        else:
            st.info("Geen alternatief B beschikbaar.")

    st.divider()

    # Detailtabel van geselecteerde optie
    st.markdown("### Gekozen producten & aantallen")

    labels = ["GOOD — referentie", "ALTERNATIEF A", "ALTERNATIEF B"]
    existing = [l for l, k in zip(labels, ["GOOD", "ALT_A", "ALT_B"]) if k in by_role]
    sel_label = st.selectbox("Toon details van", existing, index=0)
    role_map = {"GOOD — referentie": "GOOD", "ALTERNATIEF A": "ALT_A", "ALTERNATIEF B": "ALT_B"}
    sel = by_role[role_map[sel_label]]

    sel_df = sel["result"].copy()
    sel_df.columns = sel_df.columns.str.lower()

    # veilig presenteren
    if "price_effective" not in sel_df.columns and "prijs" in sel_df.columns:
        sel_df["price_effective"] = sel_df["prijs"]
    if "price_buy_unit" not in sel_df.columns:
        sel_df["price_buy_unit"] = 0.0

    sel_df["aantal"] = pd.to_numeric(sel_df["aantal"], errors="coerce").fillna(0).astype(int)
    sel_df = sel_df[sel_df["aantal"] > 0].copy()

    sel_df["omzet_subtotaal"] = sel_df["aantal"].astype(float) * sel_df["price_effective"].astype(float)

    show = sel_df[["soort", "artikel", "merk", "aantal", "price_effective", "omzet_subtotaal"]].rename(
        columns={
            "soort": "Soort",
            "artikel": "Artikel",
            "merk": "Merk",
            "aantal": "Aantal",
            "price_effective": "Prijs/stuk (klant)",
            "omzet_subtotaal": "Subtotaal (klant)",
        }
    ).sort_values(["Soort", "Artikel"])

    st.dataframe(
        show.style.format({"Prijs/stuk (klant)": lambda x: euro(x, 0), "Subtotaal (klant)": lambda x: euro(x, 0)}),
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info("Vul de invoer in en klik op **Optimaliseren** om resultaten te zien.")

