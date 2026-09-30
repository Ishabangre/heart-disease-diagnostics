"""Streamlit app.  Run from project root:  streamlit run app/main.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app.dashboard import render_dashboard
from src.explain import (build_shap_explainer, default_grid, load_background,
                         risk_curve, shap_explain, shap_summary_png,
                         shap_waterfall_png)
from src.predict import build_input_frame, load_artifacts, predict
from src.recommendations import build_advice
from src.preprocessing import RAW_FEATURES

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide",
                   initial_sidebar_state="collapsed")

MODEL_PATH = ROOT / "models" / "heart_model.pkl"
DEFAULT_PATIENT = dict(age=55, sex=1, cp=0, trestbps=130, chol=240, fbs=0, restecg=1,
                       thalach=150, exang=0, oldpeak=1.0, slope=1, ca=0, thal=2)


# ---------------- cached heavy objects ----------------
@st.cache_resource
def get_explainer():
    model, _ = load_artifacts()
    return build_shap_explainer(model, load_background())


@st.cache_data(show_spinner=False)
def get_global_shap_png() -> bytes:
    # PNG bytes (not a Figure) -> never turns blank on rerun
    exp = shap_explain(get_explainer(), load_background())
    return shap_summary_png(exp)


# ---------------- shared patient form ----------------
def patient_form(prefix: str, d: dict, sliders: bool = False) -> dict:
    """Same 13 inputs for Prediction (number boxes) and What-If (sliders)."""
    num = st.slider if sliders else st.number_input

    def sel(label, key, options, fmt=None):
        return st.selectbox(label, options, index=options.index(d[key]),
                            key=f"{prefix}_{key}", format_func=fmt or str)

    c1, c2 = st.columns(2)
    with c1:
        age = num("Age", min_value=20, max_value=100, value=int(d["age"]), step=1, key=f"{prefix}_age")
        sex = sel("Sex", "sex", [1, 0], lambda x: "Male" if x else "Female")
        cp = sel("Chest pain type (cp)", "cp", [0, 1, 2, 3])
        trestbps = num("Resting blood pressure", min_value=80, max_value=220,
                       value=int(d["trestbps"]), step=1, key=f"{prefix}_trestbps")
        chol = num("Cholesterol (mg/dl)", min_value=100, max_value=600,
                   value=int(d["chol"]), step=1, key=f"{prefix}_chol")
        fbs = sel("Fasting blood sugar > 120", "fbs", [0, 1])
        restecg = sel("Resting ECG (restecg)", "restecg", [0, 1, 2])
    with c2:
        thalach = num("Max heart rate (thalach)", min_value=60, max_value=220,
                      value=int(d["thalach"]), step=1, key=f"{prefix}_thalach")
        exang = sel("Exercise induced angina", "exang", [0, 1])
        oldpeak = num("ST depression (oldpeak)", min_value=0.0, max_value=7.0,
                      value=float(d["oldpeak"]), step=0.1, key=f"{prefix}_oldpeak")
        slope = sel("Slope of ST segment", "slope", [0, 1, 2])
        ca = sel("Major vessels (ca)", "ca", [0, 1, 2, 3, 4])
        thal = sel("Thal", "thal", [0, 1, 2, 3])

    return dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs,
                restecg=restecg, thalach=thalach, exang=exang, oldpeak=oldpeak,
                slope=slope, ca=ca, thal=thal)


def need_model():
    if not MODEL_PATH.exists():
        st.error("Model not found. Run `python -m src.train` first.")
        st.stop()


def show_result(res):
    if res["prediction"]:
        st.error(f"⚠️ {res['message']}  (risk: {res['probability']:.1%})")
    else:
        st.success(f"✅ {res['message']}  (risk: {res['probability']:.1%})")
    st.progress(min(max(res["probability"], 0.0), 1.0))


# ---------------- PAGE 1: PREDICTION ----------------
def prediction_page():
    st.title("🩺 Heart Disease Prediction")
    need_model()

    patient = patient_form("pred", st.session_state.get("patient", DEFAULT_PATIENT))

    if st.button("Predict", type="primary"):
        st.session_state["patient"] = patient      # used as baseline in What-If
        res = predict(patient)
        show_result(res)

        with st.expander("🔍 Why this result? (SHAP explanation)", expanded=True):
            with st.spinner("Computing SHAP values..."):
                exp = shap_explain(get_explainer(), build_input_frame(patient))
                st.image(shap_waterfall_png(exp[0]))
            st.caption("Red bars push the risk UP, blue bars push it DOWN. "
                       "E[f(X)] = average risk of the training patients, f(x) = this patient's risk.")
        st.caption("Educational project only. Not a medical diagnosis.")


# ---------------- PAGE 2: WHAT-IF ----------------
def whatif_page():
    st.title("🎛️ What-If Analysis")
    st.write("Move the sliders and watch the risk change **live**. "
             "Baseline = your last prediction (or a default patient).")
    need_model()

    baseline = st.session_state.get("patient", DEFAULT_PATIENT)

    if st.button("↩️ Reset sliders to baseline"):
        for k, v in baseline.items():
            st.session_state[f"wi_{k}"] = v
        st.rerun()

    left, right = st.columns([3, 2])
    with left:
        new = patient_form("wi", baseline, sliders=True)

    base_res, new_res = predict(baseline), predict(new)
    delta = new_res["probability"] - base_res["probability"]

    with right:
        st.subheader("Risk")
        m1, m2 = st.columns(2)
        m1.metric("Baseline", f"{base_res['probability']:.1%}")
        m2.metric("What-if", f"{new_res['probability']:.1%}",
                  delta=f"{delta:+.1%}", delta_color="inverse")
        show_result(new_res)

        changed = {k: (baseline[k], new[k]) for k in RAW_FEATURES if baseline[k] != new[k]}
        if changed:
            st.write("**Changed:** " + ", ".join(f"`{k}` {a} → {b}" for k, (a, b) in changed.items()))
        else:
            st.caption("No changes yet. Move a slider.")

    # ---- ICE curve for this patient ----
    st.divider()
    st.subheader("📈 Risk curve for this patient")
    model, cols = load_artifacts()
    ref = load_background()
    numeric = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    feat = st.selectbox("Feature to vary", numeric + [f for f in RAW_FEATURES if f not in numeric])
    grid = default_grid(ref, feat)
    curve = risk_curve(model, cols, pd.DataFrame([new]), feat, grid)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(curve[feat], curve["risk"], color="#c0392b", lw=2)
    ax.scatter([new[feat]], [new_res["probability"]], color="black", zorder=3, label="current value")
    ax.axhline(new_res["threshold"], color="gray", ls="--", lw=1, label=f"threshold {new_res['threshold']:.2f}")
    ax.set_xlabel(feat)
    ax.set_ylabel("Predicted risk")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Everything else is kept fixed. The curve shows how risk would change for THIS patient.")

# ---------------- PAGE: HEALTH ADVICE ----------------
def advice_page():
    st.title("🌿 Health Advice")
    need_model()

    if "patient" not in st.session_state:
        st.info("👈 First go to the **Prediction** page, enter the patient details and press **Predict**. "
                "Your personal advice will appear here.")
        st.stop()

    patient = st.session_state["patient"]
    res = predict(patient)
    adv = build_advice(patient, res["probability"], res["threshold"])
    band = adv["band"]

    # ---- risk header ----
    c1, c2 = st.columns([1, 3])
    c1.metric("Predicted risk", f"{res['probability']:.1%}")
    box = [st.success, st.warning, st.error][band["level"]]
    with c2:
        box(f"{band['emoji']} **{band['label']} risk.** {band['summary']}")

    if adv["urgent"]:
        st.error("⚠️ **Please see a doctor soon**, because: " + "; ".join(adv["urgent"]) + ".")

    # ---- personalised findings ----
    st.subheader("🔎 What your numbers say")
    if not adv["findings"]:
        st.success("None of your individual values stand out. Keep up your healthy routine and do yearly check-ups.")
    for f in adv["findings"]:
        with st.expander(f"{f['icon']} {f['title']}: {f['value']}", expanded=True):
            st.write(f["why"])
            for a in f["actions"]:
                st.markdown(f"- {a}")

    # ---- daily routine ----
    st.subheader("🗓️ A simple daily routine")
    cols = st.columns(3)
    for col, (title, items) in zip(cols, adv["routine"].items()):
        with col:
            st.markdown(f"**{title}**")
            for it in items:
                st.markdown(f"- {it}")

    # ---- detail tabs ----
    st.subheader("📚 Details")
    t_food, t_ex, t_life, t_mon = st.tabs(["🥗 Food", "🏃 Exercise", "🧘 Lifestyle", "🩺 Monitoring"])
    with t_food:
        a, b = st.columns(2)
        with a:
            st.markdown("**✅ Eat more**")
            for it in adv["food_do"]:
                st.markdown(f"- {it}")
        with b:
            st.markdown("**❌ Cut down**")
            for it in adv["food_avoid"]:
                st.markdown(f"- {it}")
    with t_ex:
        ex = adv["exercise"]
        if ex["needs_clearance"]:
            st.warning("Get a doctor's clearance before regular exercise.")
        for it in ex["plan"]:
            st.markdown(f"- {it}")
        lo, hi = ex["hr_zone"]
        st.info(f"Rough moderate-intensity heart rate zone: **{lo}-{hi} beats/min**. {ex['hr_note']}")
    with t_life:
        for it in adv["lifestyle"]:
            st.markdown(f"- {it}")
    with t_mon:
        for it in adv["monitoring"]:
            st.markdown(f"- {it}")

    # ---- checklist ----
    st.subheader("✅ Today's checklist")
    done = sum(st.checkbox(item, key=f"chk_{i}") for i, item in enumerate(adv["checklist"]))
    st.progress(done / len(adv["checklist"]), text=f"{done} of {len(adv['checklist'])} done today")

    # ---- warning signs ----
    st.subheader("🚨 When to get emergency help")
    st.error("**Call emergency services immediately if you notice:**\n\n"
             + "\n".join(f"- {w}" for w in adv["warning_signs"]))
    st.markdown("**What to do:**")
    for i, step in enumerate(adv["emergency_steps"], 1):
        st.markdown(f"{i}. {step}")

    st.caption("General lifestyle information for education only. It is not a diagnosis or medical advice. "
               "Please talk to your doctor before changing medicines or starting a new exercise programme.")

    
# ---------------- PAGE 3: EXPLAINABILITY ----------------
def explain_page():
    st.title("🔍 Model Explainability")
    need_model()

    st.subheader("SHAP summary (global)")
    saved = ROOT / "outputs" / "graphs" / "shap_summary.png"
    if saved.exists():                       # pre-computed by `python -m src.explain` (fast, good for deployment)
        st.image(str(saved))
    else:
        with st.spinner("Computing SHAP values (first time only)..."):
            st.image(get_global_shap_png())
    st.caption("Each dot is a patient. Right = pushes risk up. Colour = feature value (red high, blue low).")

    st.subheader("Partial dependence: average risk vs one feature")
    model, cols = load_artifacts()
    ref = load_background()
    feat = st.selectbox("Feature", RAW_FEATURES, index=RAW_FEATURES.index("age"))
    curve = risk_curve(model, cols, ref, feat, default_grid(ref, feat))
    st.line_chart(curve.set_index(feat)["risk"])
    st.caption(f"Average predicted risk of {len(ref)} training patients when `{feat}` is set to each value.")

    pdp = ROOT / "outputs" / "graphs" / "partial_dependence.png"
    if pdp.exists():
        st.image(str(pdp))


# ---------------- PAGE 4: DASHBOARD ----------------
def dashboard_page():
    render_dashboard()

# ---------------- PAGE 0: HOME (front page: heading + Start button only) ----------------
HOME_HTML = """<style>
.hero {display:flex; flex-direction:column; align-items:center; justify-content:center;
 min-height: 55vh; text-align:center;}
.hero h1 {font-size: clamp(2.4rem, 7vw, 4.8rem); font-weight: 800; letter-spacing: 1px; margin: 0;}
.hero .grad {background: linear-gradient(90deg,#ff6b6b,#e74c3c,#c0392b); -webkit-background-clip: text;
 background-clip: text; color: transparent;}
.hero .beat {display:inline-block; animation: beat 1.3s infinite; transform-origin: center;}
@keyframes beat {
 0%, 100% {transform: scale(1);}
 15% {transform: scale(1.25);}
 30% {transform: scale(1);}
 45% {transform: scale(1.18);}
}
</style>
<div class="hero"><h1><span class="grad">Heart Disease Diagnostics</span> <span class="beat">🫀</span></h1></div>"""
def home_page():
    st.markdown(HOME_HTML, unsafe_allow_html=True)

    # centred Start button
    _, mid, _ = st.columns([2, 1.2, 2])
    with mid:
        if st.button("Start Diagnosis", type="primary"):
            st.session_state["page"] = "Prediction"
            st.rerun()

# ---------------- BUTTON NAVIGATION (no sidebar) ----------------
PAGES = {
    "Home": ("🏠", home_page),
    "Prediction": ("🩺", prediction_page),
    "What-If": ("🎛️", whatif_page),
    "Health Advice": ("🌿", advice_page),
    "Explainability": ("🔍", explain_page),
    "Dashboard": ("📊", dashboard_page),
}

# hide the sidebar completely
st.markdown(
    """<style>
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {display: none;}
    </style>""",
    unsafe_allow_html=True,
)

if "page" not in st.session_state:
    st.session_state["page"] = "Home"


def nav_bar():
    # Home is only the front page: it is NOT shown in the nav bar
    items = [(n, v) for n, v in PAGES.items() if n != "Home"]
    cols = st.columns(len(items))
    for col, (name, (icon, _)) in zip(cols, items):
        active = st.session_state["page"] == name
        if col.button(f"{icon} {name}", key=f"nav_{name}",
                      type="primary" if active else "secondary"):
            st.session_state["page"] = name
            st.rerun()
    st.divider()


# page buttons appear only AFTER "Start Diagnosis" (hidden on the Home page)
if st.session_state["page"] != "Home":
    nav_bar()
PAGES[st.session_state["page"]][1]()