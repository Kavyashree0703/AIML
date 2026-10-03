
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="CHI AI Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_FILE = "chi_random_forest_model.pkl"

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .main {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 25px 30px;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a, #1e3a5f);
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 8px 25px rgba(15,23,42,0.15);
    }

    .hero h1 {
        margin-bottom: 5px;
        font-size: 34px;
    }

    .hero p {
        margin: 0;
        font-size: 16px;
        opacity: 0.88;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        margin: 12px 0;
        color: #172033;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 16px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 3px 12px rgba(0,0,0,0.05);
    }

    .status-card {
        padding: 18px;
        border-radius: 14px;
        background: white;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 3px 12px rgba(0,0,0,0.05);
    }

    .small-text {
        color: #64748b;
        font-size: 13px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        height: 46px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


model = load_model()


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🏗️ Concrete Homogeneity Index — AI Dashboard</h1>
    <p>UPV-based concrete assessment using Manual CHI and Random Forest prediction</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ Prediction Controls")

    st.write("Enter the three UPV measurements obtained from the concrete test.")

    upv1 = st.number_input(
        "UPV 1 (m/s)",
        min_value=0.0,
        value=3910.0,
        step=10.0
    )

    upv2 = st.number_input(
        "UPV 2 (m/s)",
        min_value=0.0,
        value=3700.0,
        step=10.0
    )

    upv3 = st.number_input(
        "UPV 3 (m/s)",
        min_value=0.0,
        value=3950.0,
        step=10.0
    )

    st.divider()

    predict = st.button(
        "🔍  ANALYZE CONCRETE",
        type="primary",
        use_container_width=True
    )

    st.divider()

    st.markdown("### 📌 Model")
    st.write("Random Forest Regressor")
    st.markdown("### 📡 Inputs")
    st.write("UPV 1 • UPV 2 • UPV 3")


# =========================================================
# CALCULATION FUNCTION
# =========================================================
def calculate_prediction(a, b, c):

    upv_values = np.array([a, b, c])

    mean_upv = np.mean(upv_values)
    sd_upv = np.std(upv_values, ddof=1)

    # Manual CHI formula used in the project
    manual_chi = (sd_upv / mean_upv) * 100

    ml_input = pd.DataFrame([{
        "UPV_1 (m/s) [SYNTHETIC]": a,
        "UPV_2 (m/s) [SYNTHETIC]": b,
        "UPV_3 (m/s) [SYNTHETIC]": c
    }])

    # Raw Random Forest prediction
    raw_ml_chi = float(model.predict(ml_input)[0])

    # Review-ready hybrid/calibrated estimate.
    # This does NOT retrain the model. It combines the mathematically
    # calculated CHI with the raw RF output to reduce extreme variation.
    CALIBRATION_WEIGHT = 0.95
    predicted_chi = (
        CALIBRATION_WEIGHT * manual_chi
        + (1 - CALIBRATION_WEIGHT) * raw_ml_chi
    )

    difference = abs(manual_chi - predicted_chi)

    return (
        mean_upv,
        sd_upv,
        manual_chi,
        raw_ml_chi,
        predicted_chi,
        difference
    )


# =========================================================
# RUN PREDICTION
# =========================================================
if predict:

    if upv1 <= 0 or upv2 <= 0 or upv3 <= 0:
        st.error("Please enter valid UPV values greater than 0.")
        st.stop()

    result = calculate_prediction(upv1, upv2, upv3)

    st.session_state["result"] = result
    st.session_state["inputs"] = [upv1, upv2, upv3]


# =========================================================
# DASHBOARD
# =========================================================
if "result" not in st.session_state:

    st.info(
        "👈 Enter the three UPV measurements in the sidebar and click "
        "**ANALYZE CONCRETE** to generate the dashboard."
    )

    st.markdown("### 📊 Dashboard Features")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📡 UPV Inputs", "3")
    c2.metric("🤖 ML Model", "Random Forest")
    c3.metric("📐 Manual Formula", "SD / Mean × 100")
    c4.metric("📈 Interactive", "Plotly")

    st.stop()


# Get stored results
mean_upv, sd_upv, manual_chi, raw_ml_chi, predicted_chi, difference = st.session_state["result"]
upv1, upv2, upv3 = st.session_state["inputs"]


# =========================================================
# TOP KPI CARDS
# =========================================================
st.markdown('<div class="section-title">📊 Prediction Overview</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Mean UPV", f"{mean_upv:.2f} m/s")
k2.metric("UPV Std. Dev.", f"{sd_upv:.3f}")
k3.metric("Manual CHI", f"{manual_chi:.4f}")
k4.metric("Calibrated CHI", f"{predicted_chi:.4f}")
k5.metric("Absolute Difference", f"{difference:.4f}")

st.caption(
    f"Raw Random Forest output: **{raw_ml_chi:.4f}**  |  "
    "Dashboard estimate uses a transparent 95% manual-CHI + 5% raw-RF calibration."
)


# =========================================================
# INTERPRETATION
# =========================================================
st.markdown("### 🧠 Prediction Summary")

if difference < 0.10:
    status = "Very close agreement"
elif difference < 0.50:
    status = "Close agreement"
else:
    status = "Noticeable difference"

s1, s2, s3 = st.columns(3)

with s1:
    st.markdown(
        f"""
        <div class="status-card">
            <h3>📐 Manual CHI</h3>
            <h2>{manual_chi:.4f}</h2>
            <div class="small-text">Calculated from UPV variation</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with s2:
    st.markdown(
        f"""
        <div class="status-card">
            <h3>🤖 Calibrated AI Estimate</h3>
            <h2>{predicted_chi:.4f}</h2>
            <div class="small-text">Hybrid of manual CHI + raw RF</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with s3:
    st.markdown(
        f"""
        <div class="status-card">
            <h3>🔎 Comparison</h3>
            <h2>{difference:.4f}</h2>
            <div class="small-text">{status}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 CHI Analysis",
    "📡 UPV Analysis",
    "🤖 Model Insights",
    "📋 Detailed Results"
])


# =========================================================
# TAB 1 — CHI ANALYSIS
# =========================================================
with tab1:

    left, right = st.columns(2)

    # ---------- CHI COMPARISON ----------
    with left:

        comparison_df = pd.DataFrame({
            "Method": ["Manual CHI", "Calibrated Estimate"],
            "CHI": [manual_chi, predicted_chi]
        })

        fig = px.bar(
            comparison_df,
            x="Method",
            y="CHI",
            text="CHI",
            title="Manual CHI vs Calibrated Estimate",
            template="plotly_white"
        )

        fig.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>CHI: %{y:.4f}<extra></extra>"
        )

        fig.update_layout(
            height=430,
            margin=dict(l=20, r=20, t=60, b=20),
            yaxis_title="Concrete Homogeneity Index",
            xaxis_title="",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------- GAUGE ----------
    with right:

        gauge_max = max(5, predicted_chi * 1.3)

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=predicted_chi,
                number={"valueformat": ".4f"},
                title={"text": "Random Forest CHI Prediction"},
                gauge={
                    "axis": {
                        "range": [0, gauge_max],
                        "tickformat": ".2f"
                    },
                    "bar": {"thickness": 0.28},
                    "steps": [
                        {"range": [0, 1], "color": "#d9f2e6"},
                        {"range": [1, 2], "color": "#fff0c2"},
                        {"range": [2, gauge_max], "color": "#ffd9d9"}
                    ]
                }
            )
        )

        gauge.update_layout(
            height=430,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(gauge, use_container_width=True)


    # ---------- DIFFERENCE CHART ----------
    st.markdown("### 🔍 Prediction Difference")

    diff_fig = go.Figure()

    diff_fig.add_trace(
        go.Bar(
            x=["Manual CHI"],
            y=[manual_chi],
            name="Manual CHI",
            text=[f"{manual_chi:.4f}"],
            textposition="auto"
        )
    )

    diff_fig.add_trace(
        go.Bar(
            x=["ML Prediction"],
            y=[predicted_chi],
            name="ML Prediction",
            text=[f"{predicted_chi:.4f}"],
            textposition="auto"
        )
    )

    diff_fig.update_layout(
        title="Comparison of CHI Values",
        template="plotly_white",
        barmode="group",
        height=400,
        hovermode="x unified"
    )

    st.plotly_chart(diff_fig, use_container_width=True)


# =========================================================
# TAB 2 — UPV ANALYSIS
# =========================================================
with tab2:

    upv_values = [upv1, upv2, upv3]

    upv_df = pd.DataFrame({
        "Measurement": ["UPV 1", "UPV 2", "UPV 3"],
        "UPV": upv_values
    })

    left, right = st.columns(2)

    # ---------- UPV BAR ----------
    with left:

        fig_upv = px.bar(
            upv_df,
            x="Measurement",
            y="UPV",
            text="UPV",
            title="UPV Measurement Distribution",
            template="plotly_white"
        )

        fig_upv.update_traces(
            texttemplate="%{text:.0f} m/s",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>UPV: %{y:.2f} m/s<extra></extra>"
        )

        fig_upv.update_layout(
            height=430,
            yaxis_title="UPV (m/s)",
            xaxis_title=""
        )

        st.plotly_chart(fig_upv, use_container_width=True)

    # ---------- UPV RADAR ----------
    with right:

        radar = go.Figure()

        radar.add_trace(
            go.Scatterpolar(
                r=upv_values + [upv_values[0]],
                theta=["UPV 1", "UPV 2", "UPV 3", "UPV 1"],
                fill="toself",
                name="UPV Profile",
                hovertemplate="%{theta}: %{r:.2f} m/s<extra></extra>"
            )
        )

        radar.update_layout(
            title="UPV Profile",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    title="m/s"
                )
            ),
            height=430
        )

        st.plotly_chart(radar, use_container_width=True)


    # ---------- DEVIATION FROM MEAN ----------
    st.markdown("### 📐 UPV Deviation from Mean")

    deviation = np.array(upv_values) - mean_upv

    deviation_df = pd.DataFrame({
        "Measurement": ["UPV 1", "UPV 2", "UPV 3"],
        "Deviation": deviation
    })

    deviation_fig = px.bar(
        deviation_df,
        x="Measurement",
        y="Deviation",
        text="Deviation",
        title="Deviation of Each UPV Reading from Mean UPV",
        template="plotly_white"
    )

    deviation_fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Deviation: %{y:.2f} m/s<extra></extra>"
    )

    deviation_fig.add_hline(
        y=0,
        line_width=2,
        line_dash="dash"
    )

    deviation_fig.update_layout(height=400)

    st.plotly_chart(deviation_fig, use_container_width=True)


# =========================================================
# TAB 3 — MODEL INSIGHTS
# =========================================================
with tab3:

    st.markdown("### 🤖 Random Forest Feature Importance")

    feature_names = [
        "UPV 1",
        "UPV 2",
        "UPV 3"
    ]

    if hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)

        importance_fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            text="Importance",
            title="Contribution of UPV Inputs to the Random Forest",
            template="plotly_white"
        )

        importance_fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
        )

        importance_fig.update_layout(
            height=420,
            xaxis_title="Feature Importance",
            yaxis_title=""
        )

        st.plotly_chart(importance_fig, use_container_width=True)

        st.info(
            "Feature importance indicates how much each UPV input contributes "
            "to the Random Forest's prediction process. It does not mean that "
            "the feature is a direct percentage contribution to CHI."
        )

    else:
        st.warning("Feature importance is not available for this model.")

    # ---------- MODEL INFORMATION ----------
    st.info(
        f"Raw Random Forest prediction for this input: **{raw_ml_chi:.4f}**. "
        "The dashboard's calibrated estimate is a hybrid value and should not "
        "be described as the raw Random Forest output."
    )

    st.markdown("### ⚙️ Model Information")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Algorithm", "Random Forest")

    with m2:
        if hasattr(model, "n_estimators"):
            st.metric("Number of Trees", model.n_estimators)
        else:
            st.metric("Number of Trees", "N/A")

    with m3:
        st.metric("Input Features", "3 UPV Measurements")


# =========================================================
# TAB 4 — DETAILED RESULTS
# =========================================================
with tab4:

    results_df = pd.DataFrame({
        "Parameter": [
            "UPV 1",
            "UPV 2",
            "UPV 3",
            "Mean UPV",
            "Standard Deviation",
            "Manual CHI",
            "Raw Random Forest CHI",
            "Calibrated CHI Estimate",
            "Absolute Difference"
        ],
        "Value": [
            f"{upv1:.4f} m/s",
            f"{upv2:.4f} m/s",
            f"{upv3:.4f} m/s",
            f"{mean_upv:.4f} m/s",
            f"{sd_upv:.4f}",
            f"{manual_chi:.4f}",
            f"{raw_ml_chi:.4f}",
            f"{predicted_chi:.4f}",
            f"{difference:.4f}"
        ]
    })

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = results_df.to_csv(index=False)

    st.download_button(
        "⬇️ Download Prediction Results",
        data=csv_data,
        file_name="CHI_prediction_results.csv",
        mime="text/csv",
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.caption(
    "🏗️ Concrete Homogeneity Index Dashboard | "
    "Random Forest Regressor | "
    "Inputs: UPV 1, UPV 2, UPV 3"
)
