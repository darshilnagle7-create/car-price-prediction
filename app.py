"""
app.py — Car Price Predictor
A polished Streamlit UI around the Linear / Ridge / Lasso Regression
models trained in train_model.py.
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

BUNDLE_PATH = "car_price_bundle.pkl"

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        #MainMenu, footer, header { visibility: hidden; }

        .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px; }

        .hero {
            background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
            border-radius: 20px;
            padding: 2.4rem 2.6rem;
            color: white;
            margin-bottom: 1.6rem;
            box-shadow: 0 12px 30px rgba(37, 117, 252, 0.25);
        }
        .hero h1 { font-size: 2.1rem; font-weight: 800; margin: 0 0 .4rem 0; }
        .hero p { font-size: 1.02rem; opacity: 0.92; margin: 0; }

        .panel {
            background: var(--background-color, #ffffff);
            border: 1px solid rgba(120,120,120,0.15);
            border-radius: 16px;
            padding: 1.5rem 1.6rem;
            box-shadow: 0 4px 16px rgba(0,0,0,0.05);
            margin-bottom: 1.2rem;
        }
        .panel h3 { margin-top: 0; }

        .price-card {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            border-radius: 18px;
            padding: 1.8rem 2rem;
            color: white;
            text-align: center;
            box-shadow: 0 10px 26px rgba(17, 153, 142, 0.28);
        }
        .price-card .label { font-size: 0.95rem; opacity: 0.9; font-weight: 500; letter-spacing: 0.02em; }
        .price-card .value { font-size: 2.8rem; font-weight: 800; margin: 0.2rem 0; }
        .price-card .sub { font-size: 0.85rem; opacity: 0.85; }

        .metric-chip {
            display: inline-block;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            background: rgba(106, 17, 203, 0.1);
            color: #6a11cb;
            font-weight: 600;
            font-size: 0.85rem;
            margin-right: 0.4rem;
        }

        div.stButton > button {
            background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.7rem 1.2rem;
            font-weight: 700;
            font-size: 1rem;
            width: 100%;
            box-shadow: 0 6px 16px rgba(37, 117, 252, 0.3);
            transition: transform 0.05s ease-in-out;
        }
        div.stButton > button:hover { transform: translateY(-1px); }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Load model bundle
# --------------------------------------------------------------------------
@st.cache_resource
def load_bundle(path: str):
    return joblib.load(path)


try:
    bundle = load_bundle(BUNDLE_PATH)
except FileNotFoundError:
    st.markdown(
        """
        <div class="hero">
            <h1>🚗 Car Price Predictor</h1>
            <p>Model bundle not found.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error(
        f"Couldn't find **{BUNDLE_PATH}**. Run `python train_model.py` first "
        "to generate it, then place the resulting file next to `app.py` "
        "before deploying."
    )
    st.stop()

models = bundle["models"]
encoders = bundle["encoders"]
feature_order = bundle["feature_order"]
results = bundle["results"]
best_model_name = bundle["best_model"]
ranges = bundle.get("numeric_ranges", {})

# --------------------------------------------------------------------------
# Hero header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🚗 Car Price Predictor</h1>
        <p>Estimate a used car's market price with Linear, Ridge & Lasso Regression —
        trained on real second-hand car sales data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar — model choice + performance
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_name = st.selectbox(
        "Prediction model",
        options=list(models.keys()),
        index=list(models.keys()).index(best_model_name),
        help="All three were trained on the same data — pick one to see how its estimate differs.",
    )
    if model_name == best_model_name:
        st.success(f"⭐ {model_name} had the best R² on the test set.")

    st.markdown("---")
    st.markdown("### 📊 Model performance")
    perf_df = pd.DataFrame(results).T.rename(
        columns={"r2": "R²", "mae": "MAE (log price)", "rmse": "RMSE (log price)"}
    )
    st.dataframe(
        perf_df.style.format({"R²": "{:.3f}", "MAE (log price)": "{:.3f}", "RMSE (log price)": "{:.3f}"}),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown(
        "### ℹ️ About\n"
        "Predictions are made on **log(price)** and converted back to actual "
        "price. Categorical fields (brand, body, engine type, registration) "
        "are label-encoded exactly as in training."
    )

# --------------------------------------------------------------------------
# Main layout — inputs | prediction
# --------------------------------------------------------------------------
col_input, col_result = st.columns([1.15, 1], gap="large")

with col_input:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 🧾 Car details")

    c1, c2 = st.columns(2)
    with c1:
        brand = st.selectbox("Brand", options=list(encoders["Brand"].classes_))
        body = st.selectbox("Body type", options=list(encoders["Body"].classes_))
        engine_type = st.selectbox("Engine type", options=list(encoders["Engine Type"].classes_))
        registration = st.radio("Registered?", options=list(encoders["Registration"].classes_), horizontal=True)

    with c2:
        y_min, y_max = ranges.get("Year", (1970, 2016))
        year = st.slider("Year", min_value=int(y_min), max_value=max(int(y_max), 2024), value=int((y_min + y_max) // 2))

        m_min, m_max = ranges.get("Mileage", (0, 300000))
        mileage = st.slider("Mileage (in thousand km)", min_value=int(m_min), max_value=int(m_max), value=int((m_min + m_max) // 2))

        e_min, e_max = ranges.get("EngineV", (0.6, 6.5))
        engine_v = st.slider("Engine volume (L)", min_value=float(e_min), max_value=float(e_max), value=round((e_min + e_max) / 2, 1), step=0.1)

    predict_clicked = st.button("🔮 Predict price")
    st.markdown("</div>", unsafe_allow_html=True)

with col_result:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 💰 Estimated price")

    if predict_clicked:
        row = {
            "Brand": encoders["Brand"].transform([brand])[0],
            "Body": encoders["Body"].transform([body])[0],
            "Mileage": mileage,
            "EngineV": engine_v,
            "Engine Type": encoders["Engine Type"].transform([engine_type])[0],
            "Registration": encoders["Registration"].transform([registration])[0],
            "Year": year,
        }
        X_input = pd.DataFrame([row])[feature_order]

        model = models[model_name]
        log_price = model.predict(X_input)[0]
        price = float(np.exp(log_price))

        st.markdown(
            f"""
            <div class="price-card">
                <div class="label">{model_name} estimate</div>
                <div class="value">${price:,.0f}</div>
                <div class="sub">{brand} · {body} · {year} · {engine_v}L {engine_type}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**All models on this car:**")

        all_preds = {}
        for name, m in models.items():
            all_preds[name] = float(np.exp(m.predict(X_input)[0]))

        fig = go.Figure(
            go.Bar(
                x=list(all_preds.keys()),
                y=list(all_preds.values()),
                marker_color=["#2575fc" if n != model_name else "#11998e" for n in all_preds],
                text=[f"${v:,.0f}" for v in all_preds.values()],
                textposition="outside",
            )
        )
        fig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="Predicted price ($)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Fill in the car details on the left and click **Predict price**.")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; opacity:0.6; font-size:0.85rem; margin-top:1.5rem;">
        Built with Streamlit · Linear / Ridge / Lasso Regression
    </div>
    """,
    unsafe_allow_html=True,
)
