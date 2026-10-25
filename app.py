from pathlib import Path
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "datasets"
IMAGE_DIR = BASE_DIR / "images"

st.set_page_config(
    page_title="LungAI | Clinical Intelligence",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CLINICAL NAVY + TEAL THEME
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #0B1120;
        --panel: #121C2D;
        --panel-light: #172437;
        --border: #27364A;
        --teal: #2DD4BF;
        --text: #F1F5F9;
        --muted: #9AAAC0;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, 0.9);
    }

    [data-testid="stSidebar"] {
        background: #0E1727;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -0.5px;
    }

    p, li, label {
        color: #C4CFDD;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 0 0 8px 0;
    }

    .brand-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        background: #103A3C;
        color: var(--teal);
        width: 46px;
        height: 46px;
        border-radius: 14px;
        font-size: 25px;
    }

    .brand-name {
        color: #F8FAFC;
        font-size: 23px;
        font-weight: 800;
        font-family: 'Manrope', sans-serif;
    }

    .eyebrow {
        color: var(--teal);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .muted {
        color: var(--muted);
        font-size: 13px;
    }

    .hero {
        background: linear-gradient(120deg, #132638, #102B32);
        border: 1px solid #24504F;
        border-radius: 20px;
        padding: 30px;
        margin: 8px 0 24px 0;
    }

    .hero h1 {
        font-size: clamp(27px, 4vw, 42px);
        line-height: 1.2;
        margin: 10px 0;
    }

    .hero p {
        max-width: 650px;
        line-height: 1.7;
        color: #B7C9D7;
    }

    .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 14px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: var(--text);
        font-size: 25px;
        font-weight: 800;
        margin-top: 8px;
        font-family: 'Manrope', sans-serif;
    }

    .metric-note {
        color: var(--teal);
        font-size: 12px;
        margin-top: 5px;
    }

    .section-label {
        color: var(--teal);
        text-transform: uppercase;
        letter-spacing: 1.6px;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .stButton > button {
        background: #0F9488;
        color: white;
        border: 1px solid #14B8A6;
        border-radius: 10px;
        font-weight: 700;
        min-height: 43px;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        background: #0D766E;
        border-color: #5EEAD4;
        color: white;
    }

    .stDownloadButton > button {
        border-radius: 10px;
    }

    [data-testid="stFileUploader"] {
        background: #111D2D;
        border: 1px dashed #426174;
        border-radius: 14px;
        padding: 12px;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid var(--border);
        border-radius: 10px;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background: #111D2D;
        border-color: #34445B;
        border-radius: 9px;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: var(--border);
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 768px) {
        .hero {
            padding: 20px;
        }
        .metric-value {
            font-size: 21px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL LOADING — KEEP ORIGINAL MODEL FILES
# =========================================================

@st.cache_resource
def load_ml_model():
    model_path = MODEL_DIR / "final_model.sav"

    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)


@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_DIR / "keras_model.h5")


# Load the existing tabular ML model.
try:
    cancer_model = load_ml_model()
    model_load_error = None
except Exception as exc:
    cancer_model = None
    model_load_error = str(exc)


# =========================================================
# SESSION HISTORY
# =========================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


def add_history(model_name, filename, prediction, score=None):
    """Store results in the current Streamlit session."""
    st.session_state.prediction_history.insert(
        0,
        {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Model": model_name,
            "File / Input": filename,
            "Prediction": str(prediction),
            "Score": (
                f"{score:.2%}" if score is not None else "Not available"
            ),
        },
    )

    # Keep the latest 100 results in memory.
    st.session_state.prediction_history = (
        st.session_state.prediction_history[:100]
    )


def render_metric(label, value, note=""):
    st.markdown(
        f"""
        <div class="panel">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_heading(eyebrow, title, description):
    st.markdown(
        f"""
        <div class="section-label">{eyebrow}</div>
        <h1 style="margin-top:0">{title}</h1>
        <p class="muted">{description}</p>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🫁</div>
            <div>
                <div class="brand-name">LungAI</div>
                <div class="muted">Clinical Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown('<div class="eyebrow">WORKSPACE</div>',
                unsafe_allow_html=True)

    selection = option_menu(
        menu_title=None,
        options=[
            "Dashboard",
            "Scan Analysis",
            "ML Prediction",
            "Dataset Explorer",
            "History",
        ],
        icons=[
            "grid-1x2",
            "activity",
            "sliders",
            "database",
            "clock-history",
        ],
        default_index=0,
        styles={
            "container": {
                "padding": "5px 0",
                "background-color": "transparent",
            },
            "icon": {
                "color": "#9AAAC0",
                "font-size": "17px",
            },
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "5px 0",
                "padding": "12px 14px",
                "border-radius": "10px",
                "color": "#C4CFDD",
                "--hover-color": "#1B2A3D",
            },
            "nav-link-selected": {
                "background-color": "#123B3C",
                "color": "#5EEAD4",
                "font-weight": "700",
            },
        },
    )

    st.markdown("---")
    st.markdown(
        """
        <div class="panel">
            <div class="eyebrow">SYSTEM STATUS</div>
            <p style="margin:10px 0 4px 0;color:#F1F5F9;">
                ● Application ready
            </p>
            <div class="muted">
                Local demonstration environment
            </div>
        </div>
        <p class="muted">
            Research prototype only. Not a substitute for clinical
            assessment by a qualified professional.
        </p>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# DASHBOARD
# =========================================================

if selection == "Dashboard":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">CLINICAL AI WORKSPACE</div>
            <h1>Medical imaging.<br>Made more accessible.</h1>
            <p>
                Explore your lung cancer classification models, upload
                an image for an experimental model prediction, and review
                your recent analysis history in one workspace.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total = len(st.session_state.prediction_history)
    cnn_count = sum(
        item["Model"] == "CNN" for item in st.session_state.prediction_history
    )
    ml_count = sum(
        item["Model"] == "ML" for item in st.session_state.prediction_history
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        render_metric("SESSION ANALYSES", str(total),
                      "Current browser session")

    with col2:
        render_metric("CNN ANALYSES", str(cnn_count),
                      "Uploaded image predictions")

    with col3:
        render_metric("ML ANALYSES", str(ml_count),
                      "Tabular model predictions")

    st.markdown("### Your workspace")
    st.caption("Choose a workflow to get started.")

    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.markdown("#### 🫁 Scan analysis")
            st.write(
                "Upload a JPG or PNG image and inspect the existing "
                "CNN model's output."
            )
            if st.button("Open scan analysis", use_container_width=True):
                st.session_state["navigation_target"] = "Scan Analysis"
                st.rerun()

    with c2:
        with st.container(border=True):
            st.markdown("#### 🧪 ML prediction")
            st.write(
                "Explore the existing tabular model using records "
                "from your test dataset."
            )
            if st.button("Open ML prediction", use_container_width=True):
                st.session_state["navigation_target"] = "ML Prediction"
                st.rerun()

    # A button can request navigation on the next rerun.
    if st.session_state.get("navigation_target"):
        target = st.session_state.pop("navigation_target")
        st.info(
            f"Use the sidebar to open **{target}**. "
            "Navigation is controlled by the sidebar menu."
        )

    st.markdown("### Recent activity")

    if st.session_state.prediction_history:
        st.dataframe(
            pd.DataFrame(st.session_state.prediction_history).head(5),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No analyses yet. Results will appear here after you "
            "run a prediction."
        )


# =========================================================
# CNN IMAGE SCAN ANALYSIS
# =========================================================

elif selection == "Scan Analysis":
    render_heading(
        "IMAGE CLASSIFICATION",
        "Scan analysis",
        "Upload a supported image to run the existing CNN model.",
    )

    st.markdown(
        """
        <div class="panel">
            <div class="eyebrow">CNN MODEL</div>
            <h3 style="margin-top:8px;">Image assessment workspace</h3>
            <p class="muted">
                Supported formats: PNG, JPG, JPEG. Images are resized
                to 224 × 224 pixels and normalized to the 0–1 range,
                matching the preprocessing in the original application.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"],
        help="Choose an image to preview and analyze.",
        key="cnn_upload",
    )

    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")

            left, right = st.columns([1, 1.15], gap="large")

            with left:
                st.markdown("#### Uploaded image")
                st.image(
                    pil_image,
                    use_column_width=True,
                    caption=uploaded_file.name,
                )
                st.caption(
                    f"Image dimensions: {pil_image.width} × "
                    f"{pil_image.height} pixels"
                )

            with right:
                st.markdown("#### Analysis")

                st.markdown(
                    f"""
                    <div class="panel">
                        <div class="metric-label">FILE NAME</div>
                        <p style="color:#F1F5F9;overflow-wrap:anywhere;">
                            {uploaded_file.name}
                        </p>
                        <div class="metric-label">FILE SIZE</div>
                        <p style="color:#F1F5F9;">
                            {uploaded_file.size / 1024:.1f} KB
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button(
                    "Run CNN prediction",
                    use_container_width=True,
                    type="primary",
                ):
                    try:
                        with st.spinner("Loading model and analyzing image..."):
                            cnn = load_cnn_model()

                            # Preserve the original preprocessing pipeline.
                            model_image = pil_image.resize((224, 224))
                            image_array = np.asarray(
                                model_image, dtype=np.float32
                            )
                            image_array = image_array / 255.0
                            image_array = np.expand_dims(
                                image_array, axis=0
                            )

                            predictions = cnn.predict(
                                image_array, verbose=0
                            )

                            # Preserve original binary output threshold.
                            score = float(np.asarray(predictions).reshape(-1)[0])

                            if score >= 0.5:
                                result = "Normal"
                                st.success("Model output: Normal")
                                st.metric(
                                    "Model output score",
                                    f"{score:.2%}",
                                )
                            else:
                                result = "Lung Cancer"
                                st.error("Model output: Lung Cancer")
                                st.metric(
                                    "Complement of model output",
                                    f"{1 - score:.2%}",
                                )

                            st.caption(
                                "These values reflect the existing model's "
                                "output and threshold. They are not "
                                "clinically validated probabilities."
                            )

                            add_history(
                                "CNN",
                                uploaded_file.name,
                                result,
                                score,
                            )

                            st.success(
                                "Analysis recorded in this session's history."
                            )

                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

        except Exception as exc:
            st.error(f"Could not open this image: {exc}")

    else:
        st.info("Upload an image above to begin.")


# =========================================================
# TABULAR ML PREDICTION
# =========================================================

elif selection == "ML Prediction":
    render_heading(
        "TABULAR CLASSIFICATION",
        "ML prediction",
        "Explore the original SVC-based model using your test dataset.",
    )

    if cancer_model is None:
        st.error(
            "The ML model could not be loaded. Check that "
            "models/final_model.sav exists."
        )
        st.code(model_load_error or "Unknown model loading error")
        st.stop()

    try:
        testx = pd.read_csv(DATA_DIR / "testx.csv", index_col=0)
        testy = pd.read_csv(DATA_DIR / "testy.csv", index_col=0)

        testx.reset_index(drop=True, inplace=True)
        testy.reset_index(drop=True, inplace=True)

        if testx.empty:
            st.warning("The test dataset contains no records.")
            st.stop()

        combined = pd.concat([testx, testy], axis=1)

    except Exception as exc:
        st.error(f"Could not load the test dataset: {exc}")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        render_metric("TEST RECORDS", str(len(testx)),
                      "Available input records")

    with col2:
        render_metric("MODEL", "SVC / ML",
                      "Loaded from final_model.sav")

    st.markdown("### Select a test record")

    selected_index = st.slider(
        "Dataset record",
        min_value=0,
        max_value=len(testx) - 1,
        value=min(25, len(testx) - 1),
    )

    selected_row = testx.iloc[selected_index]

    st.dataframe(
        selected_row.to_frame(name="Value"),
        use_container_width=True,
    )

    st.markdown("### Input features")

    feature_columns = list(testx.columns)

    with st.form("ml_prediction_form"):
        st.caption(
            "Values are prefilled from the selected test record. "
            "Review the numeric values before submitting."
        )

        input_values = {}

        columns = st.columns(3)

        for i, feature in enumerate(feature_columns):
            raw_value = selected_row[feature]

            try:
                default_value = float(raw_value)
                if not np.isfinite(default_value):
                    default_value = 0.0
            except (TypeError, ValueError):
                st.error(
                    f"Feature {feature!r} is not numeric. "
                    "This model's original input pipeline expects "
                    "numeric features."
                )
                st.stop()

            with columns[i % 3]:
                input_values[feature] = st.number_input(
                    str(feature),
                    value=default_value,
                    key=f"ml_feature_{feature}",
                    format="%.4f",
                )

        submitted = st.form_submit_button(
            "Run ML prediction",
            use_container_width=True,
        )

    if submitted:
        try:
            input_frame = pd.DataFrame(
                [[input_values[name] for name in feature_columns]],
                columns=feature_columns,
            )

            with st.spinner("Running ML model..."):
                prediction = cancer_model.predict(input_frame)[0]

            result = str(prediction)

            st.markdown("### Model result")

            if result.lower() == "high":
                st.error(f"Predicted risk category: {result}")
            elif result.lower() == "medium":
                st.warning(f"Predicted risk category: {result}")
            else:
                st.success(f"Predicted category: {result}")

            st.caption(
                "This is the model's category, not a medical diagnosis."
            )

            add_history(
                "ML",
                f"Test record {selected_index}",
                result,
            )

        except Exception as exc:
            st.error(
                "Prediction failed. The selected features may not match "
                f"the model's expected schema. Details: {exc}"
            )

    with st.expander("Preview test dataset"):
        st.dataframe(combined.head(20), use_container_width=True)


# =========================================================
# DATASET EXPLORER
# =========================================================

elif selection == "Dataset Explorer":
    render_heading(
        "DATA & METHODOLOGY",
        "Dataset explorer",
        "Inspect the datasets and existing model documentation.",
    )

    tabs = st.tabs([
        "Dataset",
        "Training data",
        "Test data",
        "CNN methodology",
    ])

    with tabs[0]:
        st.markdown("### Lung cancer dataset")

        csv_path = DATA_DIR / "data.csv"

        if csv_path.exists():
            data = pd.read_csv(csv_path)
            st.metric("Rows", f"{len(data):,}")
            st.metric("Columns", f"{len(data.columns):,}")
            st.dataframe(data.head(20), use_container_width=True)

            if st.button("Show dataset statistics"):
                st.dataframe(
                    data.describe(include="all").transpose(),
                    use_container_width=True,
                )
        else:
            st.warning(f"Dataset not found: {csv_path}")

        corr_path = IMAGE_DIR / "coors.png"

        if corr_path.exists():
            st.markdown("### Correlation matrix")
            st.image(
                Image.open(corr_path),
                use_column_width=True,
            )

    with tabs[1]:
        st.markdown("### Training data")

        for filename in ["train.csv", "trainy.csv"]:
            path = DATA_DIR / filename

            if path.exists():
                with st.expander(filename):
                    st.dataframe(
                        pd.read_csv(path).head(50),
                        use_container_width=True,
                    )
            else:
                st.info(f"{filename} was not found.")

    with tabs[2]:
        st.markdown("### Test data")

        for filename in ["testx.csv", "testy.csv"]:
            path = DATA_DIR / filename

            if path.exists():
                with st.expander(filename):
                    st.dataframe(
                        pd.read_csv(path).head(50),
                        use_container_width=True,
                    )
            else:
                st.info(f"{filename} was not found.")

    with tabs[3]:
        st.markdown("### CNN image classification")
        st.write(
            "The existing application uses a Keras CNN saved as "
            "models/keras_model.h5. Uploaded images are resized to "
            "224 × 224 pixels, converted to arrays, and divided by 255."
        )

        st.warning(
            "Confirm the class order and label mapping from the original "
            "training code before interpreting model predictions. "
            "The original application labels outputs using a 0.5 threshold."
        )

        summary_path = IMAGE_DIR / "summary.png"

        if summary_path.exists():
            st.image(
                Image.open(summary_path),
                caption="Existing CNN model summary",
                use_column_width=True,
            )


# =========================================================
# SESSION HISTORY
# =========================================================

elif selection == "History":
    render_heading(
        "RECENT ACTIVITY",
        "Prediction history",
        "Review results created during this active Streamlit session.",
    )

    history = st.session_state.prediction_history

    if not history:
        st.info(
            "Your history is empty. Run a scan or ML prediction to add "
            "the first entry."
        )
    else:
        h1, h2 = st.columns(2)

        with h1:
            render_metric("RECORDED RESULTS", str(len(history)),
                          "Current session only")

        with h2:
            render_metric(
                "MODEL TYPES",
                str(len(set(item["Model"] for item in history))),
                "CNN and ML",
            )

        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download history as CSV",
            data=csv_data,
            file_name="lungai_prediction_history.csv",
            mime="text/csv",
        )

        st.caption(
            "History is stored in Streamlit session state. It is not "
            "a permanent database and may be lost when the session ends."
        )

        if st.button("Clear session history"):
            st.session_state.prediction_history = []
            st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;padding:12px 0 20px 0;">
        <div style="color:#5EEAD4;font-weight:700;">
            LungAI · Clinical Intelligence
        </div>
        <div class="muted" style="margin-top:6px;">
            Academic research prototype · Not for clinical decision-making
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
