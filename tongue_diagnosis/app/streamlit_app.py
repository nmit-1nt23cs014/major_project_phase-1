import os
import sys
import json
import tempfile
from textwrap import dedent

import numpy as np
import streamlit as st
from PIL import Image

# -- path setup ---------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# -- lazy imports (only after path is set) ------------------------------------
from modules.preprocessor import preprocess_image
from modules.segmentor import segment_tongue
from modules.model import load_saved_model, predict_single


def _html(markup: str):
    cleaned = "\n".join(line.lstrip() for line in dedent(markup).strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


# =============================================================================
#  PAGE CONFIG  -- must be the very first Streamlit call
# =============================================================================
st.set_page_config(
    page_title="Tongue Diagnosis",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# =============================================================================
#  GLOBAL STYLES
# =============================================================================
_html(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet">

    <style>

    /* -- page background ------------------------------------------------ */
    .stApp                          { background: #0A0E17; }
    .block-container                { padding-top: 0 !important;
                                      padding-bottom: 4rem !important;
                                      max-width: 860px !important; }

    /* -- hide sidebar & toolbar ----------------------------------------- */
    [data-testid="stSidebar"]       { display: none !important; }
    [data-testid="stHeader"]        { background: transparent !important; }
    #MainMenu, footer               { display: none !important; }

    /* -- global text colour --------------------------------------------- */
    html, body, [class*="css"]      { color: #F0F4F8 !important;
                                      font-family: 'DM Sans', sans-serif; }

    /* -- file uploader --------------------------------------------------- */
    [data-testid="stFileUploader"]  { background: #111827 !important;
                                      border: 1.5px dashed #1F2D40 !important;
                                      border-radius: 12px !important;
                                      padding: 0.5rem 1rem !important; }
    [data-testid="stFileUploader"] *{ color: #8899AA !important; }
    [data-testid="stFileUploader"] small { color: #4A5568 !important; }

    /* -- metrics --------------------------------------------------------- */
    [data-testid="stMetric"]        { background: #111827;
                                      border: 1px solid #1F2D40;
                                      border-radius: 10px;
                                      padding: 1rem !important; }
    [data-testid="stMetricLabel"]   { color: #8899AA !important;
                                      font-size: 0.7rem !important;
                                      text-transform: uppercase;
                                      letter-spacing: .1em; }
    [data-testid="stMetricValue"]   { color: #00C9A7 !important;
                                      font-family: 'Syne', sans-serif !important;
                                      font-size: 1.8rem !important;
                                      font-weight: 800 !important; }

    /* -- progress bar ---------------------------------------------------- */
    [data-testid="stProgress"] > div { background: #1F2D40 !important;
                                        border-radius: 99px !important; }
    [data-testid="stProgress"] > div > div { background: #00C9A7 !important;
                                              border-radius: 99px !important; }

    /* -- image captions -------------------------------------------------- */
    .stImage p, .stImage figcaption { color: #4A5568 !important;
                                      font-size: 0.68rem !important;
                                      text-align: center;
                                      text-transform: uppercase;
                                      letter-spacing: .1em;
                                      margin-top: 0.4rem; }

    /* -- spinner --------------------------------------------------------- */
    .stSpinner > div { border-top-color: #00C9A7 !important; }

    /* -- divider --------------------------------------------------------- */
    hr { border-color: #1F2D40 !important; margin: 1.5rem 0 !important; }

    </style>
    """
)


# =============================================================================
#  CACHED RESOURCES
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    path = os.path.join(ROOT, "training", "saved_model", "best_model.h5")
    if not os.path.exists(path):
        return None
    return load_saved_model(path)


@st.cache_data(show_spinner=False)
def load_label_map():
    path = os.path.join(ROOT, "training", "saved_model", "label_map.json")
    if not os.path.exists(path):
        return {"Normal": 0, "GI_Disease": 1}
    with open(path) as f:
        return json.load(f)


# =============================================================================
#  REUSABLE HTML HELPERS
# =============================================================================
def _panel_label(text: str):
    """Small uppercase label shown above each pipeline step image."""
    _html(
        f"""
        <p style="
            font-family:'DM Sans',sans-serif;
            font-size:.68rem;
            color:#8899AA;
            text-transform:uppercase;
            letter-spacing:.12em;
            margin:0 0 .4rem 0;
        ">{text}</p>
        """
    )


def _section_label(text: str):
    """Thin uppercase section header."""
    _html(
        f"""
        <p style="
            font-family:'DM Sans',sans-serif;
            font-size:.72rem;
            color:#8899AA;
            text-transform:uppercase;
            letter-spacing:.14em;
            margin:1.5rem 0 .6rem 0;
        ">{text}</p>
        """
    )


# =============================================================================
#  PAGE HEADER
# =============================================================================
_html(
    """
    <div style="text-align:center; padding:3.5rem 0 2.5rem 0;">

        <span style="
            display:inline-block;
            background:#041F1B;
            color:#00C9A7;
            font-family:'DM Sans',sans-serif;
            font-size:.7rem;
            font-weight:500;
            letter-spacing:.16em;
            text-transform:uppercase;
            padding:.35rem 1.1rem;
            border-radius:99px;
            border:1px solid #00856F;
            margin-bottom:1.4rem;
        ">AI-Powered Medical Screening Tool</span>

        <h1 style="
            font-family:'Syne',sans-serif;
            font-weight:800;
            font-size:2.6rem;
            color:#F0F4F8;
            letter-spacing:-.025em;
            line-height:1.15;
            margin:0 0 .85rem 0;
        ">
            Tongue Diagnosis<br>
            <span style="color:#00C9A7;">Intelligence Model</span>
        </h1>

        <p style="
            font-family:'DM Sans',sans-serif;
            font-size:.95rem;
            color:#8899AA;
            font-weight:300;
            max-width:460px;
            margin:0 auto;
            line-height:1.65;
        ">
            Upload a tongue image to screen for gastrointestinal disease
            indicators using deep&nbsp;learning image analysis.
        </p>

    </div>
    """
)

st.markdown("<hr>", unsafe_allow_html=True)


# =============================================================================
#  LOAD MODEL
# =============================================================================
model = load_model()
label_map = load_label_map()

if model is None:
    st.markdown(
        dedent(
            """
        <div style="
            text-align:center;
            padding:4rem 2rem;
            max-width:460px;
            margin:0 auto;
        ">
            <div style="font-size:2.2rem; margin-bottom:1rem; color:#FF4D6D;">⚠</div>
            <h2 style="
                font-family:'Syne',sans-serif;
                font-weight:700;
                color:#F0F4F8;
                margin-bottom:.6rem;
            ">Model Not Found</h2>
            <p style="
                font-family:'DM Sans',sans-serif;
                font-size:.9rem;
                color:#8899AA;
                line-height:1.6;
                margin-bottom:1.4rem;
            ">Train the model first, then relaunch the app.</p>
            <code style="
                background:#111827;
                border:1px solid #1F2D40;
                border-radius:6px;
                padding:.6rem 1.2rem;
                font-size:.85rem;
                color:#00C9A7;
            ">python training/train.py</code>
        </div>
        """
        ).strip(),
        unsafe_allow_html=True,
    )
    st.stop()


# =============================================================================
#  UPLOAD SECTION
# =============================================================================
_section_label("Step 1 -- Upload Image")

uploaded_file = st.file_uploader(
    label="",
    type=["jpg", "jpeg", "png"],
    help="Frontal tongue photo. Ensure even lighting. JPG or PNG.",
    label_visibility="collapsed",
)


# =============================================================================
#  PIPELINE + RESULT  (shown only after upload)
# =============================================================================
if uploaded_file is not None:

    # -- save to temp file -------------------------------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # -- PIPELINE PANELS ------------------------------------------------
        _html("""<div style='height:1.5rem'></div>""")
        _section_label("Step 2 -- Pipeline Preview")

        col1, col2, col3 = st.columns(3, gap="medium")

        # Panel 1 -- Original
        with col1:
            _panel_label("01 -- Original")
            original_pil = Image.open(uploaded_file)
            st.image(original_pil, use_column_width=True, caption="uploaded image")

        # Panel 2 -- Preprocessed
        with col2:
            _panel_label("02 -- Preprocessed")
            with st.spinner("Processing..."):
                preprocessed = preprocess_image(tmp_path)
            st.image(preprocessed, use_column_width=True, caption="resized · normalized · denoised")

        # Panel 3 -- Segmented
        with col3:
            _panel_label("03 -- Segmented")
            with st.spinner("Isolating tongue region..."):
                segmented, mask = segment_tongue(preprocessed, return_mask=True)
            st.image(segmented, use_column_width=True, caption="tongue region isolated")

        # -- DIVIDER --------------------------------------------------------
        _html("""<div style='height:1rem'></div>""")
        st.markdown("<hr>", unsafe_allow_html=True)

        # -- RUN PREDICTION -------------------------------------------------
        with st.spinner("Running analysis..."):
            result = predict_single(model, segmented)

        label = result["label"]
        confidence = result["confidence"]
        raw_score = result["raw_score"]

        # -- RESULT CARD ----------------------------------------------------
        _section_label("Step 3 -- Diagnosis Result")

        is_normal = label == "Normal"
        accent_colour = "#00C9A7" if is_normal else "#FF4D6D"
        card_bg = "#041F1B" if is_normal else "#1A0A0F"
        result_heading = "Normal" if is_normal else "Potential GI Disease Detected"
        result_body = (
            "No significant gastrointestinal indicators detected in tongue morphology."
            if is_normal else
            "Tongue morphology shows patterns consistent with gastrointestinal conditions. "
            "Please consult a qualified physician for further evaluation."
        )

        _html(
            f"""
            <div style="
                background:{card_bg};
                border:1px solid {accent_colour};
                border-left:4px solid {accent_colour};
                border-radius:12px;
                padding:1.6rem 2rem;
                margin-bottom:1.2rem;
            ">
                <div style="
                    font-family:'DM Sans',sans-serif;
                    font-size:.68rem;
                    color:{accent_colour};
                    text-transform:uppercase;
                    letter-spacing:.16em;
                    margin-bottom:.5rem;
                ">Classification</div>

                <div style="
                    font-family:'Syne',sans-serif;
                    font-size:2rem;
                    font-weight:800;
                    color:{accent_colour};
                    letter-spacing:-.02em;
                    margin-bottom:.5rem;
                ">{result_heading}</div>

                <div style="
                    font-family:'DM Sans',sans-serif;
                    font-size:.9rem;
                    color:#8899AA;
                    line-height:1.6;
                ">{result_body}</div>
            </div>
            """
        )

        # -- CONFIDENCE ROW -------------------------------------------------
        m_col, p_col = st.columns([1, 2], gap="large")

        with m_col:
            st.metric(label="Confidence", value=f"{confidence}%")

        with p_col:
            _html(
                """
                <p style="
                    font-family:'DM Sans',sans-serif;
                    font-size:.68rem;
                    color:#8899AA;
                    text-transform:uppercase;
                    letter-spacing:.1em;
                    margin:1rem 0 .5rem 0;
                ">Prediction Confidence</p>
                """
            )
            st.progress(int(confidence))

        # -- DISCLAIMER -----------------------------------------------------
        _html(
            """
            <div style="
                border-top:1px solid #1F2D40;
                margin-top:2.5rem;
                padding-top:1rem;
            ">
                <p style="
                    font-family:'DM Sans',sans-serif;
                    font-size:.75rem;
                    color:#4A5568;
                    line-height:1.65;
                    max-width:680px;
                    margin:0;
                ">
                    <span style="color:#8899AA; font-weight:500;">Medical Disclaimer:</span>
                    This tool is an academic research prototype developed for educational purposes
                    only. It is not a certified medical device and must not be used for clinical
                    diagnosis. Always consult a qualified healthcare professional for medical advice.
                </p>
            </div>
            """
        )

    except Exception as exc:
        _html(
            f"""
            <div style="
                background:#1A0A0F;
                border:1px solid #FF4D6D;
                border-radius:10px;
                padding:1.2rem 1.5rem;
                margin-top:1rem;
            ">
                <p style="
                    font-family:'DM Sans',sans-serif;
                    font-size:.85rem;
                    color:#FF4D6D;
                    margin:0;
                ">
                    Failed to process image -- {exc}<br>
                    <span style="color:#8899AA;">
                        Ensure the file is a valid JPG or PNG and try again.
                    </span>
                </p>
            </div>
            """
        )

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# =============================================================================
#  FOOTER
# =============================================================================
_html(
    """
    <div style="
        text-align:center;
        padding:3rem 0 1.5rem 0;
        border-top:1px solid #1F2D40;
        margin-top:3rem;
    ">
        <p style="
            font-family:'DM Sans',sans-serif;
            font-size:.7rem;
            color:#4A5568;
            letter-spacing:.05em;
            margin:0;
        ">
            Intelligent Tongue Diagnosis Model &nbsp;·&nbsp;
            VTU Major Project Phase 1 &nbsp;·&nbsp;
            Academic Year 2025-26
        </p>
    </div>
    """
)
