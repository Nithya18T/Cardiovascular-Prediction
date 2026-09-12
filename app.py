import streamlit as st
import pandas as pd
import joblib
import os
import uuid
from datetime import datetime
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CardioCare AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("heart_model.pkl")


@st.cache_data
def load_feature_importance():
    return pd.read_csv("feature_importance.csv")


model = load_model()
feature_df = load_feature_importance()

# =========================================================
# SESSION STATE
# =========================================================

if "patient_data" not in st.session_state:
    st.session_state.patient_data = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probability" not in st.session_state:
    st.session_state.probability = None

if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""

if "patient_id" not in st.session_state:
    st.session_state.patient_id = ""

if "assessment_date" not in st.session_state:
    st.session_state.assessment_date = ""

if "history" not in st.session_state:
    st.session_state.history = []


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #172554;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #64748b;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #172554;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.06);
        margin-bottom: 18px;
    }

    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.06);
    }

    .metric-title {
        color: #64748b;
        font-size: 15px;
        font-weight: 600;
    }

    .metric-value {
        color: #172554;
        font-size: 32px;
        font-weight: 800;
        margin-top: 8px;
    }

    .success-box {
        background-color: #f0fdf4;
        border-left: 5px solid #16a34a;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 15px;
    }

    .warning-box {
        background-color: #fff7ed;
        border-left: 5px solid #f97316;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 15px;
    }

    .danger-box {
        background-color: #fef2f2;
        border-left: 5px solid #dc2626;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 15px;
    }

    .info-box {
        background-color: #eff6ff;
        border-left: 5px solid #2563eb;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 15px;
    }

    .recommendation {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin: 8px 0px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
    }

    .patient-header {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0px 3px 12px rgba(0,0,0,0.05);
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 45px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_risk_level(risk):
    if risk < 30:
        return "Low Risk 🟢"
    elif risk < 70:
        return "Medium Risk 🟡"
    else:
        return "High Risk 🔴"


def get_risk_factors(data):
    factors = []

    age = data["age"].iloc[0]
    chol = data["chol"].iloc[0]
    bp = data["trestbps"].iloc[0]
    exang = data["exang"].iloc[0]
    oldpeak = data["oldpeak"].iloc[0]
    ca = data["ca"].iloc[0]
    fbs = data["fbs"].iloc[0]

    if chol > 240:
        factors.append("High Cholesterol")

    if bp > 140:
        factors.append("High Blood Pressure")

    if age > 60:
        factors.append("Advanced Age")

    if exang == 1:
        factors.append("Exercise-Induced Angina")

    if oldpeak > 2:
        factors.append("Elevated Old Peak")

    if ca >= 2:
        factors.append("Multiple Affected Vessels")

    if fbs == 1:
        factors.append("Elevated Fasting Blood Sugar")

    if len(factors) == 0:
        factors.append("No major rule-based risk factor detected")

    return factors


def get_recommendations(data, prediction):

    recommendations = []

    age = data["age"].iloc[0]
    chol = data["chol"].iloc[0]
    bp = data["trestbps"].iloc[0]
    exang = data["exang"].iloc[0]
    oldpeak = data["oldpeak"].iloc[0]
    fbs = data["fbs"].iloc[0]
    ca = data["ca"].iloc[0]

    if prediction == 0:
        recommendations.append(
            "Consult a qualified cardiologist for professional evaluation."
        )

    if bp > 140:
        recommendations.append(
            "Monitor blood pressure regularly and discuss elevated readings with a healthcare professional."
        )

    if chol > 240:
        recommendations.append(
            "Reduce foods high in saturated and trans fats and discuss cholesterol management with a clinician."
        )

    if fbs == 1:
        recommendations.append(
            "Monitor blood glucose and discuss fasting blood sugar results with a healthcare professional."
        )

    if exang == 1:
        recommendations.append(
            "Exercise-induced chest discomfort should be evaluated by a healthcare professional."
        )

    if oldpeak > 2:
        recommendations.append(
            "Discuss the elevated stress-test indicator with a qualified clinician."
        )

    if ca >= 2:
        recommendations.append(
            "The vessel-related input warrants professional cardiovascular evaluation."
        )

    if age > 60:
        recommendations.append(
            "Maintain regular cardiovascular screening appropriate for your age."
        )

    recommendations.extend([
        "Maintain a balanced, heart-healthy diet.",
        "Stay physically active according to your healthcare professional's advice.",
        "Maintain a healthy body weight.",
        "Avoid smoking and limit alcohol consumption.",
        "Maintain regular health checkups."
    ])

    return list(dict.fromkeys(recommendations))


def create_risk_gauge(risk):

    fig, ax = plt.subplots(figsize=(7, 2.5))

    ax.barh(
        [0],
        [100],
        height=0.35,
        alpha=0.15
    )

    ax.barh(
        [0],
        [risk],
        height=0.35
    )

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Estimated Risk (%)")
    ax.set_title(
        f"Cardiovascular Risk: {risk:.2f}%",
        fontsize=16,
        fontweight="bold"
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    return fig


def create_xai_chart(feature_data):

    df = feature_data.copy()

    # Detect likely feature and importance columns
    feature_column = None
    importance_column = None

    for col in df.columns:
        lower = col.lower()

        if "feature" in lower or "name" in lower:
            feature_column = col

        if "importance" in lower or "score" in lower:
            importance_column = col

    # Fallback
    if feature_column is None:
        feature_column = df.columns[0]

    if importance_column is None:
        importance_column = df.columns[1]

    df = df.sort_values(
        importance_column,
        ascending=True
    ).head(10)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        df[feature_column].astype(str),
        df[importance_column]
    )

    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Feature")
    ax.set_title(
        "Top Prediction Factors",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    return fig


def generate_pdf():

    data = st.session_state.patient_data
    prediction = st.session_state.prediction
    probability = st.session_state.probability

    risk = probability[0] * 100
    wellness = 100 - risk

    risk_level = get_risk_level(risk)

    patient_name = st.session_state.patient_name
    patient_id = st.session_state.patient_id
    date = st.session_state.assessment_date

    risk_factors = get_risk_factors(data)
    recommendations = get_recommendations(data, prediction)

    filename = "cardiovascular_assessment_report.pdf"

    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=15,
        spaceBefore=15,
        spaceAfter=10
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontSize=10,
        leading=15
    )

    story = []

    story.append(
        Paragraph(
            "🫀 CardioCare AI",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Cardiovascular Risk Assessment Report",
            title_style
        )
    )

    story.append(Spacer(1, 10))

    patient_table = Table([
        ["Patient ID", patient_id],
        ["Patient Name", patient_name],
        ["Assessment Date", date]
    ], colWidths=[150, 300])

    patient_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("PADDING", (0, 0), (-1, -1), 8)
        ])
    )

    story.append(patient_table)

    story.append(
        Paragraph(
            "Risk Assessment",
            heading_style
        )
    )

    result_text = (
        "Heart Disease Detected by the Model"
        if prediction == 0
        else "No Heart Disease Detected by the Model"
    )

    assessment_table = Table([
        ["Prediction", result_text],
        ["Risk Score", f"{risk:.2f}%"],
        ["Risk Level", risk_level],
        ["Wellness Score", f"{wellness:.2f}/100"]
    ], colWidths=[150, 300])

    assessment_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8)
        ])
    )

    story.append(assessment_table)

    story.append(
        Paragraph(
            "Patient Clinical Parameters",
            heading_style
        )
    )

    clinical = [
        ["Parameter", "Value"],
        ["Age", str(data["age"].iloc[0])],
        ["Sex", "Male" if data["sex"].iloc[0] == 1 else "Female"],
        ["Chest Pain Type", str(data["cp"].iloc[0])],
        ["Resting Blood Pressure", str(data["trestbps"].iloc[0])],
        ["Cholesterol", str(data["chol"].iloc[0])],
        ["Fasting Blood Sugar", str(data["fbs"].iloc[0])],
        ["Rest ECG", str(data["restecg"].iloc[0])],
        ["Maximum Heart Rate", str(data["thalach"].iloc[0])],
        ["Exercise Angina", str(data["exang"].iloc[0])],
        ["Old Peak", str(data["oldpeak"].iloc[0])],
        ["Slope", str(data["slope"].iloc[0])],
        ["Major Vessels", str(data["ca"].iloc[0])],
        ["Thal", str(data["thal"].iloc[0])]
    ]

    clinical_table = Table(
        clinical,
        colWidths=[220, 230]
    )

    clinical_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 7)
        ])
    )

    story.append(clinical_table)

    story.append(
        Paragraph(
            "Identified Risk Factors",
            heading_style
        )
    )

    for factor in risk_factors:
        story.append(
            Paragraph(
                "• " + factor,
                normal_style
            )
        )

    story.append(
        Paragraph(
            "Personalized Recommendations",
            heading_style
        )
    )

    for recommendation in recommendations:
        story.append(
            Paragraph(
                "• " + recommendation,
                normal_style
            )
        )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Disclaimer: This report is generated using an AI-based "
            "prediction system for educational and preliminary screening "
            "purposes. It is not a substitute for professional medical "
            "diagnosis or treatment.",
            normal_style
        )
    )

    doc.build(story)

    return filename


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    """
    <h1 style="text-align:center;">🫀</h1>
    <h2 style="text-align:center;">CardioCare AI</h2>
    """,
    unsafe_allow_html=True
)

st.sidebar.caption(
    "Explainable Cardiovascular Risk Prediction"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "👤 Patient Details",
        "📊 Risk Assessment",
        "🔍 Explainable AI",
        "📄 Medical Report"
    ]
)

st.sidebar.markdown("---")

if st.session_state.patient_id:

    st.sidebar.markdown(
        f"""
        **Current Patient**

        🆔 {st.session_state.patient_id}

        👤 {st.session_state.patient_name}
        """
    )

else:

    st.sidebar.info(
        "Enter patient details to begin an assessment."
    )

st.sidebar.markdown("---")

st.sidebar.caption(
    "CardioCare AI • AI/ML Healthcare Project"
)


# =========================================================
# PAGE 1
# PATIENT DETAILS
# =========================================================

if page == "👤 Patient Details":

    st.markdown(
        '<div class="main-title">🫀 Patient Details</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter patient information and cardiovascular parameters.'
        '</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # BASIC DETAILS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">👤 Basic Information</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        patient_name = st.text_input(
            "Patient Name",
            value=st.session_state.patient_name,
            placeholder="Enter patient name"
        )

    with col2:

        patient_id = st.text_input(
            "Patient ID",
            value=st.session_state.patient_id
            if st.session_state.patient_id
            else "PT-" + str(uuid.uuid4())[:8].upper()
        )

    with col3:

        assessment_date = st.date_input(
            "Assessment Date"
        )

    # -----------------------------------------------------
    # CLINICAL DATA
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🩺 Clinical Parameters</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50
        )

    with col2:

        sex = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda x:
                "Female" if x == 0 else "Male"
        )

    with col3:

        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3]
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80,
            max_value=250,
            value=120
        )

    with col2:

        chol = st.number_input(
            "Cholesterol (mg/dl)",
            min_value=100,
            max_value=600,
            value=200
        )

    with col3:

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        restecg = st.selectbox(
            "Rest ECG",
            [0, 1, 2]
        )

    with col2:

        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=60,
            max_value=220,
            value=150
        )

    with col3:

        exang = st.selectbox(
            "Exercise-Induced Angina",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )

    # -----------------------------------------------------
    # ADDITIONAL PARAMETERS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📋 Additional Parameters</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        oldpeak = st.number_input(
            "Old Peak",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

    with col2:

        slope = st.selectbox(
            "Slope",
            [0, 1, 2]
        )

    with col3:

        ca = st.selectbox(
            "Major Vessels",
            [0, 1, 2, 3, 4]
        )

    with col4:

        thal = st.selectbox(
            "Thal",
            [0, 1, 2, 3]
        )

    # -----------------------------------------------------
    # QUICK INPUT SUMMARY
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📌 Quick Summary</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Age", age)

    with c2:
        st.metric("Blood Pressure", trestbps)

    with c3:
        st.metric("Cholesterol", chol)

    with c4:
        st.metric("Max Heart Rate", thalach)

    st.markdown("---")

    # -----------------------------------------------------
    # ANALYZE
    # -----------------------------------------------------

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        if st.button(
            "🔍 Analyze Cardiovascular Risk",
            use_container_width=True
        ):

            if not patient_name.strip():

                st.warning(
                    "Please enter the patient name."
                )

            else:

                input_data = pd.DataFrame(
                    [[
                        age,
                        sex,
                        cp,
                        trestbps,
                        chol,
                        fbs,
                        restecg,
                        thalach,
                        exang,
                        oldpeak,
                        slope,
                        ca,
                        thal
                    ]],
                    columns=[
                        "age",
                        "sex",
                        "cp",
                        "trestbps",
                        "chol",
                        "fbs",
                        "restecg",
                        "thalach",
                        "exang",
                        "oldpeak",
                        "slope",
                        "ca",
                        "thal"
                    ]
                )

                prediction = model.predict(
                    input_data
                )[0]

                probability = model.predict_proba(
                    input_data
                )[0]

                st.session_state.patient_data = input_data

                st.session_state.prediction = prediction

                st.session_state.probability = probability

                st.session_state.patient_name = patient_name

                st.session_state.patient_id = patient_id

                st.session_state.assessment_date = str(
                    assessment_date
                )

                # Add history
                risk = probability[0] * 100

                history_entry = {
                    "Patient ID": patient_id,
                    "Patient Name": patient_name,
                    "Date": str(assessment_date),
                    "Risk Score": round(risk, 2),
                    "Risk Level": get_risk_level(risk),
                    "Prediction":
                        "Disease Detected"
                        if prediction == 0
                        else "No Disease"
                }

                st.session_state.history.append(
                    history_entry
                )

                st.success(
                    "Analysis completed successfully!"
                )

                st.info(
                    "Use the sidebar to open "
                    "**Risk Assessment**."
                )


# =========================================================
# PAGE 2
# RISK ASSESSMENT
# =========================================================

elif page == "📊 Risk Assessment":

    st.markdown(
        '<div class="main-title">📊 Risk Assessment</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Overview of the AI-generated cardiovascular assessment.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.patient_data is None:

        st.warning(
            "No assessment found. Please complete Patient Details first."
        )

        st.stop()

    data = st.session_state.patient_data
    prediction = st.session_state.prediction
    probability = st.session_state.probability

    risk = probability[0] * 100
    wellness = 100 - risk
    risk_level = get_risk_level(risk)

    # -----------------------------------------------------
    # PATIENT HEADER
    # -----------------------------------------------------

    st.markdown(
        f"""
        <div class="patient-header">
            <b>Patient:</b> {st.session_state.patient_name}
            &nbsp;&nbsp;&nbsp;
            <b>Patient ID:</b> {st.session_state.patient_id}
            &nbsp;&nbsp;&nbsp;
            <b>Date:</b> {st.session_state.assessment_date}
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Risk Score</div>
                <div class="metric-value">{risk:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Risk Level</div>
                <div class="metric-value">{risk_level}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Wellness Score</div>
                <div class="metric-value">{wellness:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        confidence = max(probability) * 100

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Model Confidence</div>
                <div class="metric-value">{confidence:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # RISK GAUGE
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🎯 Cardiovascular Risk Gauge</div>',
        unsafe_allow_html=True
    )

    gauge = create_risk_gauge(risk)

    st.pyplot(
        gauge,
        use_container_width=True
    )

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    if prediction == 0:

        st.markdown(
            """
            <div class="danger-box">
                <h3>⚠️ Heart Disease Detected by the Model</h3>
                The model indicates an elevated cardiovascular risk profile.
                Professional medical evaluation is recommended.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="success-box">
                <h3>✅ No Heart Disease Detected by the Model</h3>
                The analyzed indicators do not indicate heart disease
                according to this model's prediction.
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # PATIENT METRICS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🩺 Clinical Summary</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Age",
            data["age"].iloc[0]
        )

    with c2:
        st.metric(
            "Blood Pressure",
            data["trestbps"].iloc[0]
        )

    with c3:
        st.metric(
            "Cholesterol",
            data["chol"].iloc[0]
        )

    with c4:
        st.metric(
            "Maximum Heart Rate",
            data["thalach"].iloc[0]
        )

    # -----------------------------------------------------
    # HISTORY
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📈 Assessment History</div>',
        unsafe_allow_html=True
    )

    if len(st.session_state.history) > 0:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if len(history_df) >= 2:

            chart_data = history_df[
                ["Date", "Risk Score"]
            ].copy()

            chart_data = chart_data.set_index(
                "Date"
            )

            st.line_chart(
                chart_data
            )

    else:

        st.info(
            "No previous assessments available."
        )


# =========================================================
# PAGE 3
# EXPLAINABLE AI
# =========================================================

elif page == "🔍 Explainable AI":

    st.markdown(
        '<div class="main-title">🔍 Explainable AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Understand which features are most important to the model.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.patient_data is None:

        st.warning(
            "Please complete a patient assessment first."
        )

        st.stop()

    data = st.session_state.patient_data
    prediction = st.session_state.prediction

    # -----------------------------------------------------
    # MODEL EXPLANATION
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🧠 Model Interpretation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        <b>How does Explainable AI help?</b><br><br>
        The system displays the most influential features used by
        the trained machine-learning model. This makes the prediction
        easier to understand instead of presenting only a final result.
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Feature Importance</div>',
        unsafe_allow_html=True
    )

    try:

        xai_chart = create_xai_chart(
            feature_df
        )

        st.pyplot(
            xai_chart,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            "Unable to generate feature importance chart."
        )

        st.write(e)

    # -----------------------------------------------------
    # TABLE
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📋 Top Important Features</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        feature_df.head(10),
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # PATIENT-SPECIFIC FACTORS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">⚠️ Patient-Specific Risk Factors</div>',
        unsafe_allow_html=True
    )

    factors = get_risk_factors(data)

    for factor in factors:

        st.markdown(
            f"""
            <div class="recommendation">
                🔴 <b>{factor}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # PERSONALIZED RECOMMENDATIONS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🏥 Personalized Recommendations</div>',
        unsafe_allow_html=True
    )

    recommendations = get_recommendations(
        data,
        prediction
    )

    for recommendation in recommendations:

        st.markdown(
            f"""
            <div class="recommendation">
                ✅ {recommendation}
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# PAGE 4
# MEDICAL REPORT
# =========================================================

elif page == "📄 Medical Report":

    st.markdown(
        '<div class="main-title">📄 Medical Report</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Generate and download a complete cardiovascular assessment report.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.patient_data is None:

        st.warning(
            "Please complete a patient assessment first."
        )

        st.stop()

    data = st.session_state.patient_data
    prediction = st.session_state.prediction
    probability = st.session_state.probability

    risk = probability[0] * 100
    wellness = 100 - risk

    # -----------------------------------------------------
    # REPORT PREVIEW
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📋 Report Preview</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            f"""
            <div class="card">
                <h3>👤 Patient Information</h3>
                <b>Patient Name:</b> {st.session_state.patient_name}<br>
                <b>Patient ID:</b> {st.session_state.patient_id}<br>
                <b>Assessment Date:</b> {st.session_state.assessment_date}
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        result = (
            "Heart Disease Detected"
            if prediction == 0
            else "No Heart Disease Detected"
        )

        st.markdown(
            f"""
            <div class="card">
                <h3>📊 Assessment Result</h3>
                <b>Prediction:</b> {result}<br>
                <b>Risk Score:</b> {risk:.2f}%<br>
                <b>Risk Level:</b> {get_risk_level(risk)}<br>
                <b>Wellness Score:</b> {wellness:.2f}/100
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # CLINICAL DATA
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🩺 Clinical Data</div>',
        unsafe_allow_html=True
    )

    display_data = data.copy()

    display_data = display_data.T

    display_data.columns = ["Value"]

    st.dataframe(
        display_data,
        use_container_width=True
    )

    # -----------------------------------------------------
    # RISK FACTORS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">⚠️ Risk Factors</div>',
        unsafe_allow_html=True
    )

    for factor in get_risk_factors(data):

        st.write(
            "🔴",
            factor
        )

    # -----------------------------------------------------
    # RECOMMENDATIONS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🏥 Recommendations</div>',
        unsafe_allow_html=True
    )

    for recommendation in get_recommendations(
        data,
        prediction
    ):

        st.write(
            "✅",
            recommendation
        )

    # -----------------------------------------------------
    # PDF
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">⬇️ Download Report</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "📄 Generate PDF Report",
        use_container_width=True
    ):

        try:

            pdf_file = generate_pdf()

            with open(
                pdf_file,
                "rb"
            ) as file:

                pdf_bytes = file.read()

            st.download_button(
                label="⬇️ Download Cardiovascular Report",
                data=pdf_bytes,
                file_name=(
                    f"{st.session_state.patient_id}"
                    "_cardiovascular_report.pdf"
                ),
                mime="application/pdf",
                use_container_width=True
            )

            st.success(
                "PDF report generated successfully."
            )

        except Exception as e:

            st.error(
                "Unable to generate PDF report."
            )

            st.write(e)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#64748b;">
        🫀 <b>CardioCare AI</b><br>
        Explainable Ensemble Learning Framework for
        Cardiovascular Risk Prediction and Preventive Healthcare
        <br><br>
        <small>
        This application is intended for educational and preliminary
        screening purposes and does not replace professional medical diagnosis.
        </small>
    </div>
    """,
    unsafe_allow_html=True
)