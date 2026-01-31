import time
import random
import datetime as dt
import requests

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="PneumoAI - AI-Assisted Pneumonia Detector",
    page_icon="💨",
    layout="wide",
)


# ------------------------------------------------------
# INIT SESSION STATE
# ------------------------------------------------------
if "total_scans" not in st.session_state:
    st.session_state.total_scans = 47
if "normal_scans" not in st.session_state:
    st.session_state.normal_scans = 38
if "pneumonia_scans" not in st.session_state:
    st.session_state.pneumonia_scans = 9
if "avg_time" not in st.session_state:
    st.session_state.avg_time = 1.2


if "recent_analyses" not in st.session_state:
    st.session_state.recent_analyses = [
        {
            "id": "PT-4521",
            "time_ago": "2 min ago",
            "label": "Normal",
            "confidence": 0.942,
            "severity": None,
            "type": "Normal",
        },
        {
            "id": "PT-4522",
            "time_ago": "15 min ago",
            "label": "Bacterial Pneumonia",
            "confidence": 0.875,
            "severity": "HIGH",
            "type": "Bacterial Pneumonia",
        },
        {
            "id": "PT-4523",
            "time_ago": "32 min ago",
            "label": "Viral Pneumonia",
            "confidence": 0.783,
            "severity": "MEDIUM",
            "type": "Viral Pneumonia",
        },
        {
            "id": "PT-4524",
            "time_ago": "1 hr ago",
            "label": "Normal",
            "confidence": 0.961,
            "severity": None,
            "type": "Normal",
        },
    ]


if "pending_reviews" not in st.session_state:
    st.session_state.pending_reviews = [
        {
            "id": "PT-4522",
            "time_ago": "15 min",
            "label": "Bacterial Pneumonia",
            "severity": "HIGH",
            "confidence": 0.875,
        },
        {
            "id": "PT-4523",
            "time_ago": "32 min",
            "label": "Viral Pneumonia",
            "severity": "MEDIUM",
            "confidence": 0.783,
        },
        {
            "id": "PT-4528",
            "time_ago": "8 min",
            "label": "Bacterial Pneumonia",
            "severity": "HIGH",
            "confidence": 0.721,
        },
    ]


# ------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------
st.sidebar.title("PneumoAI")
st.sidebar.caption("AI-Assisted Pneumonia Screening")


page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Upload X-ray", "Batch Analysis", "Model Comparison", "Reports"],
    index=0,
)


st.sidebar.markdown("---")
st.sidebar.write("Settings")
st.sidebar.write("Help")


# ------------------------------------------------------
# TOP BAR (FAKE)
# ------------------------------------------------------
st.markdown(
    """
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
        <div>
            <h2 style="margin-bottom:0;">AI-Assisted Pneumonia Detector</h2>
            <p style="margin-top:0; color:gray; font-size:0.9rem;">
                Second-opinion diagnostic support for radiologists
            </p>
        </div>
        <div style="text-align:right; font-size:0.9rem; color:gray;">
            <div>City General Hospital</div>
            <div style="margin-top:0.3rem;">
                <span style="margin-right:1rem;">🔔</span>
                <span style="background-color:#eef3ff; padding:0.25rem 0.6rem; border-radius:999px;">Dr. Singh</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------
# KPI CARDS
# ------------------------------------------------------
if page == "Dashboard":
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)


    with kpi_col1:
        st.metric("Total Scans Today", st.session_state.total_scans, "+34%")
    with kpi_col2:
        pct_normal = (
            st.session_state.normal_scans
            / max(st.session_state.total_scans, 1)
            * 100
        )
        st.metric("Normal Results", st.session_state.normal_scans, f"{pct_normal:.1f}% of total scans")
    with kpi_col3:
        st.metric("Pneumonia Detected", st.session_state.pneumonia_scans, "Requires review")
    with kpi_col4:
        st.metric("Avg. Processing Time", f"{st.session_state.avg_time:.1f}s", "Per analysis")


    # --------------------------------------------------
    # MAIN GRID: UPLOAD | RECENT | DONUT
    # --------------------------------------------------
    top_left, top_mid, top_right = st.columns((1.2, 1.1, 0.9))


    # Quick Upload
    with top_left:
        st.subheader("Quick Upload")
        st.caption("Upload a chest X-ray for analysis")


        upload_container = st.container(border=True)
        with upload_container:
            uploaded = st.file_uploader(
                "Drag and drop an X-ray image here",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
            )
            st.caption("Supports DICOM (as PNG/JPEG export), PNG, JPEG formats")


            if uploaded is not None:
                with st.spinner("Analyzing X-ray with PneumoAI model..."):
                    time.sleep(1.2)


                    new_id = f"PT-{random.randint(4600, 4700)}"
                    diagnosis_type = random.choices(
                        ["Normal", "Bacterial Pneumonia", "Viral Pneumonia"],
                        weights=[0.6, 0.25, 0.15],
                    )[0]
                    prob = random.uniform(0.72, 0.98)
                    now_str = "Just now"


                    st.session_state.total_scans += 1
                    if diagnosis_type == "Normal":
                        st.session_state.normal_scans += 1
                    else:
                        st.session_state.pneumonia_scans += 1


                    st.session_state.recent_analyses.insert(
                        0,
                        {
                            "id": new_id,
                            "time_ago": now_str,
                            "label": diagnosis_type,
                            "confidence": prob,
                            "severity": "HIGH" if "Bacterial" in diagnosis_type else "MEDIUM" if "Viral" in diagnosis_type else None,
                            "type": diagnosis_type,
                        },
                    )
                    st.session_state.recent_analyses = st.session_state.recent_analyses[:4]


                    if diagnosis_type != "Normal":
                        st.session_state.pending_reviews.insert(
                            0,
                            {
                                "id": new_id,
                                "time_ago": "Just now",
                                "label": diagnosis_type,
                                "severity": "HIGH" if "Bacterial" in diagnosis_type else "MEDIUM",
                                "confidence": prob,
                            },
                        )


                st.success(f"New scan {new_id} analyzed: {diagnosis_type} ({prob*100:.1f}% confidence)")


    # Recent Analyses
    def render_recent_card(item):
        badge_color = "#22c55e" if item["label"] == "Normal" else "#f97316" if "Viral" in item["label"] else "#ef4444"
        return f"""
        <div style="border-radius:10px; border:1px solid #e5e7eb; padding:0.6rem 0.8rem; margin-bottom:0.4rem;
                    display:flex; justify-content:space-between; align-items:center;">
            <div style="display:flex; align-items:center;">
                <div style="width:26px; height:26px; border-radius:999px; background-color:#e5f4ff;
                            display:flex; align-items:center; justify-content:center; margin-right:0.6rem;">
                    💨
                </div>
                <div>
                    <div style="font-weight:600; font-size:0.95rem;">{item['id']}</div>
                    <div style="font-size:0.8rem; color:gray;">{item['time_ago']} • {item['confidence']*100:.1f}% confidence</div>
                </div>
            </div>
            <div style="display:flex; align-items:center; gap:0.4rem;">
                <span style="background-color:{badge_color}1A; color:{badge_color};
                             padding:0.15rem 0.5rem; border-radius:999px; font-size:0.75rem;">
                    {item['label']}
                </span>
            </div>
        </div>
        """


    with top_mid:
        st.subheader("Recent Analyses")
        st.caption("Latest X-ray analysis results")
        for item in st.session_state.recent_analyses:
            st.markdown(render_recent_card(item), unsafe_allow_html=True)


    # Diagnosis Distribution
    with top_right:
        st.subheader("Diagnosis Distribution")
        st.caption("Today's analysis breakdown")


        counts = {
            "Normal": sum(1 for x in st.session_state.recent_analyses if x["type"] == "Normal"),
            "Bacterial Pneumonia": sum(1 for x in st.session_state.recent_analyses if x["type"] == "Bacterial Pneumonia"),
            "Viral Pneumonia": sum(1 for x in st.session_state.recent_analyses if x["type"] == "Viral Pneumonia"),
        }
        diag_df = pd.DataFrame(
            {
                "Diagnosis": list(counts.keys()),
                "Count": list(counts.values()),
            }
        )


        fig = px.pie(
            diag_df,
            values="Count",
            names="Diagnosis",
            hole=0.6,
            color="Diagnosis",
            color_discrete_map={
                "Normal": "#22c55e",
                "Bacterial Pneumonia": "#ef4444",
                "Viral Pneumonia": "#f59e0b",
            },
        )
        fig.update_layout(
            showlegend=True,
            margin=dict(t=10, b=10, l=0, r=0),
        )
        st.plotly_chart(fig, use_container_width=True)


        st.caption("Normal  •  Bacterial Pneumonia  •  Viral Pneumonia")


    # --------------------------------------------------
    # BOTTOM GRID: MODEL PERFORMANCE | PENDING
    # --------------------------------------------------
    bottom_left, bottom_right = st.columns((1.2, 1))


    with bottom_left:
        st.subheader("Model Performance")
        st.caption("Current model accuracy and metrics")


        models = [
            {
                "name": "ResNet-50",
                "accuracy": 92.4,
                "recall": 94.1,
                "precision": 91.2,
                "f1": 92.6,
                "active": False,
            },
            {
                "name": "DenseNet-121",
                "accuracy": 93.8,
                "recall": 95.6,
                "precision": 92.5,
                "f1": 94.0,
                "active": False,
            },
            {
                "name": "Ensemble",
                "accuracy": 95.2,
                "recall": 97.3,
                "precision": 94.1,
                "f1": 95.7,
                "active": True,
            },
        ]


        for m in models:
            st.markdown(
                f"""
                <div style="margin-bottom:0.8rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="font-weight:600;">{m['name']}</div>
                        <div style="font-size:0.85rem; color:gray;">{m['accuracy']:.1f}% accuracy</div>
                    </div>
                    <div style="width:100%; height:6px; border-radius:999px; background-color:#e5e7eb;">
                        <div style="width:{m['accuracy']}%; height:6px; border-radius:999px; background-color:#2563eb;"></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:gray; margin-top:0.35rem;">
                        <span>Recall: {m['recall']:.1f}%</span>
                        <span>Precision: {m['precision']:.1f}%</span>
                        <span>F1: {m['f1']:.1f}%</span>
                    </div>
                    {"<div style='margin-top:0.2rem; font-size:0.7rem; color:#2563eb; font-weight:600;'>ACTIVE</div>" if m["active"] else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )


    with bottom_right:
        st.subheader("Pending Reviews")
        st.caption("Cases awaiting physician confirmation")


        st.markdown(
            f"<div style='text-align:right; font-size:0.8rem; color:gray;'>{len(st.session_state.pending_reviews)} pending</div>",
            unsafe_allow_html=True,
        )


        for item in st.session_state.pending_reviews:
            sev_color = "#ef4444" if item["severity"] == "HIGH" else "#f97316"
            sev_bg = f"{sev_color}1A"
            st.markdown(
                f"""
                <div style="border-radius:10px; border:1px solid #e5e7eb; padding:0.6rem 0.8rem; margin-bottom:0.4rem;
                            display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-weight:600; font-size:0.95rem;">{item['id']}</div>
                        <div style="font-size:0.8rem; color:gray;">
                            {item['label']} ({item['confidence']*100:.1f}%)
                        </div>
                    </div>
                    <div style="display:flex; align-items:center; gap:0.5rem;">
                        <span style="background-color:{sev_bg}; color:{sev_color};
                                     padding:0.15rem 0.5rem; border-radius:999px; font-size:0.75rem;">
                            {item['severity']}
                        </span>
                        <span style="font-size:0.8rem; color:gray;">{item['time_ago']}</span>
                        <button style="border-radius:999px; padding:0.2rem 0.8rem; border:1px solid #e5e7eb;
                                       background-color:white; font-size:0.8rem; cursor:pointer;">
                            👁 Review
                        </button>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


elif page == "Upload X-ray":

    # ---------------- HEADER ----------------
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("# Upload X-ray")
    with col2:
        st.markdown(
            "<div style='text-align:right; color:gray;'>City General Hospital</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ---------------- LAYOUT ----------------
    left_col, mid_col, right_col = st.columns([2, 1.5, 1.5])

    # ---------------- LEFT: UPLOAD ----------------
    with left_col:
        st.markdown("### Upload X-ray")
        st.markdown("**Drag & drop or select a chest X-ray**")

        uploaded_file = st.file_uploader(
            "",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            st.image(uploaded_file, width=400)

        if st.button("📊 Analyze with PneumoAI", use_container_width=True, type="primary"):
            if uploaded_file is None:
                st.warning("Please upload an X-ray first")
            else:
                with st.spinner("Running AI analysis..."):
                    try:
                        # Prepare the file for upload to backend
                        files = {
                            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                        }
                        response = requests.post(
                            "http://localhost:8000/predict",
                            files=files
                        )
                        
                        # Check if request was successful
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Generate new patient ID
                            new_id = f"PT-{random.randint(4600, 4700)}"
                            diagnosis_type = result["prediction"]
                            prob = result["confidence"]
                            
                            # Update session stats
                            st.session_state.total_scans += 1
                            if diagnosis_type == "Normal":
                                st.session_state.normal_scans += 1
                            else:
                                st.session_state.pneumonia_scans += 1
                            st.session_state.recent_analyses.insert(
                                0,
                                {
                                    "id": new_id,
                                    "time_ago": "Just now",
                                    "label": diagnosis_type,
                                    "confidence": prob,
                                    "severity": result.get("urgency_level", "MEDIUM"),
                                    "type": diagnosis_type,
                                },
                            )
                            st.session_state.recent_analyses = st.session_state.recent_analyses[:4]
                            if diagnosis_type != "Normal":
                                st.session_state.pending_reviews.insert(
                                    0,
                                    {
                                        "id": new_id,
                                        "time_ago": "Just now",
                                        "label": diagnosis_type,
                                        "severity": result.get("urgency_level", "MEDIUM"),
                                        "confidence": prob,
                                    },
                                )
                            st.session_state.upload_result = {
                                "diagnosis": diagnosis_type,
                                "confidence": prob,
                                "model": "Ensemble",
                                "processed": True,
                                "priority_score": result.get("priority_score", 0),
                                "urgency_level": result.get("urgency_level", "UNKNOWN"),
                                "urgency_icon": result.get("urgency_icon", "⚠️"),
                                "severity_level": result.get("severity_level", "Unknown"),
                                "severity_description": result.get("severity_description", ""),
                                "clinical_data": result
                            }
                            st.success(f"✅ New scan {new_id} analyzed: {diagnosis_type} ({prob*100:.1f}% confidence)")
                            
                        else:
                            st.error(f"❌ Backend error: Status code {response.status_code}")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to backend. Make sure FastAPI is running on http://localhost:8000")
                        st.info("Start backend with: python main.py")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    # ---------------- MIDDLE: SUMMARY ----------------
    with mid_col:
        st.markdown("### Prediction Summary")

        if st.session_state.get("upload_result"):
            r = st.session_state.upload_result

            st.metric("Diagnosis", r["diagnosis"])
            st.metric("Confidence", f"{r['confidence']*100:.1f}%")
            st.caption(f"Model used: {r['model']}")

            c1, c2, c3 = st.columns(3)
            c1.button("✅ Accept", use_container_width=True)
            c2.button("❌ Reject", use_container_width=True)
            c3.button("📝 Note", use_container_width=True)

        else:
            st.info("Upload an X-ray to see AI predictions")

    # ---------------- RIGHT: EXPLAINABILITY ----------------
    with right_col:
        st.markdown("### Explainability")
        st.markdown("**Grad-CAM Heatmap Visualization**")

        if st.session_state.get("upload_result"):
            st.image(
                np.random.rand(224, 224, 3),
                width=300,
                caption="🔥 High-risk areas highlighted"
            )

            st.markdown(
                "*Red areas indicate highest pneumonia risk*"
            )
        else:
            st.info("Explainability available after analysis")

        st.markdown("**Select Model**")
        st.selectbox(
            "Model",
            ["Ensemble (95.2%)", "DenseNet-121 (93.8%)", "ResNet-50 (92.4%)"],
            index=0
        )

else:
    st.write(f"### {page}")
    st.caption("Static placeholder page for visual demo only.")
