import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Page configuration
st.set_page_config(
    page_title="AI-Assisted Pneumonia Detector",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #0891b2;
        --primary-light: #06b6d4;
        --accent: #14b8a6;
        --background: #f8fafc;
        --card-bg: #ffffff;
        --text-primary: #1e293b;
        --text-muted: #64748b;
        --border: #e2e8f0;
        --success: #22c55e;
        --warning: #f59e0b;
        --destructive: #ef4444;
    }
    
    /* Dark mode */
    [data-theme="dark"] {
        --background: #0f172a;
        --card-bg: #1e293b;
        --text-primary: #f1f5f9;
        --text-muted: #94a3b8;
        --border: #334155;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card styling */
    .metric-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }
    
    .metric-delta-positive {
        color: var(--success);
        font-size: 0.75rem;
    }
    
    .metric-delta-negative {
        color: var(--destructive);
        font-size: 0.75rem;
    }
    
    /* Status badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .badge-normal {
        background: #dcfce7;
        color: #166534;
    }
    
    .badge-pneumonia {
        background: #fee2e2;
        color: #991b1b;
    }
    
    .badge-uncertain {
        background: #fef3c7;
        color: #92400e;
    }
    
    .badge-pending {
        background: #e0f2fe;
        color: #075985;
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed var(--border);
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        background: var(--card-bg);
        transition: all 0.2s ease;
    }
    
    .upload-area:hover {
        border-color: var(--primary);
        background: rgba(8, 145, 178, 0.05);
    }
    
    /* Analysis item */
    .analysis-item {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        background: var(--background);
        border: 1px solid var(--border);
    }
    
    /* Header styling */
    .page-header {
        margin-bottom: 1.5rem;
    }
    
    .page-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }
    
    .page-subtitle {
        font-size: 0.875rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: var(--card-bg);
        border-right: 1px solid var(--border);
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: var(--primary);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom button styles */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
            
    .island {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sample data
@st.cache_data
def get_recent_analyses():
    statuses = ["Normal", "Pneumonia", "Uncertain"]
    return [
        {"id": "XR-2024-001", "patient": "John D.", "status": "Normal", "confidence": 94.2, "time": "2 min ago"},
        {"id": "XR-2024-002", "patient": "Sarah M.", "status": "Pneumonia", "confidence": 87.5, "time": "15 min ago"},
        {"id": "XR-2024-003", "patient": "Michael R.", "status": "Normal", "confidence": 91.8, "time": "32 min ago"},
        {"id": "XR-2024-004", "patient": "Emily K.", "status": "Uncertain", "confidence": 62.3, "time": "1 hr ago"},
        {"id": "XR-2024-005", "patient": "David L.", "status": "Pneumonia", "confidence": 89.1, "time": "2 hrs ago"},
    ]

@st.cache_data
def get_pending_reviews():
    return [
        {"id": "XR-2024-006", "patient": "Anna W.", "prediction": "Pneumonia", "confidence": 78.5, "priority": "High"},
        {"id": "XR-2024-007", "patient": "Robert T.", "prediction": "Uncertain", "confidence": 55.2, "priority": "Medium"},
        {"id": "XR-2024-008", "patient": "Lisa H.", "prediction": "Normal", "confidence": 68.9, "priority": "Low"},
    ]

def get_status_color(status):
    colors = {
        "Normal": "#22c55e",
        "Pneumonia": "#ef4444",
        "Uncertain": "#f59e0b",
        "Pending": "#0891b2"
    }
    return colors.get(status, "#64748b")

def get_badge_class(status):
    classes = {
        "Normal": "badge-normal",
        "Pneumonia": "badge-pneumonia",
        "Uncertain": "badge-uncertain",
        "Pending": "badge-pending"
    }
    return classes.get(status, "")

# Sidebar
with st.sidebar:
    st.markdown("""
<div style="
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    margin-bottom:2rem;
">
    <div style="
        width:48px;
        height:48px;
        background:linear-gradient(135deg,#0891b2,#14b8a6);
        border-radius:12px;
        display:flex;
        align-items:center;
        justify-content:center;
        margin-bottom:0.5rem;
    ">
        <span style="color:white;font-size:1.5rem;">🫁</span>
    </div>
    <span style="font-size:1.25rem;font-weight:700;">Consensus Scan</span>
</div>
""", unsafe_allow_html=True)
    
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    for tab in ["Dashboard", "Upload X-ray", "Batch Analysis", "Reports"]:
        if st.button(tab, use_container_width=True):
            st.session_state.page = tab

    page = st.session_state.page

    
    st.markdown("---")
    
    st.markdown("""
        <div style="padding: 1rem; background: #f1f5f9; border-radius: 8px; margin-top: 1rem;">
            <p style="font-size: 0.75rem; color: #64748b; margin: 0;">Institution</p>
            <p style="font-size: 0.875rem; font-weight: 500; color: #1e293b; margin: 0.25rem 0 0 0;">City General Hospital</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.expander("Settings"):
        st.selectbox("Model Version", ["ResNet-50 v2.1", "DenseNet-121", "EfficientNet-B4"])
        st.slider("Confidence Threshold", 0.5, 1.0, 0.75)
        st.toggle("Show Grad-CAM Overlay", value=True)

# Main content based on page selection
if page == "Dashboard":
    # Header
    st.markdown("""
        <div class="page-header">
            <h1 class="page-title">Dashboard</h1>
            <p class="page-subtitle">Overview of AI-assisted pneumonia detection system</p>
        </div>
    """, unsafe_allow_html=True)
    
    # toggle
    col_spacer, col_toggle, col_bell = st.columns([8,1,1])

    with col_toggle:
        st.toggle("🌙", help="Light / Dark Mode")

    with col_bell:
        st.button("🔔", help="Notifications")


    # Stats Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Scans Today",
            value="147",
            delta="+12% from yesterday"
        )
    
    with col2:
        st.metric(
            label="Normal Results",
            value="89",
            delta="60.5%"
        )
    
    with col3:
        st.metric(
            label="Pneumonia Detected",
            value="52",
            delta="35.4%"
        )
    
    with col4:
        st.metric(
            label="Avg. Processing Time",
            value="1.2s",
            delta="-0.3s improvement"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main dashboard content
    col_left, col_center, col_right = st.columns([1, 1, 1])
    
    with col_left:
        st.markdown('<div class="island">', unsafe_allow_html=True)
        st.markdown("### Quick Upload")
        st.markdown('</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag and drop X-ray image",
            type=["png", "jpg", "jpeg", "dcm"],
            help="Supported formats: PNG, JPG, JPEG, DICOM"
        )
        
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded X-ray", use_container_width=True)
            if st.button("Analyze Now", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    import time
                    time.sleep(2)
                st.success("Analysis complete!")
        else:
            st.info("Upload a chest X-ray image to begin analysis")
    
    with col_center:
        st.markdown('<div class="island">', unsafe_allow_html=True)
        st.markdown("### Recent Analysis")
        st.markdown('</div>', unsafe_allow_html=True)
        
        analyses = get_recent_analyses()
        for analysis in analyses:
            status_color = get_status_color(analysis["status"])
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem; background: #f8fafc; border-radius: 8px; margin-bottom: 0.5rem; border: 1px solid #e2e8f0;">
                    <div>
                        <p style="font-weight: 500; margin: 0; color: #1e293b;">{analysis["id"]}</p>
                        <p style="font-size: 0.75rem; color: #64748b; margin: 0;">{analysis["patient"]} • {analysis["time"]}</p>
                    </div>
                    <div style="text-align: right;">
                        <span style="background: {status_color}20; color: {status_color}; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 500;">{analysis["status"]}</span>
                        <p style="font-size: 0.75rem; color: #64748b; margin: 0.25rem 0 0 0;">{analysis["confidence"]}%</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.button("View All Analyses", use_container_width=True)
    
    with col_right:
        st.markdown('<div class="island">', unsafe_allow_html=True)
        st.markdown("### Diagnosis Distribution")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Donut chart
        distribution_data = pd.DataFrame({
            "Diagnosis": ["Normal", "Pneumonia", "Uncertain"],
            "Count": [89, 52, 6]
        })
        
        fig = px.pie(
            distribution_data, 
            values="Count", 
            names="Diagnosis",
            hole=0.6,
            color="Diagnosis",
            color_discrete_map={
                "Normal": "#22c55e",
                "Pneumonia": "#ef4444",
                "Uncertain": "#f59e0b"
            }
        )
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            margin=dict(t=20, b=20, l=20, r=20),
            height=250
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Bottom section
    col_bottom_left, col_bottom_right = st.columns(2)
    
    with col_bottom_left:
        st.markdown('<div class="island">', unsafe_allow_html=True)
        st.markdown("### Model Performance")
        st.markdown('</div>', unsafe_allow_html=True)
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.markdown("""
                <div style="text-align: center; padding: 1rem; background: #f0fdf4; border-radius: 8px;">
                    <p style="font-size: 1.5rem; font-weight: 700; color: #166534; margin: 0;">94.2%</p>
                    <p style="font-size: 0.75rem; color: #166534; margin: 0;">Sensitivity</p>
                </div>
            """, unsafe_allow_html=True)
        
        with metrics_col2:
            st.markdown("""
                <div style="text-align: center; padding: 1rem; background: #eff6ff; border-radius: 8px;">
                    <p style="font-size: 1.5rem; font-weight: 700; color: #1e40af; margin: 0;">91.8%</p>
                    <p style="font-size: 0.75rem; color: #1e40af; margin: 0;">Specificity</p>
                </div>
            """, unsafe_allow_html=True)
        
        with metrics_col3:
            st.markdown("""
                <div style="text-align: center; padding: 1rem; background: #faf5ff; border-radius: 8px;">
                    <p style="font-size: 1.5rem; font-weight: 700; color: #6b21a8; margin: 0;">0.96</p>
                    <p style="font-size: 0.75rem; color: #6b21a8; margin: 0;">AUC-ROC</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### ROC Curve")

        roc_x = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
        roc_y = [0, 0.75, 0.85, 0.92, 0.96, 0.98, 1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
        x=roc_x,
        y=roc_y,
        mode="lines",
        line=dict(width=3, color="#0891b2"),
        name="Super Model (AUC = 0.96)"
        ))
        fig.add_trace(go.Scatter(
            x=[0,1],
            y=[0,1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Random"
        ))

        fig.update_layout(
        height=300,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
        )

        st.plotly_chart(fig, use_container_width=True)

    
    with col_bottom_right:
        st.markdown('<div class="island">', unsafe_allow_html=True)
        st.markdown("### Pending Physician Review")
        st.markdown('</div>', unsafe_allow_html=True)
        
        pending = get_pending_reviews()
        for item in pending:
            priority_color = "#ef4444" if item["priority"] == "High" else "#f59e0b" if item["priority"] == "Medium" else "#22c55e"
            pred_color = get_status_color(item["prediction"])
            
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 1rem; background: #f8fafc; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid {priority_color};">
                    <div>
                        <p style="font-weight: 500; margin: 0; color: #1e293b;">{item["id"]}</p>
                        <p style="font-size: 0.75rem; color: #64748b; margin: 0;">{item["patient"]}</p>
                    </div>
                    <div style="text-align: center;">
                        <span style="background: {pred_color}20; color: {pred_color}; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 500;">{item["prediction"]}</span>
                        <p style="font-size: 0.75rem; color: #64748b; margin: 0.25rem 0 0 0;">{item["confidence"]}% confidence</p>
                    </div>
                    <div>
                        <span style="background: {priority_color}20; color: {priority_color}; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.625rem; font-weight: 600;">{item["priority"]}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.button("Review All", use_container_width=True, type="primary")
        with col_btn2:
            st.button("Export List", use_container_width=True)

elif page == "Upload X-ray":
    st.markdown("""
        <div class="page-header">
            <h1 class="page-title">Upload X-ray</h1>
            <p class="page-subtitle">Upload chest X-ray images for AI-assisted pneumonia detection</p>
        </div>
    """, unsafe_allow_html=True)
    
    col_upload, col_result = st.columns([1, 1])
    
    with col_upload:
        st.markdown('<div class="island">', unsafe_allow_html=True)
        st.markdown("### Image Upload")
        st.markdown('</div>', unsafe_allow_html=True)
        
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Chest X-ray",
            type=["png", "jpg", "jpeg", "dcm"],
            help="Supported formats: PNG, JPG, JPEG, DICOM. Max size: 10MB"
        )
        run_btn = st.button("▶ Run", type="primary", use_container_width=True)

        if run_btn and uploaded_file:
            with st.spinner("Running AI inference..."):
                import time
            time.sleep(2)
            st.session_state.analysis_complete = True

        
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded X-ray Image", use_container_width=True)
            
            # Patient info (optional)
            with st.expander("Patient Information (Optional)"):
                patient_id = st.text_input("Patient ID")
                patient_age = st.number_input("Age", min_value=0, max_value=120, value=45)
                patient_sex = st.selectbox("Sex", ["Male", "Female", "Other"])
            
            analyze_btn = st.button("Run Analysis", type="primary", use_container_width=True)
            
            if analyze_btn:
                with st.spinner("Analyzing X-ray image..."):
                    import time
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)
                    
                st.session_state.analysis_complete = True
                st.session_state.result = {
                    "prediction": "Pneumonia",
                    "confidence": 87.5,
                    "normal_prob": 12.5,
                    "pneumonia_prob": 87.5
                }
        else:
            st.markdown("""
                <div style="border: 2px dashed #e2e8f0; border-radius: 12px; padding: 3rem; text-align: center; background: #f8fafc;">
                    <p style="font-size: 3rem; margin: 0;">📤</p>
                    <p style="font-weight: 500; color: #1e293b; margin: 0.5rem 0;">Drag and drop your X-ray image here</p>
                    <p style="font-size: 0.875rem; color: #64748b; margin: 0;">or click to browse files</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col_result:
        st.markdown('<div class="island">', unsafe_allow_html=True)
        st.markdown("### Analysis Results")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'analysis_complete') and st.session_state.analysis_complete:
            result = st.session_state.result
            
            # Main prediction
            pred_color = "#ef4444" if result["prediction"] == "Pneumonia" else "#22c55e"
            st.markdown(f"""
                <div style="background: {pred_color}10; border: 1px solid {pred_color}40; border-radius: 12px; padding: 1.5rem; text-align: center; margin-bottom: 1rem;">
                    <p style="font-size: 0.875rem; color: #64748b; margin: 0;">AI Prediction</p>
                    <p style="font-size: 2rem; font-weight: 700; color: {pred_color}; margin: 0.5rem 0;">{result["prediction"]}</p>
                    <p style="font-size: 1rem; color: #1e293b; margin: 0;">{result["confidence"]}% Confidence</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Probability breakdown
            st.markdown('<div class="island">', unsafe_allow_html=True)
            st.markdown("### Probability Breakdown")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span style="font-size: 0.875rem; color: #1e293b;">Normal</span>
                        <span style="font-size: 0.875rem; font-weight: 500; color: #22c55e;">{result["normal_prob"]}%</span>
                    </div>
                    <div style="background: #e2e8f0; border-radius: 9999px; height: 8px; overflow: hidden;">
                        <div style="background: #22c55e; height: 100%; width: {result["normal_prob"]}%;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span style="font-size: 0.875rem; color: #1e293b;">Pneumonia</span>
                        <span style="font-size: 0.875rem; font-weight: 500; color: #ef4444;">{result["pneumonia_prob"]}%</span>
                    </div>
                    <div style="background: #e2e8f0; border-radius: 9999px; height: 8px; overflow: hidden;">
                        <div style="background: #ef4444; height: 100%; width: {result["pneumonia_prob"]}%;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Grad-CAM section
            st.markdown('<div class="island">', unsafe_allow_html=True)
            st.markdown("### Grad CAM Explainability")
            st.markdown('</div>', unsafe_allow_html=True)
            st.info("The heatmap overlay shows regions that most influenced the AI's prediction. Warmer colors (red/orange) indicate areas of higher attention.")
            
            # Placeholder for Grad-CAM visualization
            st.markdown("""
                <div style="background: linear-gradient(135deg, #fef3c7, #fee2e2, #fecaca); border-radius: 8px; height: 200px; display: flex; align-items: center; justify-content: center;">
                    <p style="color: #92400e; font-weight: 500;">Grad-CAM Heatmap Overlay</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Action buttons
            col_action1, col_action2, col_action3 = st.columns(3)
            with col_action1:
                st.button("Download Report", use_container_width=True)
            with col_action2:
                st.button("Request Review", use_container_width=True, type="primary")
            with col_action3:
                st.button("New Analysis", use_container_width=True)
            
            # Disclaimer
            st.markdown("""
                <div style="background: #fef3c7; border: 1px solid #fcd34d; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                    <p style="font-size: 0.75rem; color: #92400e; margin: 0;">
                        <strong>Disclaimer:</strong> This AI system is intended for use as a second-opinion diagnostic support tool only. 
                        All results must be reviewed and confirmed by a qualified radiologist or physician before clinical decisions are made.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 400px; background: #f8fafc; border-radius: 12px; border: 1px solid #e2e8f0;">
                    <p style="font-size: 3rem; margin: 0;">🔬</p>
                    <p style="font-weight: 500; color: #1e293b; margin: 1rem 0 0.5rem 0;">No Analysis Yet</p>
                    <p style="font-size: 0.875rem; color: #64748b; text-align: center; max-width: 300px;">Upload an X-ray image and click "Run Analysis" to see AI predictions</p>
                </div>
            """, unsafe_allow_html=True)

elif page == "Batch Analysis":
    st.markdown("""
        <div class="page-header">
            <h1 class="page-title">Batch Analysis</h1>
            <p class="page-subtitle">Process multiple X-ray images simultaneously</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload Multiple X-rays",
        type=["png", "jpg", "jpeg", "dcm"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.markdown('<div class="island">', unsafe_allow_html=True)
        st.markdown(f"### {len(uploaded_files)} Images Selected")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display thumbnails in a grid
            cols = st.columns(4)
            for idx, file in enumerate(uploaded_files[:8]):
                with cols[idx % 4]:
                    st.image(file, caption=file.name[:15], use_container_width=True)
        
        with col2:
            st.markdown("#### Batch Settings")
            st.selectbox("Model", ["ResNet-50 v2.1", "Ensemble"])
            st.checkbox("Generate individual reports", value=True)
            st.checkbox("Flag high-risk cases", value=True)
            
            if st.button("Start Batch Analysis", type="primary", use_container_width=True):
                progress = st.progress(0)
                status = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status.text(f"Analyzing {file.name}...")
                    import time
                    time.sleep(0.5)
                    progress.progress((i + 1) / len(uploaded_files))
                
                st.success(f"Batch analysis complete! Processed {len(uploaded_files)} images.")
    else:
        st.info("Upload multiple X-ray images to begin batch processing")

 
    # Model performance comparison
elif page == "Reports":
    st.markdown("""
            <div class="page-header">
            <h1 class="page-title">Reports</h1>
            <p class="page-subtitle">Generate and download analysis reports</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Date range filter
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    with col3:
        report_type = st.selectbox("Report Type", ["Summary Report", "Detailed Analysis", "Model Performance", "Audit Log"])
    
    st.markdown("---")
    
    # Report preview
    st.markdown("### Report Preview")
    
    st.markdown("### Diagnosis Summary")

    df = pd.DataFrame({
        "Diagnosis": ["Normal", "Pneumonia", "Uncertain"],
        "Cases": [512, 298, 37],
        "Percentage": ["60.4%", "35.2%", "4.4%"]
    })

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        st.button("Download PDF", type="primary", use_container_width=True)
    with col_btn2:
        st.button("Download CSV", use_container_width=True)
    with col_btn3:
        st.button("Email Report", use_container_width=True)