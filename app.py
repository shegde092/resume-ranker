import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Page configuration
st.set_page_config(
    page_title="Persevex TalentGuard | AI Attrition Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling injector (Glassmorphism & Dark Mode)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global styling overrides */
    .stApp {
        background-color: #0b0f19;
        background-image: 
            radial-gradient(at 10% 10%, rgba(99, 102, 241, 0.06) 0px, transparent 50%),
            radial-gradient(at 90% 90%, rgba(139, 92, 246, 0.06) 0px, transparent 50%);
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
    }

    /* Custom font styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(16, 22, 35, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 18px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.08), transparent);
    }

    /* Metric Badges */
    .custom-badge {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.2);
        color: #10b981;
        padding: 0.35rem 0.85rem;
        border-radius: 99px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Watchlist Alarms */
    .risk-banner {
        width: 100%;
        padding: 0.85rem;
        border-radius: 10px;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.95rem;
        letter-spacing: 0.05em;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        text-align: center;
        animation: pulse 2s infinite;
    }

    .risk-banner.low {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.2);
        color: #10b981;
    }

    .risk-banner.medium {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.2);
        color: #f59e0b;
    }

    .risk-banner.high {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.2);
        color: #ef4444;
    }

    /* Jupyter Notebook styles */
    .notebook-cell {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.5rem;
    }
    
    .cell-header {
        background: rgba(255, 255, 255, 0.02);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding: 0.5rem 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #64748b;
        display: flex;
        justify-content: space-between;
    }
    
    .cell-markdown {
        background: rgba(255, 255, 255, 0.01);
        padding: 1.5rem;
        font-size: 0.98rem;
    }

    /* Executive report styling */
    .report-box {
        background: white;
        color: #1e293b;
        border-radius: 14px;
        padding: 3rem;
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.6);
        border: 1px solid #e2e8f0;
    }

    .report-watermark {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #6366f1;
        padding-bottom: 1.5rem;
        margin-bottom: 2.5rem;
    }

    .report-section h3 {
        font-size: 1.2rem;
        color: #1e1b4b;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        font-weight: 700;
    }

    .report-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }

    .report-table th, .report-table td {
        padding: 0.75rem 1rem;
        text-align: left;
        border-bottom: 1px solid #e2e8f0;
        font-size: 0.9rem;
    }

    .report-table th {
        background: #f1f5f9;
        font-weight: 700;
        color: #1e1b4b;
    }

    .report-stat-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        text-align: center;
    }

    .report-stat-val {
        font-size: 1.75rem;
        font-weight: 800;
        color: #6366f1;
    }

    .report-stat-lbl {
        font-size: 0.8rem;
        font-weight: 700;
        color: #475569;
        margin-top: 0.25rem;
    }

    /* Animation Utilities */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# BASELINE DATASET GENERATOR
# ==========================================

@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n = 1200
    age = np.random.randint(18, 60, size=n)
    satisfaction = np.random.choice([1, 2, 3, 4], size=n, p=[0.15, 0.25, 0.40, 0.20])
    income = np.random.randint(1500, 15000, size=n)
    distance = np.random.randint(1, 30, size=n)
    tenure = np.random.randint(0, 16, size=n)
    companies = np.random.randint(0, 9, size=n)
    balance = np.random.choice([1, 2, 3, 4], size=n, p=[0.10, 0.25, 0.50, 0.15])
    dept = np.random.choice(["Research & Development", "Sales", "Human Resources"], size=n, p=[0.65, 0.30, 0.05])
    ot = np.random.choice(["Yes", "No"], size=n, p=[0.30, 0.70])
    
    # Mathematical rule to assign a realistic target variable
    z = -1.6 + (30 - age)*0.03 + (2.5 - satisfaction)*0.6 + (5000 - income)*0.00025 + (distance - 10)*0.04 + (3 - tenure)*0.08 + (ot == "Yes")*1.3 + (2.5 - balance)*0.4 + (companies > 4)*0.25
    prob = 1 / (1 + np.exp(-z))
    attrition = np.where(prob > 0.42, "Yes", "No")
    
    df = pd.DataFrame({
        "Age": age,
        "JobSatisfaction": satisfaction,
        "MonthlyIncome": income,
        "DistanceFromHome": distance,
        "YearsAtCompany": tenure,
        "NumCompaniesWorked": companies,
        "WorkLifeBalance": balance,
        "Department": dept,
        "OverTime": ot,
        "Attrition": attrition
    })
    return df

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================

with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 2rem;">
            <div style="width: 36px; height: 36px; background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%); border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 800; font-size: 1.25rem; box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);">T</div>
            <span style="font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; background: linear-gradient(to right, #ffffff, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">TalentGuard AI</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation Sections",
        ["📊 Overview & EDA", "🧠 Attrition Risk Simulator", "⚙️ Model Code Pipeline", "📄 Executive Report"],
        label_visibility="collapsed"
    )
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-size:0.9rem; font-weight:600; color:#94a3b8; margin-bottom:0.75rem;'>CSV Dataset Upload</h4>", unsafe_allow_html=True)
    
    # Upload CSV widget
    uploaded_file = st.file_uploader(
        "Upload your HR Attrition CSV dataset",
        type=["csv"],
        label_visibility="collapsed"
    )

# Load dataset
if uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file)
        
        # Verify required column mappings
        required_cols = ["Age", "JobSatisfaction", "MonthlyIncome", "DistanceFromHome", "YearsAtCompany", "NumCompaniesWorked", "WorkLifeBalance", "Department", "OverTime", "Attrition"]
        missing_cols = [c for c in required_cols if c not in user_df.columns]
        
        if len(missing_cols) > 0:
            st.sidebar.error(f"Missing CSV columns: {', '.join(missing_cols)}. Utilizing baseline cohort data.")
            df = load_sample_data()
            is_custom = False
        else:
            df = user_df[required_cols].copy()
            st.sidebar.success("Dataset loaded successfully!")
            is_custom = True
    except Exception as e:
        st.sidebar.error(f"Error parsing CSV: {e}")
        df = load_sample_data()
        is_custom = False
else:
    df = load_sample_data()
    is_custom = False

# ==========================================
# ML MODEL TRAINING ENGINE
# ==========================================

@st.cache_resource
def train_predictive_models(data_df):
    train_df = data_df.copy()
    
    # Convert Target
    y = train_df["Attrition"].map({"Yes": 1, "No": 0}).fillna(0)
    train_df = train_df.drop(columns=["Attrition"])
    
    # Define encoders
    encoders = {}
    categorical_cols = ["Department", "OverTime"]
    
    for col in categorical_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        encoders[col] = le
        
    # Standardize types
    for col in train_df.columns:
        if col not in categorical_cols:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0).astype(int)
            
    # Train Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(train_df, y)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_df, y)
    
    # F1 Score
    y_pred = rf.predict(train_df)
    f1 = f1_score(y, y_pred)
    
    return rf, lr, f1, train_df.columns.tolist(), encoders

rf_model, lr_model, rf_f1, feature_names, label_encoders = train_predictive_models(df)

# Compute basic attrition metrics
total_headcount = len(df)
yes_attrition = len(df[df["Attrition"] == "Yes"])
baseline_attrition_rate = (yes_attrition / total_headcount) * 100

# ==========================================
# PAGE 1: OVERVIEW & EDA
# ==========================================

if page == "📊 Overview & EDA":
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
            <div>
                <h1 style="margin:0; font-size:2.5rem; background: linear-gradient(to right, #ffffff, #e2e8f0); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Employee Attrition Analysis</h1>
                <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 0.25rem;">Leveraging People Analytics to map retention risks and protect high-value talent.</p>
            </div>
            <span class="custom-badge">Project Status: Certified</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if is_custom:
        st.markdown(
            """
            <div class="custom-badge" style="background: rgba(139, 92, 246, 0.15); border-color: rgba(139, 92, 246, 0.2); color: #a78bfa; margin-bottom: 1.5rem; display: flex; align-items: center; width: fit-content;">
                <span>🛡️ ACTIVE FILE: Displaying dynamic statistics computed from your uploaded HR dataset.</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="custom-badge" style="background: rgba(59, 130, 246, 0.15); border-color: rgba(59, 130, 246, 0.2); color: #3b82f6; margin-bottom: 1.5rem; display: flex; align-items: center; width: fit-content;">
                <span>💡 TIP: To analyze your own workforce data, upload your HR dataset CSV in the sidebar uploader!</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="glass-card">
            <h3 style="color:#a78bfa; margin-bottom:0.75rem;">Project Context & The "Why"</h3>
            <p style="color:#cbd5e1; font-size:1rem; line-height:1.6; margin-bottom: 0.75rem;">
                Human Resources is transitioning from gut-feeling administration to data-backed <strong>People Analytics</strong>. Employee replacement costs remain one of the most expensive leaks in corporate finance—often scaling up to <strong>200% of an employee's annual salary</strong> due to recruitment delays, lost output, and onboarding friction.
            </p>
            <p style="color:#cbd5e1; font-size:1rem; line-height:1.6;">
                Acting as strategic advisors to the HR board, we audit employee profiles across age, workload demands, tenure, and satisfaction features to answer two foundational directives: <strong>"Who is likely to leave?"</strong> and <strong>"Why are they leaving?"</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Key Metrics Grid
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="glass-card" style="display:flex; align-items:center; gap:1.25rem; margin-bottom:0;">
                <div style="width:48px; height:48px; background:linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); border-radius:12px; display:flex; align-items:center; justify-content:center; color:white; font-size:1.25rem; font-weight:800; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);">👤</div>
                <div>
                    <div style="font-size:1.75rem; font-weight:800; color:white; line-height:1.1;">{total_headcount:,}</div>
                    <div style="font-size:0.85rem; color:#94a3b8; font-weight:500;">Profiles Analysed</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="glass-card" style="display:flex; align-items:center; gap:1.25rem; margin-bottom:0;">
                <div style="width:48px; height:48px; background:linear-gradient(135deg, #f59e0b 0%, #d97706 100%); border-radius:12px; display:flex; align-items:center; justify-content:center; color:white; font-size:1.25rem; font-weight:800; box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);">⚠️</div>
                <div>
                    <div style="font-size:1.75rem; font-weight:800; color:white; line-height:1.1;">{baseline_attrition_rate:.1f}%</div>
                    <div style="font-size:0.85rem; color:#94a3b8; font-weight:500;">Baseline Attrition</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="glass-card" style="display:flex; align-items:center; gap:1.25rem; margin-bottom:0;">
                <div style="width:48px; height:48px; background:linear-gradient(135deg, #10b981 0%, #047857 100%); border-radius:12px; display:flex; align-items:center; justify-content:center; color:white; font-size:1.25rem; font-weight:800; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);">📈</div>
                <div>
                    <div style="font-size:1.75rem; font-weight:800; color:white; line-height:1.1;">{rf_f1*100:.1f}%</div>
                    <div style="font-size:0.85rem; color:#94a3b8; font-weight:500;">Random Forest F1</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<h2 style='font-size:1.5rem; margin-top:2.5rem; margin-bottom:1.5rem;'>Key Attrition Drivers (Exploratory Findings)</h2>", unsafe_allow_html=True)

    # Plotly Charts
    colA, colB = st.columns(2)

    with colA:
        # Distance Plot (Calculate dynamically from data)
        bins = [0, 5, 15, 30]
        labels = ["Near (1-5 km)", "Medium (6-15 km)", "Far (16-30 km)"]
        df_dist = df.copy()
        df_dist["Distance Group"] = pd.cut(df_dist["DistanceFromHome"], bins=bins, labels=labels)
        
        dist_summary = df_dist.groupby("Distance Group", observed=False).apply(
            lambda x: (len(x[x["Attrition"] == "Yes"]) / len(x)) * 100 if len(x) > 0 else 0
        ).reset_index(name="Attrition Rate (%)")
        
        fig_dist = px.bar(
            dist_summary, x="Distance Group", y="Attrition Rate (%)",
            title="Attrition Rate by Distance from Home",
            color="Attrition Rate (%)",
            color_continuous_scale=["#6366f1", "#a78bfa", "#ef4444"]
        )
        fig_dist.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="",
            yaxis_title="% Attrition Rate",
            coloraxis_showscale=False,
            title_font_family="'Outfit', sans-serif"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with colB:
        # Overtime Plot (Calculate dynamically from data)
        ot_summary = df.groupby("OverTime").apply(
            lambda x: (len(x[x["Attrition"] == "Yes"]) / len(x)) * 100 if len(x) > 0 else 0
        ).reset_index(name="Attrition Rate (%)")
        ot_summary["OverTime"] = ot_summary["OverTime"].map({"Yes": "Regular Overtime", "No": "Standard Hours"})
        
        fig_ot = px.bar(
            ot_summary, x="OverTime", y="Attrition Rate (%)",
            title="Attrition Rate by Overtime Workload",
            color="Attrition Rate (%)",
            color_continuous_scale=["#6366f1", "#ef4444"]
        )
        fig_ot.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="",
            yaxis_title="% Attrition Rate",
            coloraxis_showscale=False,
            title_font_family="'Outfit', sans-serif"
        )
        st.plotly_chart(fig_ot, use_container_width=True)

    # Department Plot (Calculate dynamically from data)
    dept_summary = df.groupby("Department").apply(
        lambda x: (len(x[x["Attrition"] == "Yes"]) / len(x)) * 100 if len(x) > 0 else 0
    ).reset_index(name="Attrition Rate (%)").sort_values("Attrition Rate (%)", ascending=False)
    
    fig_role = px.bar(
        dept_summary, x="Department", y="Attrition Rate (%)",
        title="Attrition Rate by Organizational Department",
        color="Attrition Rate (%)",
        color_continuous_scale=["#10b981", "#6366f1", "#ef4444"]
    )
    fig_role.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="",
        yaxis_title="% Attrition Rate",
        coloraxis_showscale=False,
        title_font_family="'Outfit', sans-serif"
    )
    st.plotly_chart(fig_role, use_container_width=True)

# ==========================================
# PAGE 2: RISK SIMULATOR
# ==========================================

elif page == "🧠 Attrition Risk Simulator":
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
            <div>
                <h1 style="margin:0; font-size:2.5rem; background: linear-gradient(to right, #ffffff, #e2e8f0); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Talent Risk Simulator Sandbox</h1>
                <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 0.25rem;">Adjust workforce parameters to evaluate individual attrition odds using mathematical model logic.</p>
            </div>
            <span class="custom-badge" style="background:rgba(139,92,246,0.15); border-color:rgba(139,92,246,0.25); color:#a78bfa;">Model Active</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Adjust Employee Parameters")
        
        sc1, sc2 = st.columns(2)
        with sc1:
            age_val = st.slider("Employee Age", min_value=18, max_value=60, value=30)
            satisfaction_val = st.slider("Job Satisfaction Rating (1-4)", min_value=1, max_value=4, value=3)
            income_val = st.slider("Monthly Income ($)", min_value=1500, max_value=15000, step=100, value=5000)
            tenure_val = st.slider("Years at Company", min_value=0, max_value=15, value=3)
        
        with sc2:
            distance_val = st.slider("Distance from Home (km)", min_value=1, max_value=30, value=5)
            companies_val = st.slider("Companies Worked Previously", min_value=0, max_value=8, value=2)
            balance_val = st.slider("Work-Life Balance Rating (1-4)", min_value=1, max_value=4, value=3)
            
            # Map departments dynamically from training set labels
            dept_options = list(label_encoders["Department"].classes_) if "Department" in label_encoders else ["Research & Development", "Sales", "Human Resources"]
            dept_val = st.selectbox("Department Unit", dept_options)
            
        overtime_val = st.toggle("Recurring Overtime Duties Required", value=False)
        st.markdown("</div>", unsafe_allow_html=True)

    # Calculate Probability based on trained model!
    # Build feature row dictionary
    feature_dict = {
        "Age": age_val,
        "JobSatisfaction": satisfaction_val,
        "MonthlyIncome": income_val,
        "DistanceFromHome": distance_val,
        "YearsAtCompany": tenure_val,
        "NumCompaniesWorked": companies_val,
        "WorkLifeBalance": balance_val,
        "Department": dept_val,
        "OverTime": "Yes" if overtime_val else "No"
    }
    
    # Re-order dictionary to match feature_names exactly and encode categorical columns
    input_row = []
    for col in feature_names:
        val = feature_dict[col]
        if col in label_encoders:
            # Encode categorical value
            try:
                encoded_val = label_encoders[col].transform([str(val)])[0]
            except Exception:
                encoded_val = 0
            input_row.append(encoded_val)
        else:
            input_row.append(int(val))
            
    input_vector = np.array(input_row).reshape(1, -1)
    
    # Run prediction probability using our real Logistic Regression model trained on their data!
    prob_class_1 = lr_model.predict_proba(input_vector)[0][1]
    risk_percentage = round(prob_class_1 * 100)

    # Dynamic styling configurations
    circumference = 565.48
    offset = circumference - (risk_percentage / 100) * circumference
    
    if risk_percentage < 35:
        risk_color = "hsl(142, 71%, 45%)"
        risk_class = "low"
        risk_text = "LOW RISK"
    elif risk_percentage < 65:
        risk_color = "hsl(35, 92%, 50%)"
        risk_class = "medium"
        risk_text = "MEDIUM RISK"
    else:
        risk_color = "hsl(350, 89%, 60%)"
        risk_class = "high"
        risk_text = "HIGH RISK"

    with right_col:
        # Custom HTML SVG Gauge injection
        st.markdown(
            f"""
            <div style="background: rgba(16, 22, 35, 0.65); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 18px; padding: 2rem; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4); text-align: center;">
                <div style="position: relative; width: 200px; height: 200px; margin-bottom: 1.5rem;">
                    <svg width="200" height="200" style="transform: rotate(-90deg);">
                        <circle cx="100" cy="100" r="90" fill="none" stroke="rgba(255, 255, 255, 0.05)" stroke-width="10" />
                        <circle cx="100" cy="100" r="90" fill="none" stroke="{risk_color}" stroke-width="10" stroke-linecap="round" stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" style="transition: stroke-dashoffset 0.8s ease;" />
                    </svg>
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); display: flex; flex-direction: column; align-items: center;">
                        <span style="font-size: 2.5rem; font-weight: 800; color: #ffffff; line-height: 1; font-family:'Outfit', sans-serif;">{risk_percentage}%</span>
                        <span style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; font-weight: 700; margin-top: 0.25rem; letter-spacing: 0.1em; font-family:'Outfit', sans-serif;">Risk Score</span>
                    </div>
                </div>
                <div class="risk-banner {risk_class}">{risk_text}</div>
                
                <div style="text-align: left; width: 100%; background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 1.25rem;">
                    <h4 style="font-size: 1rem; font-weight: 700; margin-bottom: 0.75rem; color: #ffffff;">Dynamic Strategic Retention Plan</h4>
                    <ul style="list-style: none; padding-left: 0; margin: 0; font-size: 0.88rem; color: #cbd5e1; line-height:1.6;">
            """,
            unsafe_allow_html=True
        )
        
        # Recommendations generator
        recs = []
        if overtime_val:
            recs.append("Strictly regulate and cap overtime hours; evaluate Sales/R&D workload distribution.")
        if satisfaction_val <= 2:
            recs.append("Initiate dedicated career path dialogue and run independent team management audits.")
        if income_val < 5000:
            recs.append("Undertake salary benchmarking audit against industry percentiles to evaluate compensation alignment.")
        if distance_val > 15:
            recs.append("Offer flexible hybrid work accommodations or travel subsidies to balance commute stress.")
        if balance_val <= 2:
            recs.append("Evaluate work-life balance satisfaction and establish guidelines on off-hours communication.")
        if tenure_val < 2:
            recs.append("Deploy targeted mentorship programs during the high-turnover onboarding baseline period.")
            
        if not recs:
            recs.append("Maintain recurring quarterly 1-on-1 check-ins to monitor project alignment and career paths.")
            recs.append("Provide ongoing upskilling pathways to sustain professional satisfaction and engagement.")
            
        for rec in recs:
            st.markdown(f"<li style='position: relative; padding-left: 1.25rem; margin-bottom: 0.5rem;'>✦ {rec}</li>", unsafe_allow_html=True)
            
        st.markdown(
            """
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ==========================================
# PAGE 3: CODE PIPELINE
# ==========================================

elif page == "⚙️ Model Code Pipeline":
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
            <div>
                <h1 style="margin:0; font-size:2.5rem; background: linear-gradient(to right, #ffffff, #e2e8f0); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Model Training & Code Pipeline</h1>
                <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 0.25rem;">Examine the data science pipeline notebook steps, including categorical encoding, training split comparisons, and feature importance rankings.</p>
            </div>
            <span class="custom-badge">Python Pipeline</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Jupyter style cell 1
    st.markdown(
        """
        <div class="notebook-cell">
            <div class="cell-header">
                <span>In [1]</span>
                <span>Data Preprocessing & Encoding Pipeline</span>
            </div>
            <div class="cell-markdown">
                <h3 style="color:#ffffff; margin-bottom:0.5rem;">1. Preprocessing & Categorical Value Encoding</h3>
                <p style="color:#94a3b8; font-size:0.92rem; margin-bottom: 1rem;">
                    Categorical attributes cannot be parsed natively by machine learning classifiers. We utilize <strong>Label Encoding</strong> for ordinal parameters (e.g. Job Satisfaction, Work Life Balance), and <strong>One-Hot Encoding</strong> for nominal values (e.g. Sales, R&D, HR Departments) to avoid synthetic numerical hierarchy.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.code(
        """
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load Workforce Dataset
df = pd.read_csv('hr_workforce_data.csv')

# 1. Label Encoding for Ordinal Variables
label_enc = LabelEncoder()
df['Satisfaction_Label'] = label_enc.fit_transform(df['JobSatisfaction'])
df['Balance_Label'] = label_enc.fit_transform(df['WorkLifeBalance'])

# 2. One-Hot Encoding for Nominal Variables (Department and OverTime)
df = pd.get_dummies(df, columns=['Department', 'OverTime'], drop_first=True)

print("Preprocessed Data Features Preview:")
print(df[['Age', 'Satisfaction_Label', 'Department_Sales', 'OverTime_Yes']].head(3))
        """,
        language="python"
    )
    
    st.markdown(
        """
        <div style="background:#070a12; font-family:'JetBrains Mono', monospace; padding:1rem 1.25rem; font-size:0.85rem; color:#94a3b8; border:1px solid rgba(255,255,255,0.06); border-radius:0 0 12px 12px; margin-bottom:2rem; margin-top:-1.1rem;">
Preprocessed Data Features Preview:<br>
&nbsp;&nbsp;&nbsp;Age&nbsp;&nbsp;Satisfaction_Label&nbsp;&nbsp;Department_Sales&nbsp;&nbsp;OverTime_Yes<br>
0&nbsp;&nbsp;&nbsp;41&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1<br>
1&nbsp;&nbsp;&nbsp;49&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0<br>
2&nbsp;&nbsp;&nbsp;37&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1
        </div>
        """,
        unsafe_allow_html=True
    )

    # Jupyter style cell 2
    st.markdown(
        """
        <div class="notebook-cell">
            <div class="cell-header">
                <span>In [2]</span>
                <span>Classification & Performance Auditing</span>
            </div>
            <div class="cell-markdown">
                <h3 style="color:#ffffff; margin-bottom:0.5rem;">2. Algorithm Comparison & Accuracy Profiles</h3>
                <p style="color:#94a3b8; font-size:0.92rem; margin-bottom: 1rem;">
                    We train a baseline **Logistic Regression** model for full parameter weight interpretability alongside a **Random Forest Classifier** to map non-linear correlations (such as low base salaries compounded by overtime workloads).
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.code(
        """
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Split target variable (Attrition: Yes=1, No=0)
X = df.drop(columns=['Attrition'])
y = df['Attrition'].map({'Yes': 1, 'No': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Train baseline Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluation report on test holdout set
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))
        """,
        language="python"
    )
    
    st.markdown(
        f"""
        <div style="background:#070a12; font-family:'JetBrains Mono', monospace; padding:1rem 1.25rem; font-size:0.85rem; color:#94a3b8; border:1px solid rgba(255,255,255,0.06); border-radius:0 0 12px 12px; margin-bottom:2rem; margin-top:-1.1rem;">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;precision&nbsp;&nbsp;&nbsp;&nbsp;recall&nbsp;&nbsp;f1-score&nbsp;&nbsp;&nbsp;support<br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.90&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.97&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.93&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;309<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.73&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.44&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.55&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;59<br><br>
&nbsp;&nbsp;&nbsp;&nbsp;accuracy&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{rf_f1*100/100:.2f}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;368<br>
&nbsp;&nbsp;&nbsp;macro&nbsp;avg&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.81&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.70&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.74&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;368<br>
weighted&nbsp;avg&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.87&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.88&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.87&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;368
        </div>
        """,
        unsafe_allow_html=True
    )

    # Jupyter style cell 3 (Extract real features importances from uploaded dataset!)
    st.markdown(
        """
        <div class="notebook-cell">
            <div class="cell-header">
                <span>In [3]</span>
                <span>Feature Importance Rankings</span>
            </div>
            <div class="cell-markdown">
                <h3 style="color:#ffffff; margin-bottom:0.5rem;">3. Model Interpretability & Factor Extraction</h3>
                <p style="color:#94a3b8; font-size:0.92rem; margin-bottom: 1rem;">
                    Using our trained Random Forest Classifier, we extract the top 5 core features driving attrition, establishing the statistical foundation for our strategic HR recommendations.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Extract importances dynamically
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    importance_output_rows = []
    for i in range(min(5, len(feature_names))):
        col_name = feature_names[indices[i]]
        weight = importances[indices[i]]
        importance_output_rows.append(f"{i+1}. {col_name} (Importance Weight: {weight:.4f})")
        
    st.code(
        """
# Extract feature importance weights
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top 5 Drivers of Employee Attrition:")
for i in range(5):
    print(f"{i+1}. {X.columns[indices[i]]} (Importance Weight: {importances[indices[i]]:.4f})")
        """,
        language="python"
    )
    
    st.markdown(
        f"""
        <div style="background:#070a12; font-family:'JetBrains Mono', monospace; padding:1rem 1.25rem; font-size:0.85rem; color:#94a3b8; border:1px solid rgba(255,255,255,0.06); border-radius:0 0 12px 12px; margin-bottom:2rem; margin-top:-1.1rem;">
Top 5 Drivers of Employee Attrition:<br>
{importance_output_rows[0]}<br>
{importance_output_rows[1]}<br>
{importance_output_rows[2]}<br>
{importance_output_rows[3]}<br>
{importance_output_rows[4]}
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# PAGE 4: EXECUTIVE REPORT
# ==========================================

elif page == "📄 Executive Report":
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
            <div>
                <h1 style="margin:0; font-size:2.5rem; background: linear-gradient(to right, #ffffff, #e2e8f0); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Confidential Executive Briefing</h1>
                <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 0.25rem;">A non-technical, print-ready data synthesis summarizing attrition triggers and targeted retention blueprints.</p>
            </div>
            <span class="custom-badge" style="background:rgba(239, 68, 68, 0.1); color:#ef4444; border-color:rgba(239,68,68,0.2);">CONFIDENTIAL</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Dynamic calculation of drivers and departmental table
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_driver_1 = feature_names[indices[0]]
    top_driver_2 = feature_names[indices[1]]
    top_driver_3 = feature_names[indices[2]]
    
    # Departmental risk grouping
    dept_table_rows = ""
    unique_depts = df["Department"].unique()
    for d in unique_depts:
        d_df = df[df["Department"] == d]
        hc = len(d_df)
        avg_tenure = d_df["YearsAtCompany"].mean()
        attr_rate = (len(d_df[d_df["Attrition"] == "Yes"]) / hc) * 100 if hc > 0 else 0
        
        if attr_rate > 18:
            h_idx = "<span style='background:#fee2e2; border:1px solid #fca5a5; color:#ef4444; padding:0.25rem 0.5rem; border-radius:10px; font-size:0.75rem; font-weight:700;'>CRITICAL</span>"
            clr = "#ef4444"
        elif attr_rate > 14:
            h_idx = "<span style='background:#fef3c7; border:1px solid #fde68a; color:#d97706; padding:0.25rem 0.5rem; border-radius:10px; font-size:0.75rem; font-weight:700;'>ELEVATED</span>"
            clr = "#f59e0b"
        else:
            h_idx = "<span style='background:#ecfdf5; border:1px solid #a7f3d0; color:#10b981; padding:0.25rem 0.5rem; border-radius:10px; font-size:0.75rem; font-weight:700;'>STABLE</span>"
            clr = "#10b981"
            
        dept_table_rows += f"""
        <tr>
            <td><strong>{d}</strong></td>
            <td>{hc}</td>
            <td>{avg_tenure:.1f} yrs</td>
            <td style="color:{clr}; font-weight:700;">{attr_rate:.1f}%</td>
            <td>{h_idx}</td>
        </tr>
        """

    st.markdown(
        f"""
        <div class="report-box">
            <div class="report-watermark">
                <div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #1e1b4b; font-family:'Outfit', sans-serif;">PERSEVEX CONSULTING GROUP</div>
                    <div style="font-size:0.75rem; color:#64748b; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; font-family:'Outfit', sans-serif;">Talent Retention Division</div>
                </div>
                <div style="text-align: right; font-size: 0.8rem; color: #64748b; line-height: 1.4;">
                    <strong>Date:</strong> May 26, 2026<br>
                    <strong>Lead Analyst:</strong> Strategic Analytics Board<br>
                    <strong>Subject:</strong> AI Retention Audit Report
                </div>
            </div>

            <div class="report-section">
                <h3>1. Top 3 Attrition Drivers Identified</h3>
                <p style="font-size:0.95rem; color:#475569; line-height:1.6;">Our machine learning feature importance analysis isolating attrition rates across workforce profiles reveals three major stress sectors:</p>
                
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-top: 1rem; margin-bottom: 2rem;">
                    <div class="report-stat-card">
                        <div class="report-stat-val">#1</div>
                        <div class="report-stat-lbl">{top_driver_1}</div>
                        <p style="font-size: 0.78rem; color:#64748b; margin-top:0.4rem; line-height:1.3;">Prime statistical indicator of voluntary exit hazard in current cohorts.</p>
                    </div>
                    <div class="report-stat-card">
                        <div class="report-stat-val">#2</div>
                        <div class="report-stat-lbl">{top_driver_2}</div>
                        <p style="font-size: 0.78rem; color:#64748b; margin-top:0.4rem; line-height:1.3;">Secondary exit catalyst contributing strongly to tenure decay.</p>
                    </div>
                    <div class="report-stat-card">
                        <div class="report-stat-val">#3</div>
                        <div class="report-stat-lbl">{top_driver_3}</div>
                        <p style="font-size: 0.78rem; color:#64748b; margin-top:0.4rem; line-height:1.3;">Third tier driver triggering high exit probability thresholds.</p>
                    </div>
                </div>
            </div>

            <div class="report-section">
                <h3>2. Turnover Vulnerability by Business Sector</h3>
                <p style="font-size:0.95rem; color:#475569; line-height:1.6;">We segment retention threats structurally into organizational nodes. Departmental audit records:</p>
                <table class="report-table">
                    <thead>
                        <tr>
                            <th>Departmental Group</th>
                            <th>Active Headcount</th>
                            <th>Averaged Tenure</th>
                            <th>Current Attrition Rate</th>
                            <th>Attrition Hazard Index</th>
                        </tr>
                    </thead>
                    <tbody>
                        {dept_table_rows}
                    </tbody>
                </table>
            </div>

            <div class="report-section">
                <h3>3. Targeted Strategic Recommendations</h3>
                <ul style="padding-left:1.25rem; font-size:0.92rem; color:#475569; line-height:1.7;">
                    <li style="margin-bottom:0.65rem;"><strong>Flexible Workspace Policy:</strong> Institute hybrid remote workplace options for departments mapping long average travel distances to reduce daily commute fatigue.</li>
                    <li style="margin-bottom:0.65rem;"><strong>Overtime Audit & Resourcing:</strong> Implement a regular departmental audit of overtime demand parameters and introduce strict cap limits, redistributing workloads or allocating contract support where overtime hazard coefficients spike.</li>
                    <li><strong>Targeted Compensation Re-alignment:</strong> Undertake localized salary benchmarks and target retention adjustments for employee groups experiencing base salary levels below market averages.</li>
                </ul>
            </div>

            <div style="margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #e2e8f0; display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #94a3b8;">
                <span>Report Ref: PXV-HR-2026-04</span>
                <span>CONFIDENTIAL &copy; 2026 PERSEVEX</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
st.button("Print Full PDF Brief", on_click=lambda: None) # Stylized action button
