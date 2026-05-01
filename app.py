import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for fancy styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    :root {
        --primary-color: #6366F1;
        --secondary-color: #EC4899;
        --accent-color: #14B8A6;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --danger-color: #EF4444;
        --dark-bg: #0F172A;
        --light-text: #F1F5F9;
    }
    
    .main {
        padding: 2.5rem 1.5rem;
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        color: #F1F5F9;
    }
    
    h1 {
        background: linear-gradient(135deg, #6366F1 0%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-size: 2.8em;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-family: 'Poppins', sans-serif;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: #F1F5F9 !important;
        font-size: 1.5em;
        font-weight: 600;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #6366F1, #EC4899) 1;
        padding-bottom: 0.8rem;
        padding-left: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    h3 {
        color: #E0E7FF !important;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(236, 72, 153, 0.05) 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid rgba(99, 102, 241, 0.1);
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        color: #94A3B8 !important;
        border: none !important;
        border-bottom: 3px solid transparent !important;
        padding-bottom: 1rem !important;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #6366F1 !important;
        border-bottom-color: #6366F1 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: #E0E7FF !important;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #6366F1 0%, #EC4899 100%);
        color: white;
        border: none;
        border-radius: 0.75rem;
        font-weight: 600;
        padding: 0.85rem 2.2rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        letter-spacing: 0.5px;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
    }
    
    .stButton button:active {
        transform: translateY(0);
    }
    
    .stMetric {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(236, 72, 153, 0.08) 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.15);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.2);
    }
    
    .stMetric label {
        color: #CBD5E1 !important;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #6366F1 !important;
        font-size: 1.8em;
        font-weight: 700;
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        color: #10B981 !important;
        font-weight: 600;
    }
    
    .divider {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.3), rgba(236, 72, 153, 0.3), rgba(99, 102, 241, 0.3));
        margin: 2rem 0;
    }
    
    /* Sidebar Styling */
    .sidebar .stRadio > label {
        font-size: 1.05rem;
        font-weight: 600;
        color: #E0E7FF;
    }
    
    .sidebar [data-baseweb="radio"] {
        gap: 0.75rem;
    }
    
    .sidebar [data-baseweb="radio"] label {
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .sidebar [data-baseweb="radio"] label:hover {
        background: rgba(99, 102, 241, 0.1);
    }
    
    /* Input styling */
    .stNumberInput input, .stSelectbox select, .stTextInput input {
        background-color: rgba(30, 41, 59, 0.8) !important;
        color: #F1F5F9 !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput input:focus, .stSelectbox select:focus, .stTextInput input:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }
    
    /* Alert boxes */
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05)) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 0.75rem !important;
        padding: 1rem !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05)) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 0.75rem !important;
        padding: 1rem !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(99, 102, 241, 0.05)) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 0.75rem !important;
        padding: 1rem !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05)) !important;
        border: 1px solid rgba(245, 158, 11, 0.3) !important;
        border-radius: 0.75rem !important;
        padding: 1rem !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid rgba(99, 102, 241, 0.1) !important;
        border-radius: 0.75rem !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(99, 102, 241, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.5);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #64748B;
        font-size: 0.9em;
        margin-top: 3rem;
        padding: 2rem 1rem;
        border-top: 1px solid rgba(99, 102, 241, 0.1);
    }
    
    .footer strong {
        color: #CBD5E1;
    }
    
    /* Animation for page transition */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .main {
        animation: fadeIn 0.5s ease;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("<h1>📊 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; font-size: 1.05em; color: #CBD5E1; margin-bottom: 2.5rem; font-weight: 500; letter-spacing: 0.5px;">
🎯 Predict customer churn with AI-powered analytics | 📈 Real-time insights | 🚀 Data-driven decisions
</div>
""", unsafe_allow_html=True)
st.divider()

# Load data with caching
@st.cache_data
def load_data():
    try:
        data = pd.read_csv(r"C:\Users\user\OneDrive\Documents\ML projects\customer.csv")
        return data
    except FileNotFoundError:
        st.error("Data file not found. Please ensure the customer.csv file exists.")
        return None

# Train model with caching
@st.cache_resource
def train_model(data):
    # Data preprocessing
    data_clean = data.copy()
    data_clean.fillna(method='ffill', inplace=True)
    
    # Drop irrelevant columns if they exist
    if 'index' in data_clean.columns:
        data_clean.drop(['index'], axis=1, inplace=True)
    
    # Split features and target
    X = data_clean.drop('Churn', axis=1)
    y = data_clean['Churn']
    
    # Identify column types
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and train model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    
    model.fit(X_train, y_train)
    
    # Hyperparameter tuning
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__solver': ['liblinear', 'lbfgs']
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    return {
        'model': best_model,
        'preprocessor': preprocessor,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X': X,
        'y': y,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'best_params': grid_search.best_params_
    }

# Main app logic
data = load_data()

if data is not None:
    st.success(f"✅ Data loaded successfully! ({len(data):,} records)")
    
    # Sidebar navigation with improved styling
    st.sidebar.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 1.3em; color: #E0E7FF; margin: 0; font-weight: 700;">
            🧭 Navigation
        </h1>
        <div style="border-bottom: 2px solid rgba(99, 102, 241, 0.2); margin-top: 0.75rem;"></div>
    </div>
    """, unsafe_allow_html=True)
    page = st.sidebar.radio("", 
                           ["📈 Dashboard", "🔍 Data Exploration", "📊 Model Performance", "🔮 Make Prediction"],
                           label_visibility="collapsed")
    
    # Load and train model
    model_info = train_model(data)
    best_model = model_info['model']
    X_test = model_info['X_test']
    y_test = model_info['y_test']
    
    # DASHBOARD
    if page == "📈 Dashboard":
        st.markdown("<h2>📈 Dashboard Overview</h2>", unsafe_allow_html=True)
        
        # Model predictions
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        churn_rate = (data['Churn'].sum() / len(data) * 100)
        
        st.write("")  # Spacing
        
        col1, col2, col3, col4 = st.columns(4, gap="medium")
        
        with col1:
            st.metric("🎯 Model Accuracy", f"{accuracy:.1%}")
        with col2:
            st.metric("📊 Total Records", f"{len(data):,}")
        with col3:
            st.metric("⚠️  Churn Rate", f"{churn_rate:.2f}%")
        with col4:
            st.metric("💚 Retention Rate", f"{100-churn_rate:.2f}%")
        
        st.divider()
        
        # Best hyperparameters with nice formatting
        st.markdown("<h3>🎯 Best Model Parameters</h3>", unsafe_allow_html=True)
        
        params_col1, params_col2 = st.columns([1, 1], gap="large")
        
        with params_col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(236, 72, 153, 0.05)); 
                        padding: 1.5rem; border-radius: 1rem; border: 1px solid rgba(99, 102, 241, 0.15);">
            """, unsafe_allow_html=True)
            for key, value in model_info['best_params'].items():
                st.markdown(f"**{key}:** `{value}`")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with params_col2:
            st.info("💡 These parameters were optimized through 5-fold cross-validation to maximize model accuracy.", icon="🔬")
        
        st.divider()
        
        # Quick stats
        st.markdown("<h3>📊 Quick Statistics</h3>", unsafe_allow_html=True)
        stat_col1, stat_col2, stat_col3 = st.columns(3, gap="medium")
        
        with stat_col1:
            total_churn = data['Churn'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94A3B8; font-size: 0.9em; margin-bottom: 0.5rem;">Customers Churned</div>
                <div style="color: #6366F1; font-size: 2em; font-weight: 700;">{int(total_churn):,}</div>
                <div style="color: #64748B; font-size: 0.85em; margin-top: 0.5rem;">Total at-risk</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_col2:
            total_retained = len(data) - data['Churn'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94A3B8; font-size: 0.9em; margin-bottom: 0.5rem;">Customers Retained</div>
                <div style="color: #10B981; font-size: 2em; font-weight: 700;">{int(total_retained):,}</div>
                <div style="color: #64748B; font-size: 0.85em; margin-top: 0.5rem;">Active & stable</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94A3B8; font-size: 0.9em; margin-bottom: 0.5rem;">Test Set Size</div>
                <div style="color: #EC4899; font-size: 2em; font-weight: 700;">{len(X_test):,}</div>
                <div style="color: #64748B; font-size: 0.85em; margin-top: 0.5rem;">Evaluated records</div>
            </div>
            """, unsafe_allow_html=True)
    
    # DATA EXPLORATION
    elif page == "🔍 Data Exploration":
        st.markdown("<h2>🔍 Data Exploration & Analysis</h2>", unsafe_allow_html=True)
        st.write("")
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📋 Dataset Overview", "📊 Visualizations", "📈 Statistical Analysis"])
        
        with tab1:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("📋 Dataset Preview")
                st.dataframe(data.head(10), use_container_width=True, hide_index=False)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <div style="color: #CBD5E1; font-weight: 600; margin-bottom: 1rem;">Dataset Info</div>
                """, unsafe_allow_html=True)
                st.metric("Rows", f"{len(data):,}")
                st.metric("Columns", data.shape[1])
                missing_count = data.isnull().sum().sum()
                st.metric("Missing Values", missing_count)
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.divider()
            
            st.subheader("📊 Data Types & Quality")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='metric-card'><strong style='color: #E0E7FF;'>Column Data Types</strong></div>", unsafe_allow_html=True)
                dtype_df = pd.DataFrame({
                    'Column': data.dtypes.index,
                    'Type': data.dtypes.values.astype(str)
                })
                st.dataframe(dtype_df, use_container_width=True, hide_index=True)
            with col2:
                st.markdown("<div class='metric-card'><strong style='color: #E0E7FF;'>Missing Values</strong></div>", unsafe_allow_html=True)
                missing = data.isnull().sum()
                if missing.sum() > 0:
                    st.dataframe(missing[missing > 0], use_container_width=True)
                else:
                    st.success("✅ No missing values detected!")
        
        with tab2:
            st.subheader("📊 Key Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Churn distribution with Plotly
                if 'Churn' in data.columns:
                    churn_counts = data['Churn'].value_counts()
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['No Churn', 'Churn'],
                            y=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
                            marker=dict(color=['#10B981', '#EF4444'], 
                                       line=dict(color='rgba(0,0,0,0.1)', width=2)),
                            text=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
                            textposition='outside',
                            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                        )
                    ])
                    fig.update_layout(
                        title="Churn Distribution",
                        height=400,
                        showlegend=False,
                        template='plotly_dark',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Age distribution if available
                if 'Age' in data.columns:
                    fig = go.Figure(data=[
                        go.Histogram(
                            x=data['Age'],
                            nbinsx=30,
                            marker=dict(color='#6366F1', 
                                       line=dict(color='rgba(0,0,0,0.1)', width=1)),
                            name='Age',
                            hovertemplate='Age: %{x}<br>Count: %{y}<extra></extra>'
                        )
                    ])
                    fig.update_layout(
                        title="Age Distribution",
                        height=400,
                        showlegend=False,
                        template='plotly_dark'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix
            st.subheader("Correlation Heatmap")
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 0:
                fig, ax = plt.subplots(figsize=(12, 8))
                fig.patch.set_facecolor('#0F172A')
                sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax, fmt='.2f', 
                           cbar_kws={'label': 'Correlation'}, linewidths=0.5)
                ax.set_facecolor('#1E293B')
                ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', color='#E0E7FF', pad=20)
                ax.tick_params(colors='#CBD5E1')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', color='#CBD5E1')
                plt.setp(ax.get_yticklabels(), rotation=0, color='#CBD5E1')
                st.pyplot(fig, use_container_width=True)
                plt.close()
        
        with tab3:
            st.subheader("Statistical Summary")
            stats_df = data.describe().round(3).T
            st.dataframe(stats_df, use_container_width=True)
    
    # MODEL PERFORMANCE
    elif page == "📊 Model Performance":
        st.markdown("<h2>📊 Model Performance Metrics</h2>", unsafe_allow_html=True)
        st.write("")
        
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Performance metrics
        st.subheader("🎯 Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4, gap="medium")
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.2%}")
        
        with col2:
            precision = report['weighted avg']['precision']
            st.metric("Precision", f"{precision:.2%}")
        
        with col3:
            recall = report['weighted avg']['recall']
            st.metric("Recall", f"{recall:.2%}")
        
        with col4:
            f1 = report['weighted avg']['f1-score']
            st.metric("F1-Score", f"{f1:.2%}")
        
        st.divider()
        
        # Detailed report and confusion matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Classification Report")
            report_df = pd.DataFrame({
                'Precision': [round(report['0']['precision'], 4), round(report['1']['precision'], 4)],
                'Recall': [round(report['0']['recall'], 4), round(report['1']['recall'], 4)],
                'F1-Score': [round(report['0']['f1-score'], 4), round(report['1']['f1-score'], 4)],
                'Support': [int(report['0']['support']), int(report['1']['support'])]
            }, index=['No Churn', 'Churn'])
            
            st.markdown("""
            <div class="metric-card">
            """, unsafe_allow_html=True)
            st.dataframe(report_df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.subheader("🔲 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted: No Churn', 'Predicted: Churn'],
                y=['Actual: No Churn', 'Actual: Churn'],
                text=cm,
                texttemplate='%{text}',
                colorscale='Purples',
                hovertemplate='%{y}<br>%{x}<br>Count: %{z}<extra></extra>'
            ))
            fig.update_layout(
                title="Confusion Matrix",
                height=400,
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Model interpretation
        st.subheader("📊 Model Insights")
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #E0E7FF; font-weight: 600; margin-bottom: 1rem;">✅ True Positives</div>
                <div style="color: #10B981; font-size: 1.8em; font-weight: 700;">{cm[1][1]}</div>
                <div style="color: #64748B; font-size: 0.85em; margin-top: 0.5rem;">Correctly identified churners</div>
            </div>
            """, unsafe_allow_html=True)
        
        with insight_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #E0E7FF; font-weight: 600; margin-bottom: 1rem;">❌ False Positives</div>
                <div style="color: #EF4444; font-size: 1.8em; font-weight: 700;">{cm[0][1]}</div>
                <div style="color: #64748B; font-size: 0.85em; margin-top: 0.5rem;">Loyal customers flagged as risk</div>
            </div>
            """, unsafe_allow_html=True)
    
    # MAKE PREDICTION
    elif page == "🔮 Make Prediction":
        st.markdown("<h2>🔮 Customer Churn Prediction</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(236, 72, 153, 0.05)); 
                    padding: 1.5rem; border-radius: 1rem; border: 1px solid rgba(99, 102, 241, 0.2); 
                    margin-bottom: 1.5rem;">
            <strong style="color: #E0E7FF;">📝 Instructions:</strong><br>
            <span style="color: #CBD5E1;">Fill in all customer details below to get an AI prediction of churn probability. 
            The model will analyze the customer profile and provide actionable insights.</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Get sample data for feature info
        X = model_info['X']
        
        # Create input fields for all features
        col1, col2 = st.columns(2)
        input_data = {}
        
        numerical_cols = model_info['numerical_cols']
        categorical_cols = model_info['categorical_cols']
        
        # Collect numerical inputs
        for i, col in enumerate(numerical_cols):
            if i % 2 == 0:
                with col1:
                    min_val = X[col].min()
                    max_val = X[col].max()
                    input_data[col] = st.number_input(
                        f"📊 {col}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(X[col].mean()),
                        help=f"Range: {min_val:.2f} - {max_val:.2f}"
                    )
            else:
                with col2:
                    min_val = X[col].min()
                    max_val = X[col].max()
                    input_data[col] = st.number_input(
                        f"📊 {col}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(X[col].mean()),
                        help=f"Range: {min_val:.2f} - {max_val:.2f}"
                    )
        
        # Collect categorical inputs
        for i, col in enumerate(categorical_cols):
            if i % 2 == 0:
                with col1:
                    unique_vals = X[col].unique().tolist()
                    input_data[col] = st.selectbox(f"📋 {col}", unique_vals, key=f"select_{i}")
            else:
                with col2:
                    unique_vals = X[col].unique().tolist()
                    input_data[col] = st.selectbox(f"📋 {col}", unique_vals, key=f"select_{i}")
        
        st.write("")
        
        # Make prediction
        predict_col1, predict_col2, predict_col3 = st.columns([1, 1, 1])
        
        with predict_col2:
            if st.button("🚀 Generate Prediction", use_container_width=True):
                input_df = pd.DataFrame([input_data])
                prediction = best_model.predict(input_df)[0]
                prediction_proba = best_model.predict_proba(input_df)[0]
                
                st.divider()
                
                # Display prediction result with styling
                if prediction == 1:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05)); 
                                padding: 2rem; border-radius: 1rem; border: 2px solid rgba(239, 68, 68, 0.3); 
                                text-align: center;">
                        <div style="color: #EF4444; font-size: 2em; margin-bottom: 0.5rem;">⚠️ HIGH CHURN RISK</div>
                        <div style="color: #FECACA; font-size: 1.1em;">This customer is likely to churn</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05)); 
                                padding: 1.5rem; border-radius: 1rem; border-left: 4px solid #EF4444; margin-top: 1rem;'>
                    <strong style="color: #FECACA;">🎯 Recommended Actions:</strong><br>
                    <span style="color: #CBD5E1;">
                    • Initiate personal outreach and loyalty programs<br>
                    • Review service quality and satisfaction levels<br>
                    • Consider special retention offers or discounts<br>
                    • Schedule customer success check-in
                    </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05)); 
                                padding: 2rem; border-radius: 1rem; border: 2px solid rgba(16, 185, 129, 0.3); 
                                text-align: center;">
                        <div style="color: #10B981; font-size: 2em; margin-bottom: 0.5rem;">✅ LOW CHURN RISK</div>
                        <div style="color: #A7F3D0; font-size: 1.1em;">This customer is likely to stay</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05)); 
                                padding: 1.5rem; border-radius: 1rem; border-left: 4px solid #10B981; margin-top: 1rem;'>
                    <strong style="color: #A7F3D0;">🎯 Recommended Actions:</strong><br>
                    <span style="color: #CBD5E1;">
                    • Focus on maintaining service quality<br>
                    • Encourage cross-selling and upselling opportunities<br>
                    • Request for testimonials and referrals<br>
                    • Continue regular engagement and support
                    </span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.write("")
                
                # Probability metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🔴 Churn Probability", f"{prediction_proba[1]:.2%}")
                with col2:
                    st.metric("🟢 Retention Probability", f"{prediction_proba[0]:.2%}")
                
                st.write("")
                
                # Probability visualization with Plotly
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Retention', 'Churn'],
                        y=[prediction_proba[0], prediction_proba[1]],
                        marker=dict(
                            color=['#10B981', '#EF4444'],
                            line=dict(color='rgba(0,0,0,0.1)', width=2)
                        ),
                        text=[f'{prediction_proba[0]:.1%}', f'{prediction_proba[1]:.1%}'],
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Probability: %{y:.2%}<extra></extra>'
                    )
                ])
                fig.update_layout(
                    title="Prediction Probability Distribution",
                    yaxis_title="Probability",
                    height=400,
                    showlegend=False,
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.markdown("""
    <div class="footer">
        <p>🚀 <strong>Customer Churn Prediction Platform</strong></p>
        <p>Advanced ML Analytics | Real-time Predictions | Data-Driven Insights</p>
        <p style="margin-top: 1rem; font-size: 0.8em; color: #475569;">Built with Streamlit • scikit-learn • Plotly</p>
    </div>
    """, unsafe_allow_html=True)
