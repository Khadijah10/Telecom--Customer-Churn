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
    :root {
        --primary-color: #FF6B6B;
        --secondary-color: #4ECDC4;
        --success-color: #2ECC71;
        --warning-color: #F39C12;
    }
    
    .main {
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    h1 {
        color: #FF6B6B !important;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #4ECDC4 !important;
        border-bottom: 3px solid #FF6B6B;
        padding-bottom: 0.5rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .stButton button {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E8E 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 107, 107, 0.4);
    }
    
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #FF6B6B;
    }
    
    .divider {
        border-top: 2px solid #FF6B6B;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("<h1>📊 Customer Churn Prediction Application</h1>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; font-size: 1.1em; color: #555; margin-bottom: 2rem;">
Predict customer churn and gain actionable insights into customer behavior with advanced machine learning
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
    st.sidebar.markdown("<h2 style='color: #FF6B6B;'>🧭 Navigation</h2>", unsafe_allow_html=True)
    page = st.sidebar.radio("Select a section:", 
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
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Model Accuracy", f"{accuracy:.1%}", "Performance")
        with col2:
            st.metric("📊 Total Records", f"{len(data):,}", "Dataset size")
        with col3:
            st.metric("⚠️ Churn Rate", f"{churn_rate:.2f}%", "Customers at risk")
        with col4:
            st.metric("💰 Retention Rate", f"{100-churn_rate:.2f}%", "Stable customers")
        
        st.divider()
        
        # Best hyperparameters with nice formatting
        st.markdown("<h3 style='color: #4ECDC4;'>🎯 Best Model Parameters</h3>", unsafe_allow_html=True)
        
        params_col1, params_col2 = st.columns(2)
        with params_col1:
            for key, value in model_info['best_params'].items():
                st.write(f"**{key}:** `{value}`")
        
        with params_col2:
            st.info("💡 These parameters were selected through cross-validation (CV=5) to optimize model accuracy.")
    
    # DATA EXPLORATION
    elif page == "🔍 Data Exploration":
        st.markdown("<h2>🔍 Data Exploration & Analysis</h2>", unsafe_allow_html=True)
        st.write("")
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📋 Dataset Overview", "📊 Visualizations", "📈 Statistical Analysis"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Dataset Preview")
                st.dataframe(data.head(10), use_container_width=True)
            
            with col2:
                st.metric("Total Rows", f"{len(data):,}")
                st.metric("Total Columns", data.shape[1])
                missing_count = data.isnull().sum().sum()
                st.metric("Missing Values", missing_count)
            
            st.divider()
            
            st.subheader("Data Types & Info")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Column Data Types:**")
                st.write(data.dtypes)
            with col2:
                st.write("**Missing Values:**")
                missing = data.isnull().sum()
                if missing.sum() > 0:
                    st.write(missing[missing > 0])
                else:
                    st.success("✅ No missing values!")
        
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
                            marker=dict(color=['#2ECC71', '#FF6B6B']),
                            text=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
                            textposition='outside'
                        )
                    ])
                    fig.update_layout(title="Churn Distribution", height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Age distribution if available
                if 'Age' in data.columns:
                    fig = go.Figure(data=[
                        go.Histogram(
                            x=data['Age'],
                            nbinsx=30,
                            marker=dict(color='#4ECDC4'),
                            name='Age'
                        )
                    ])
                    fig.update_layout(title="Age Distribution", height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix
            st.subheader("Correlation Heatmap")
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 0:
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(numeric_data.corr(), annot=True, cmap='RdYlGn', ax=ax, fmt='.2f', cbar_kws={'label': 'Correlation'})
                ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
                st.pyplot(fig, use_container_width=True)
                plt.close()
        
        with tab3:
            st.subheader("Statistical Summary")
            st.write(data.describe().round(2))
    
    # MODEL PERFORMANCE
    elif page == "📊 Model Performance":
        st.markdown("<h2>📊 Model Performance Metrics</h2>", unsafe_allow_html=True)
        st.write("")
        
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Performance metrics
        st.subheader("🎯 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.2%}")
        
        with col2:
            report = classification_report(y_test, y_pred, output_dict=True)
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
            st.subheader("📋 Detailed Classification Report")
            report_df = pd.DataFrame({
                'Precision': [round(report['0']['precision'], 4), round(report['1']['precision'], 4)],
                'Recall': [round(report['0']['recall'], 4), round(report['1']['recall'], 4)],
                'F1-Score': [round(report['0']['f1-score'], 4), round(report['1']['f1-score'], 4)],
                'Support': [int(report['0']['support']), int(report['1']['support'])]
            }, index=['No Churn', 'Churn'])
            st.dataframe(report_df, use_container_width=True)
        
        with col2:
            st.subheader("🔲 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['No Churn', 'Churn'],
                y=['No Churn', 'Churn'],
                text=cm,
                texttemplate='%{text}',
                colorscale='Blues'
            ))
            fig.update_layout(title="Confusion Matrix", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # MAKE PREDICTION
    elif page == "🔮 Make Prediction":
        st.markdown("<h2>🔮 Customer Churn Prediction</h2>", unsafe_allow_html=True)
        
        st.write("Enter customer details below to predict churn probability:")
        st.info("💡 Fill in all fields and click the prediction button to get insights about customer churn risk.")
        st.write("")
        
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
                    input_data[col] = st.selectbox(f"📋 {col}", unique_vals)
            else:
                with col2:
                    unique_vals = X[col].unique().tolist()
                    input_data[col] = st.selectbox(f"📋 {col}", unique_vals)
        
        st.write("")
        
        # Make prediction
        predict_col1, predict_col2, predict_col3 = st.columns([1, 1, 1])
        
        with predict_col2:
            if st.button("🎯 Predict Churn Risk", use_container_width=True):
                input_df = pd.DataFrame([input_data])
                prediction = best_model.predict(input_df)[0]
                prediction_proba = best_model.predict_proba(input_df)[0]
                
                st.divider()
                
                # Display prediction result with styling
                if prediction == 1:
                    st.error("⚠️ **HIGH CHURN RISK** - This customer is likely to churn", icon="🚨")
                    st.markdown("""
                    <div style='background-color: #FFE5E5; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #FF6B6B;'>
                    <strong>Recommendation:</strong> Consider implementing retention strategies for this customer.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("✅ **LOW CHURN RISK** - This customer is likely to stay", icon="✨")
                    st.markdown("""
                    <div style='background-color: #E5F5E5; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #2ECC71;'>
                    <strong>Recommendation:</strong> Focus on maintaining service quality for this loyal customer.
                    </div>
                    """, unsafe_allow_html=True)
                
                st.write("")
                
                # Probability metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🔴 Churn Probability", f"{prediction_proba[1]:.2%}", delta=None)
                with col2:
                    st.metric("🟢 Retention Probability", f"{prediction_proba[0]:.2%}", delta=None)
                
                st.write("")
                
                # Probability visualization with Plotly
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Retention', 'Churn'],
                        y=[prediction_proba[0], prediction_proba[1]],
                        marker=dict(color=['#2ECC71', '#FF6B6B']),
                        text=[f'{prediction_proba[0]:.1%}', f'{prediction_proba[1]:.1%}'],
                        textposition='outside'
                    )
                ])
                fig.update_layout(
                    title="Prediction Probability Distribution",
                    yaxis_title="Probability",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9em; margin-top: 3rem;'>
    <p>🚀 <strong>Customer Churn Prediction</strong> © 2026 | Built with Streamlit & ML</p>
    <p>Advanced Analytics | Real-time Predictions | Data-Driven Insights</p>
    </div>
    """, unsafe_allow_html=True)
