import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- PAGE SETUP ---
st.set_page_config(page_title="Linear Regression Playground", page_icon="📈", layout="wide")

st.title("📈 Interactive Linear Regression Model")
st.write("Generate a dataset and see how Linear Regression finds the line of best fit.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Generate Data")
n_samples = st.sidebar.slider("Number of data points", min_value=10, max_value=200, value=50, step=10)
noise_level = st.sidebar.slider("Noise (randomness)", min_value=0.0, max_value=50.0, value=15.0, step=1.0)
true_slope = st.sidebar.slider("True Slope (m)", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
true_intercept = st.sidebar.slider("True Intercept (c)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)

# --- GENERATE DATA ---
# Create X values between 0 and 100
X = np.random.uniform(0, 100, n_samples).reshape(-1, 1)
# Create y = mx + c + noise
noise = np.random.normal(0, noise_level, n_samples)
y = (true_slope * X.flatten() + true_intercept) + noise

# --- TRAIN MODEL ---
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

learned_slope = model.coef_[0]
learned_intercept = model.intercept_
r2 = r2_score(y, y_pred)

# --- LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Visualizing the Model")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot actual data
    ax.scatter(X, y, color='#8b5cf6', alpha=0.7, label="Actual Data", edgecolors='white', s=60)
    
    # Plot the prediction line
    ax.plot(X, y_pred, color='#ff4757', linewidth=3, label="Line of Best Fit (Model)")
    
    # Plot the true line (hidden truth)
    true_y = true_slope * X.flatten() + true_intercept
    ax.plot(X, true_y, color='#2ed573', linestyle='--', linewidth=2, alpha=0.5, label="True Relationship")
    
    ax.set_xlabel("X (Input Feature)")
    ax.set_ylabel("y (Target)")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)

with col2:
    st.subheader("Model Metrics")
    
    st.markdown(f"""
    <div style="background-color: #1e1e24; padding: 20px; border-radius: 10px; color: white;">
        <h3 style="color: #8b5cf6; margin-top: 0;">Learned Equation:</h3>
        <p style="font-size: 1.2rem; font-family: monospace;">y = {learned_slope:.2f}x + {learned_intercept:.2f}</p>
        <hr style="border-color: #333;">
        <h3 style="color: #8b5cf6;">Performance:</h3>
        <p><b>R-squared (R²):</b> {r2:.4f}</p>
        <p><b>Mean Squared Error:</b> {mean_squared_error(y, y_pred):.2f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    st.info(f"**What is R²?** It means {r2*100:.1f}% of the variance in your data is explained by the model. Closer to 1.0 (or 100%) is better.")
    
    if st.button("🔄 Re-generate Data"):
        st.rerun()