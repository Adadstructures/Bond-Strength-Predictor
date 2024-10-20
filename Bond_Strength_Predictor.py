import streamlit as st
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Load the saved model and scalers
with open('bond_strength_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('bs_scaler_X.pkl', 'rb') as scaler_X_file:
    scaler_X = pickle.load(scaler_X_file)

with open('bs_scaler_y.pkl', 'rb') as scaler_y_file:
    scaler_y = pickle.load(scaler_y_file)

# Streamlit input form for user to input values
st.title('Ultimate Bond Strength Predictor')

concrete_width = st.number_input('Concrete Width (mm)', min_value=00.00)
compressive_strength = st.number_input('Compressive Strength (MPa)', min_value=00.00)
frp_modulus = st.number_input('FRP Modulus (GPa)', min_value=00.00)
frp_overall_thickness = st.number_input('FRP Overall Thickness (mm)', min_value=0.00)
frp_sheet_width = st.number_input('FRP Sheet Width (mm)', min_value=00.00)
bond_length = st.number_input('Bond Length (mm)', min_value=00.00)

# Make prediction on user input
if st.button('Predict'):
    # Scale the inputs
    input_data = np.array([[concrete_width, compressive_strength, frp_modulus, frp_overall_thickness, frp_sheet_width, bond_length]])
    input_scaled = scaler_X.transform(input_data)
    
    # Make prediction
    pred_scaled = model.predict(input_scaled)
    
    # Denormalize the prediction
    prediction = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    # Display the prediction with larger font size
    st.markdown(f"<h2 style='text-align: left; color: green; font-size: 32px;'>Predicted Ultimate Bond Strength: {prediction[0]:.3f} kN</h2>", unsafe_allow_html=True)

    
    # SHAP values for feature importance
    st.subheader("SHAP Feature Importance")

    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # SHAP values for the input
    shap_values = explainer.shap_values(input_scaled)

    # Plot the SHAP summary plot for feature importance
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, input_scaled, feature_names=['Concrete Width', 'Compressive Strength', 'FRP Modulus', 'FRP Overall Thickness', 'FRP Sheet Width', 'Bond Length'], plot_type="bar", show=False)
    
    # Display the SHAP plot in Streamlit
    st.pyplot(fig)
    
    # Add footnote
    st.markdown("""
        **Notes**: 
        1. This application predicts the ultimate bond strength of FRP-concrete interface using categorical boosting algorithm optimised with Optuna.
        2. The model was trained using data from single-lap shear test experiments.
    """)
    
    st.markdown("""
        **References**: 
        1. L. Prokhorenkova, G. Gusev, A. Vorobev, A.V. Dorogush, A. Gulin, CatBoost: unbiased boosting with categorical features, 2018. https://github.com/catboost/catboost.
        2. T. Akiba, S. Sano, T. Yanase, T. Ohta, M. Koyama, Optuna: A Next-generation Hyperparameter Optimization Framework, in: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, Association for Computing Machinery, New York, NY, USA, 2019: pp. 2623–2631. https://doi.org/10.1145/3292500.3330701.
    """)

# Adding a footer with contact information
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f1f1f1;
    text-align: center;
    padding: 10px;
    font-size: 12px;
    color: #6c757d;
}
</style>
<div class="footer">
    <p>© 2024 My Streamlit App. All rights reserved. |Temitope E. Dada, Silas E. Oluwadahunsi, Guobin Gong, Jun Xia, Luigi Di Sarno | For Queries: <a href="mailto: T.Dada19@student.xjtlu.edu.cn"> T.Dada19@student.xjtlu.edu.cn</a></p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)
