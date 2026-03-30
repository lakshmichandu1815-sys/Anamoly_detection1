import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("📄 Invoice Anomaly Detection (Z-Score Method)")

# Upload file
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data")
    st.write(df)

    # Let user choose columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 1:
        st.error("No numeric columns found!")
    else:
        selected_cols = st.multiselect("Select columns for anomaly detection", numeric_cols, default=numeric_cols[:2])

        if selected_cols:
            # Calculate Z-score
            z_scores = np.abs((df[selected_cols] - df[selected_cols].mean()) / df[selected_cols].std())

            threshold = st.slider("Select Threshold", 1.0, 3.0, 2.0)

            df['anomaly'] = (z_scores > threshold).any(axis=1)

            # Convert labels
            df['anomaly'] = df['anomaly'].map({True: "Anomaly", False: "Normal"})

            st.subheader("🔍 Results")
            st.write(df)

            # Download option
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "zscore_results.csv", "text/csv")

            # Visualization
            if len(selected_cols) >= 2:
                st.subheader("📈 Visualization")
                fig, ax = plt.subplots()
                colors = df['anomaly'].apply(lambda x: 1 if x == "Normal" else 0)
                ax.scatter(df[selected_cols[0]], df[selected_cols[1]], c=colors)
                ax.set_xlabel(selected_cols[0])
                ax.set_ylabel(selected_cols[1])
                ax.set_title("Z-Score Anomaly Detection")
                st.pyplot(fig)