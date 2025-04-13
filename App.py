import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load model
model = joblib.load("model.pkl")

# Expected feature names (matching training set)
FEATURES = ['State', 'Account length', 'Area code', 'International plan',
            'Voice mail plan', 'Number vmail messages', 'Total day minutes',
            'Total day calls', 'Total day charge', 'Total eve minutes',
            'Total eve calls', 'Total eve charge', 'Total night minutes',
            'Total night calls', 'Total night charge', 'Total intl minutes',
            'Total intl calls', 'Total intl charge', 'Customer service calls']

# Page setup
st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.title("📞 Telecom Churn Prediction App")
st.markdown(
    "Predict whether a customer is likely to **churn** based on service behavior and usage.")

# Sidebar
st.sidebar.title("🔧 Navigation")
mode = st.sidebar.radio(
    "Choose Mode", ["Single Prediction", "Batch Prediction"])

# Data preprocessing function


def preprocess_input(df):
    df['International plan'] = df['International plan'].map(
        {'yes': 1, 'no': 0})
    df['Voice mail plan'] = df['Voice mail plan'].map({'yes': 1, 'no': 0})
    df['State'] = df['State'].astype('category').cat.codes
    df['Area code'] = df['Area code'].astype('category').cat.codes
    return df


# Single Prediction
if mode == "Single Prediction":
    st.subheader("🔍 Enter Customer Details")

    input_data = {}
    for feature in FEATURES:
        if feature in ['International plan', 'Voice mail plan']:
            input_data[feature] = st.selectbox(f"{feature}", ['yes', 'no'])
        elif feature in ['State', 'Area code']:
            input_data[feature] = st.text_input(f"{feature}")
        else:
            input_data[feature] = st.number_input(f"{feature}", step=0.01)

    st.markdown("---")

    if st.button("🚀 Predict Churn"):
        df_input = pd.DataFrame([input_data])
        df_input = preprocess_input(df_input)
        # Ensures feature names match
        df_input = df_input[model.feature_names_in_]

        start = time.time()
        prediction = model.predict(df_input)[0]
        end = time.time()

        elapsed_time = round((end - start) * 1000, 2)

        st.markdown("### 🧠 Prediction Result")
        if prediction:
            st.error("⚠️ This customer is **likely to churn**.")
        else:
            st.success("✅ This customer is **not likely to churn**.")

        st.info(f"⏱️ Prediction Time: {elapsed_time} ms")

# Batch Prediction
elif mode == "Batch Prediction":
    st.subheader("📂 Upload CSV File for Batch Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("### 👀 Data Preview")
        st.dataframe(df.head())

        st.markdown("---")

        try:
            df_processed = preprocess_input(df[FEATURES])
            # Ensures feature names match
            df_processed = df_processed[model.feature_names_in_]

            start = time.time()
            predictions = model.predict(df_processed)
            end = time.time()

            avg_time = round(((end - start) / len(df_processed)) * 1000, 2)

            df['Prediction'] = [
                'Churn' if pred else 'Not Churn' for pred in predictions]
            st.success("✅ Batch Prediction Complete!")

            # Count plot
            st.markdown("### 📊 Churn Count Plot")
            fig1, ax1 = plt.subplots()
            sns.countplot(x='Prediction', data=df, palette='coolwarm', ax=ax1)
            ax1.set_title("Churn vs Not Churn")
            st.pyplot(fig1)

            # Pie chart
            st.markdown("### 🥧 Churn Pie Chart")
            churn_counts = df['Prediction'].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', colors=[
                    '#FF6B6B', '#4ECDC4'], startangle=90)
            ax2.axis('equal')
            st.pyplot(fig2)

            # Table
            st.markdown("### 🧾 Prediction Table")
            st.dataframe(df)

            # Prediction time
            st.info(f"⏱️ Average Prediction Time: {avg_time} ms per record")

            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predictions", data=csv,
                               file_name="churn_predictions.csv", mime='text/csv')

        except Exception as e:
            st.error(f"❌ Error: {e}")
