# 🎓 Silent Dropout Predictor (AI Dashboard)

An advanced, machine-learning-powered web application designed to help university faculty identify and support at-risk students before they drop out. 

## 🚀 Features
* **Predictive Analytics:** Uses a trained Random Forest Classifier to categorize students into High, Medium, or Low risk based on attendance and marks.
* **Explainable AI (XAI):** Integrates SHAP visualizations so teachers can see exactly *why* the AI flagged a student.
* **Automated Intervention:** * Integrates **Google Gemini AI** to draft hyper-personalized, empathetic emails to students.
  * Integrates the **Twilio API** to send instant SMS alerts to parents of critical-risk students.
* **Professional UI:** Features a custom Google Workspace/Material Design aesthetic built over Streamlit.

## 🛠️ Tech Stack
* **Frontend:** Streamlit, Custom CSS
* **Machine Learning:** Scikit-Learn (Random Forest), SHAP
* **Data Visualization:** Plotly, Matplotlib, Pandas
* **APIs:** Google Gemini (GenAI), Twilio (SMS)

## 📌 Note on Privacy
*This repository contains dummy data for demonstration purposes. No real student data is exposed. API keys are handled securely via environment variables/sidebar injection and are not stored in the source code.*
