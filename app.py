import streamlit as st
import pandas as pd
import pickle
import urllib.parse
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from twilio.rest import Client
import google.generativeai as genai # NEW: Google Gemini AI

# 1. Page Configuration
st.set_page_config(page_title="Silent Dropout Predictor", page_icon="🎓", layout="wide")

# --- LOGIN SYSTEM ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🔒 Faculty Login Portal</h1>", unsafe_allow_html=True)
    st.write("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("Demo Credentials -> Username: admin | Password: password123")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                if username == "admin" and password == "password123":
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
    st.stop()

# --- MAIN APP ---

# Sidebar: Config & File Upload
with st.sidebar:
    st.title("👨‍🏫 Welcome, Admin")
    if st.button("Logout", type="primary"):
        st.session_state['logged_in'] = False
        st.rerun()
    st.divider()
    
    # API CONFIGURATIONS
    st.title("🔑 API Integrations")
    with st.expander("🤖 Gemini AI Config"):
        gemini_key = st.text_input("Google Gemini API Key", type="password")
        
    with st.expander("📱 Twilio SMS Config"):
        tw_sid = st.text_input("Account SID", type="password")
        tw_token = st.text_input("Auth Token", type="password")
        tw_phone = st.text_input("Twilio Phone Number")
    
    st.divider()
    st.title("⚙️ Control Panel")
    uploaded_file = st.file_uploader("Upload Student Data", type=["csv", "xlsx"])

# Load Model
try:
    with open('dropout_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'dropout_model.pkl' not found.")
    st.stop()

st.title("🎓 Silent Dropout Prediction Dashboard")

if uploaded_file is not None:
    # Read Data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
        
    # Predictions
    input_features = df[['Attendance (%)', 'Marks (/100)', 'Assignments (/10)']]
    predictions = model.predict(input_features)
    risk_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    df['Predicted Risk'] = [risk_mapping[pred] for pred in predictions]
    color_map = {"High Risk": "#FF4B4B", "Medium Risk": "#FFA500", "Low Risk": "#00CC96"}
    
    # --- TABBED UI ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Main Dashboard", "📈 Deep Analytics", "🧠 Explainable AI", "✉️ Action Panel"])
    
    # [TABS 1, 2, and 3 REMAIN EXACTLY THE SAME AS BEFORE]
    with tab1:
        st.subheader("Class Overview Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Students", len(df))
        col2.metric("🔴 High Risk", len(df[df['Predicted Risk'] == 'High Risk']))
        col3.metric("🟡 Medium Risk", len(df[df['Predicted Risk'] == 'Medium Risk']))
        col4.metric("🟢 Low Risk", len(df[df['Predicted Risk'] == 'Low Risk']))
        st.divider()
        st.subheader("📋 Student Roster & Filters")
        filter_choice = st.radio("Select a category to view:", options=["All", "High Risk", "Medium Risk", "Low Risk"], horizontal=True)
        filtered_df = df if filter_choice == "All" else df[df['Predicted Risk'] == filter_choice]
        st.dataframe(filtered_df[['Name', 'Roll No', 'Attendance (%)', 'Marks (/100)', 'Predicted Risk']], use_container_width=True)
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download This Report", data=csv_data, file_name=f'{filter_choice}_Report.csv', mime='text/csv')

    with tab2:
        st.markdown("### 🔍 Advanced AI Insights & Distributions")
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            risk_counts = df['Predicted Risk'].value_counts().reset_index()
            risk_counts.columns = ['Risk Level', 'Count']
            fig_donut = px.pie(risk_counts, values='Count', names='Risk Level', title='Overall Risk Distribution', color='Risk Level', color_discrete_map=color_map, hole=0.45)
            st.plotly_chart(fig_donut, use_container_width=True)
        with row1_col2:
            avg_stats = df.groupby('Predicted Risk')[['Attendance (%)', 'Marks (/100)']].mean().reset_index()
            fig_bar = px.bar(avg_stats, x='Predicted Risk', y=['Attendance (%)', 'Marks (/100)'], barmode='group', title='Average Performance by Risk Category', color_discrete_sequence=['#1f77b4', '#ff7f0e'])
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.markdown("### 🧠 Explainable AI (XAI) Engine")
        student_list = df['Name'].tolist()
        selected_student = st.selectbox("Select a Student to Explain:", student_list)
        if selected_student:
            student_data = df[df['Name'] == selected_student]
            student_features = student_data[['Attendance (%)', 'Marks (/100)', 'Assignments (/10)']]
            risk = student_data['Predicted Risk'].values[0]
            st.markdown(f"#### Analyzing: **{selected_student}** (Status: {risk})")
            try:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_features)
                idx = student_data.index[0]
                fig = plt.figure(figsize=(8, 3))
                target_class_idx = 1 if len(shap_values) == 2 else 2 
                if isinstance(shap_values, list):
                    shap.plots.waterfall(shap.Explanation(values=shap_values[target_class_idx][idx], base_values=explainer.expected_value[target_class_idx], data=student_features.iloc[0], feature_names=input_features.columns))
                else:
                    shap.summary_plot(shap_values[idx:idx+1], student_features, plot_type="bar", show=False)
                st.pyplot(fig)
            except Exception as e:
                comp_df = pd.DataFrame({'Metric': ['Attendance (%)', 'Marks (/100)', 'Assignments (/10)'], 'Student Score': [student_data['Attendance (%)'].values[0], student_data['Marks (/100)'].values[0], student_data['Assignments (/10)'].values[0]], 'Class Average': [df['Attendance (%)'].mean(), df['Marks (/100)'].mean(), df['Assignments (/10)'].mean()]})
                fig_comp = px.bar(comp_df, x='Metric', y=['Student Score', 'Class Average'], barmode='group')
                st.plotly_chart(fig_comp, use_container_width=True)

    # ==========================================
    # TAB 4: EMAIL, TWILIO, & GEMINI AI ACTION PANEL
    # ==========================================
    with tab4:
        st.subheader("✉️ Mentor Action Panel (AI Emails & SMS)")
        search_term = st.text_input("🔍 Search for a student by name (Clear to view all):")
        display_students = df[df['Name'].str.contains(search_term, case=False, na=False)] if search_term else df
            
        if display_students.empty:
            st.warning("No students found.")
        else:
            st.write("---")
            for index, row in display_students.iterrows():
                student_name = row['Name']
                student_email = row['Email']
                parent_phone = str(row['Parent Phone'])
                risk_level = row['Predicted Risk']
                
                with st.container():
                    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
                    c1.markdown(f"**{student_name}**")
                    
                    if risk_level == 'High Risk': c2.error("🔴 " + risk_level)
                    elif risk_level == 'Medium Risk': c2.warning("🟡 " + risk_level)
                    else: c2.success("🟢 " + risk_level)
                    
                    # GEMINI AI EMAIL GENERATOR
                    if c3.button("✨ Draft AI Email", key=f"ai_{row['Roll No']}", use_container_width=True):
                        if gemini_key:
                            try:
                                genai.configure(api_key=gemini_key)
                                ai_model = genai.GenerativeModel('gemini-2.5-flash')
                                prompt = f"Write a brief, polite, and empathetic email to a university student named {student_name}. Their current attendance is {row['Attendance (%)']}% and marks are {row['Marks (/100)']}/100. Their academic status is flagged as {risk_level}. Ask them to schedule a meeting with their mentor to discuss support options. Keep it strictly under 4 sentences."
                                response = ai_model.generate_content(prompt)
                                
                                subject_encoded = urllib.parse.quote(f"Checking in regarding your progress, {student_name}")
                                body_encoded = urllib.parse.quote(response.text)
                                mail_link = f"https://mail.google.com/mail/?view=cm&fs=1&to={student_email}&su={subject_encoded}&body={body_encoded}"
                                
                                st.info(f"**AI Drafted Message:**\n\n{response.text}")
                                st.link_button("📧 Click to Send via Gmail", mail_link)
                            except Exception as e:
                                # The Presentation Safety Net!
                                if "429" in str(e) or "quota" in str(e).lower():
                                    st.warning("⚠️ AI servers are currently busy. Generating smart fallback template...")
                                    
                                    # Fallback email that still looks AI-generated
                                    fallback_msg = f"Dear {student_name},\n\nI am reaching out because I noticed your attendance is currently at {row['Attendance (%)']}% and your marks are {row['Marks (/100)']}/100. This metrics flag your academic status as {risk_level}.\n\nPlease schedule a meeting with me this week to discuss how we can support you.\n\nBest,\nYour Mentor"
                                    
                                    subject_encoded = urllib.parse.quote(f"Checking in regarding your progress, {student_name}")
                                    body_encoded = urllib.parse.quote(fallback_msg)
                                    mail_link = f"https://mail.google.com/mail/?view=cm&fs=1&to={student_email}&su={subject_encoded}&body={body_encoded}"
                                    
                                    st.info(f"**Drafted Message:**\n\n{fallback_msg}")
                                    st.link_button("📧 Click to Send via Gmail", mail_link)
                                else:
                                    st.error(f"AI Error: {e}")
                        else:
                            st.warning("⚠️ Enter Gemini API Key in the Sidebar first!")
                            
                    # TWILIO SMS BUTTON
                    if c4.button("📱 SMS Parent", key=f"sms_{row['Roll No']}", use_container_width=True):
                        if tw_sid and tw_token and tw_phone:
                            try:
                                client = Client(tw_sid, tw_token)
                                formatted_phone = f"+91{parent_phone}" 
                                msg_body = f"URGENT ALERT: {student_name}'s academic risk level is {risk_level}. Please check their portal."
                                message = client.messages.create(body=msg_body, from_=tw_phone, to=formatted_phone)
                                st.success(f"SMS Sent to {formatted_phone}!")
                            except Exception as e:
                                st.error(f"Twilio Error: {e}")
                        else:
                            st.warning("⚠️ Enter Twilio Credentials in the Sidebar!")
                st.write("---") 

else:
    st.info("👈 Please upload a student data file from the sidebar to get started.")