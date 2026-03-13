import streamlit as st
import requests

st.set_page_config(page_title="TaleOfTheTape")
st.title("TaleOfTheTape.ai")


try:
    response = requests.get("http://api:8000/")
    if response.status_code == 200:
        st.success(f"Backend Status: {response.json()['status']}")
except:
    st.error("Backend nicht erreichbar!")

st.markdown("---")

# UI für den Manual Matchmaker
st.header("Manual Matchmaker")
col1, col2 = st.columns(2)

with col1:
    fighter_1 = st.text_input("Kämpfer 1 (Rot)", "Jon Jones")
with col2:
    fighter_2 = st.text_input("Kämpfer 2 (Blau)", "Tom Aspinall")

if st.button("Vorhersage berechnen"):
    res = requests.get(f"http://api:8000/predict?fighter_1={fighter_1}&fighter_2={fighter_2}")
    
    if res.status_code == 200:
        data = res.json()
        st.subheader("Ergebnis:")
        st.write(f"**{fighter_1}:** {data['win_probability_fighter_1']}%")
        st.write(f"**{fighter_2}:** {data['win_probability_fighter_2']}%")