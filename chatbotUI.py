import streamlit as st
import numpy as np
import requests

API_URL = "http://127.0.0.1:8000/chat"
st.title("Chatbot")
# name = "Menna"

if "messages" not in st.session_state:
    st.session_state.messages = []

for role, messages in st.session_state.messages:
    with st.chat_message(role):
        st.write(messages)


user_input = st.chat_input("Ask anything")

if user_input:
    st.session_state.messages.append(("user", user_input))
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Loading"):

    #send the request to API
            try:
                response = requests.post(API_URL, json={"prompt": user_input})
                if response.status_code == 200:
                    answer = response.json()["answer"]
                else:
                    answer = f"Error: {response.status_code} - {response.text}"
            except Exception as ex:
                answer = f"Connection error: {ex}"  
        st.write(answer) 
        
    st.session_state.messages.append(("assistant", answer))
    


