import streamlit as st
import requests
import uuid

st.title("AI chatbot")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask Gemini"):

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role" :"user", "content": user_input})

    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                "http://localhost:8000/chat",
                json={"session_id": st.session_state.session_id,"message":user_input}
            )
            response.raise_for_status()
            ai_reply = response.json().get("reply")

            with st.chat_message("Assistant"):
                st.markdown(ai_reply)
            st.session_state.messages.append({"role" :"assistant", "content": ai_reply})

        except Exception as e:
            st.error(e)