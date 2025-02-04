import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from constants import key
import os

# Set API key for the model
os.environ['GROQ_API_KEY'] = key

# Streamlit UI settings
st.set_page_config(page_title="Programming Q&A Chatbot", layout="wide")
st.title("🤖 Programming Q&A Chatbot")
st.markdown("""
<style>
.chat-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f8f9fa;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    font-family: Arial, sans-serif;
}
.user-message, .ai-message {
    padding: 10px 15px;
    margin-bottom: 10px;
    border-radius: 10px;
    font-size: 1rem;
    line-height: 1.5;
}
.user-message {
    background-color: #d1ecf1;
    color: #0c5460;
    text-align: right;
    border: 1px solid #bee5eb;
}
.ai-message {
    background-color: #f8d7da;
    color: #721c24;
    text-align: left;
    border: 1px solid #f5c6cb;
}
.input-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 20px;
}
.input-box {
    width: 75%;
    font-size: 1rem;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #ced4da;
    outline: none;
}
.submit-button {
    font-size: 1rem;
    padding: 10px 20px;
    margin-left: 10px;
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
}
.submit-button:hover {
    background-color: #0056b3;
}
</style>
""", unsafe_allow_html=True)

# Initialize the chatbot
chat = ChatGroq(temperature=0.5, model="llama3-8b-8192")

# Initialize session state for message flow
if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content="You are a Programming AI assistant")
    ]

# Function to load model and get responses
def get_chatmodel_response(question):
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    answer = chat(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))
    return answer.content

# Chat interface
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display conversation flow
if st.session_state['flowmessages']:
    for msg in st.session_state['flowmessages']:
        if isinstance(msg, HumanMessage):
            st.markdown(f'<div class="user-message">{msg.content}</div>', unsafe_allow_html=True)
        elif isinstance(msg, AIMessage):
            st.markdown(f'<div class="ai-message">{msg.content}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input box and submit button
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    input_text = st.text_input("Type your question:", key="input", placeholder="Ask me anything about programming...", label_visibility="collapsed")
    submit = st.button("Ask", key="submit", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Process the input if submitted
if submit and input_text:
    response = get_chatmodel_response(input_text)
    st.experimental_rerun()
