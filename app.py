import os
import json
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# Load the Hugging Face API token from a configuration file
working_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(working_dir, "config.json")

# Check if the configuration file exists
if not os.path.exists(config_file_path):
    raise FileNotFoundError(f"Configuration file not found at {config_file_path}")

# Load the configuration file
with open(config_file_path, "r") as config_file:
    config_data = json.load(config_file)

# Get the API token from the configuration data
huggingface_token = config_data.get("HUGGINGFACE_API_TOKEN")
if huggingface_token is None:
    raise ValueError("HUGGINGFACE_API_TOKEN not found in the configuration file")

# Load the LLaMA 3.1 model
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=huggingface_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id, token=huggingface_token, torch_dtype=torch.bfloat16, device_map="auto"
)

# Configuring Streamlit page settings
st.set_page_config(page_title="Llama Chat", page_icon="💬", layout="wide")

# Initialize chat session in Streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "system",
            "content": "You are a super useful assistant that is able to provide relevant and factually accurate advice. Please do not provide any medical, legal, or financial advice. If you are unsure about a response, please let the user know.",
        },
        {
            "role": "assistant",
            "content": "Hello! I'm Llama3.1, a chatbot trained to provide information and answer questions. Feel free to ask me anything!",
        },
    ]

st.markdown(
    """
    <style>
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Generation Parameters")
    temperature = st.slider(
        "Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1
    )
    top_k = st.slider("Top K", min_value=1, max_value=100, value=50)
    top_p = st.slider("Top P", min_value=0.1, max_value=1.0, value=0.95, step=0.05)
    max_new_tokens = st.slider(
        "Max New Tokens", min_value=64, max_value=2048, value=256, step=64
    )

    st.button("Clear Chat", on_click=lambda: st.session_state.pop("chat_history", None))

# Main chat interface
st.title("🤖 Llama3.1 - ChatBot")

# Create a container for the scrollable chat history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
chat_container = st.container()
st.markdown("</div>", unsafe_allow_html=True)

# Create a container for the fixed input field at the bottom
input_container = st.container()

# Display chat history in the scrollable container
with chat_container:
    for message in st.session_state.chat_history[1:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Input field for user's message (fixed at the bottom)
with input_container:
    user_prompt = st.chat_input("Ask Llama...", key="user_input")

if user_prompt:
    # Add user's message to chat and display it
    with chat_container:
        st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Prepare the input for the model
    input_text = ""
    for message in st.session_state.chat_history:
        input_text += f"{message['role']}: {message['content']}\n"
    input_text += "assistant:"

    # Generate a response using the LLaMA 3.1 model and measure the time it takes
    start_time = time.time()
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    assistant_response = tokenizer.decode(output[0], skip_special_tokens=True)
    end_time = time.time()

    # Calculate the time taken for response generation
    generation_time = end_time - start_time
    print(f"Response generated in {generation_time:.2f} seconds")

    # Extract the assistant's response correctly
    assistant_response = assistant_response.split("assistant:")[-1].strip()

    st.session_state.chat_history.append(
        {"role": "assistant", "content": assistant_response}
    )

    # Display Llama's response
    with chat_container:
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
