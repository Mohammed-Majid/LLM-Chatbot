# Opensource LLM Chatbot

This repository contains code for a chatbot application built using Streamlit and the LLaMA 3.1 model from Hugging Face. The application allows users to interact with the chatbot, customize generation parameters, and view the chat history.

Note that you are not limited to Llama3.1, the code allows you to try out different models from the huggingface transoformers library (provided you have the compute to run them).

## Features

- **Interactive Chat Interface**: Engage in conversations with the LLaMA 3.1 model.
- **Customizable Parameters**: Adjust temperature, top-k, top-p, and maximum new tokens for generating responses.
- **Scrollable Chat History**: View and scroll through past messages in the chat.

## Prerequisites

To run this application, you need:

- Python 3.7 or later
- An API token for Hugging Face
- Required Python packages

## Installation

1. **Clone the Repository**:

    ```bash
    git clone 
    ```

2. **Install Required Packages**:

    install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. **Modify Configuration File**:

   locate the file named `config.json` in the root directory of the project and input your API key:

    ```json
    {
        "HUGGINGFACE_API_TOKEN": "(INPUT API KEY)"
    }
    ```

    Replace `(INPUT API KEY)` with your actual Hugging Face API token.

## Usage

1. **Run the Application**:

    ```bash
    streamlit run app.py
    ```

2. **Interact with the Chatbot**:

    - Use the input field to send messages to the chatbot.
    - Adjust the generation parameters using the sliders in the sidebar.
    - Click "Clear Chat" to reset the chat history.

## Configuration Parameters

- **Temperature**: Controls the randomness of responses. Higher values make the output more random.
- **Top K**: Limits the sampling to the top K most probable tokens.
- **Top P**: Controls nucleus sampling; it samples from the smallest set of tokens with a cumulative probability above this threshold.
- **Max New Tokens**: Specifies the maximum number of tokens to generate in the response.
