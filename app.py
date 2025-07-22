import gradio as gr
from llama_cpp import Llama
import os
import sys
import functools
import argparse


# --- Configuration & Argument Parsing ---
def parse_arguments():
    """Parses command-line arguments for the chatbot application."""
    default_model_path = "./models/Qwen3-8B-Q8_0.gguf"
    # Use environment variable as a fallback, but command-line args take precedence.
    env_model_path = os.getenv("MODEL_PATH", default_model_path)

    parser = argparse.ArgumentParser(
        description="Gradio Chatbot for Qwen models with SSN detection."
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=env_model_path,
        help=f'The path to the GGUF model file. Defaults to: {env_model_path}'
    )
    parser.add_argument(
        '--nothink',
        action='store_true',
        help="Disable the model's 'think' mode."
    )
    args = parser.parse_args()
    return args.model_path, not args.nothink

MODEL_PATH, enable_think_mode = parse_arguments()

if not enable_think_mode:
    print("Think mode has been disabled via the '--nothink' command-line argument.")
print(f"Using model: {MODEL_PATH}")

# --- Model Existence Check ---
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}", file=sys.stderr)
    print("Please download the GGUF model and place it in the 'models' directory.", file=sys.stderr)
    print("Or, set the MODEL_PATH environment variable or use the --model_path argument.", file=sys.stderr)
    sys.exit(1)

# --- Model Loading ---
print("Loading model... This may take a few minutes.")

try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,  # Offload all layers to GPU. Set to 0 if no GPU is available.
        n_ctx=4096,       # Context window size
        verbose=False     # Set to True for more detailed logging from llama.cpp
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    sys.exit(1)


# --- Chatbot Logic ---
def chat_with_qwen(message, history, think_mode=True):
    """
    Handles the chat interaction with the Qwen3 model.
    """
    # Convert Gradio's history format to the format expected by llama-cpp-python
    # A detailed system prompt with few-shot examples for SSN detection.
    system_prompt = """You are a helpful and secure AI assistant.

Your primary responsibility is to protect user privacy. You must never process, store, or respond to queries containing a Social Security Number (SSN).

An SSN is a nine-digit number, often formatted as XXX-XX-XXXX, XXX XX XXXX, or XXXXXXXXX. It is a sensitive US government identifier. Do not confuse it with phone numbers or other numerical data.

**Rule:** If you detect a potential SSN in the user's query, you must immediately stop and respond with ONLY the following exact message: "Sensitive SSN is detected, and the query is blocked."

Here are some examples:

User: "My SSN is 123-45-6789, can you check its validity?"
Assistant: "Sensitive SSN is detected, and the query is blocked."

User: "I think my social is 987 65 4321, what should I do?"
Assistant: "Sensitive SSN is detected, and the query is blocked."

User: "Can you call me at 555-867-5309 to discuss my account?"
Assistant: "As an AI, I cannot make phone calls, but I'd be happy to help you here. What is your question about your account?"

User: "What is the capital of France?"
Assistant: "The capital of France is Paris."

For all other queries that do not contain an SSN, answer helpfully and thoughtfully."""

    system_prompt = system_prompt + " /think" if think_mode else system_prompt + " /nothink"
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    
    messages.append({"role": "user", "content": message})
    # Append user message and a placeholder for bot's response to the history for display
    history.append([message, ""])

    # Print the user message to the console for debugging
    print(f"User: {message}")

    # Generate a response from the model
    response_stream = llm.create_chat_completion(
        messages=messages,
        stream=True
    )
    # Stream the response back to the UI
    full_response = ""
    is_thinking = False
    for chunk in response_stream:
        content = ""
        # Extract the content from the response chunk
        delta = chunk['choices'][0]['delta']
        if 'content' in delta and delta['content'] is not None:
            content = delta['content']
        if not content:
            continue

        # print(chunk)  # Debug: Print the raw chunk to see its structure
        # Print the thought process to the console, the think starts with tag <think> ends with </think>
        if is_thinking or (think_mode and '<think>' in content):
            if not is_thinking:
                is_thinking = True
                print("Thinking: ", end='', flush=True)
            elif '</think>' in content:
                is_thinking = False
                print()
            else:
                print(content, end='', flush=True)
            
            continue
        
        # If not thinking, append the content to the full response
        full_response += content
        history[-1][1] = full_response # Update the bot's response in the history
        yield history # Yield the entire updated history

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """<h1 style="text-align: center;">Ask me anything, but not anything about SSN!</h1>"""
    )
    chatbot = gr.Chatbot(label="Chat History", type="tuples", height=400, layout="bubble")
    msg = gr.Textbox(label="Your Message", placeholder="Type your question here and press Enter...", scale=7)
    clear = gr.ClearButton([msg, chatbot])
    # Create a partial function to pass the think_mode setting from the command line.
    chat_function_with_mode = functools.partial(chat_with_qwen, think_mode=enable_think_mode)

    msg.submit(chat_function_with_mode, [msg, chatbot], chatbot)
    # msg.submit(chat_with_qwen, [msg, chatbot], chatbot)
    msg.submit(lambda: "", None, msg) # Clear the textbox after submit

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
