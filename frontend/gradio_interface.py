import gradio as gr
import requests


# Define the function to call FastAPI
def call_nlp_task(task, text):
    url = "http://127.0.0.1:8000/nlp-task/"
    response = requests.post(url, json={"task": task, "text": text})
    return response.json().get("result", "Error")


# Build the Gradio interface
def build_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("# Tasks dropdown")
                task_dropdown = gr.Dropdown(
                    ["Summarization", "Sentiment Analysis", "Text Generation"], label="Select Task"
                )
                gr.Markdown("# About the app")
                gr.Markdown("This app performs NLP tasks like summarization, sentiment analysis, and text generation.")

            with gr.Column(scale=3):
                input_text = gr.Textbox(lines=5, placeholder="Enter your text here...", label="Input Text")
                send_button = gr.Button("Send")
                output_text = gr.Textbox(lines=5, placeholder="Output will appear here...", label="Output Text")

                # Scrollable Textboxes
                # input_text.style(scrollbar=True)
                # output_text.style(scrollbar=True)

                send_button.click(fn=call_nlp_task, inputs=[task_dropdown, input_text], outputs=output_text)

    return demo


# Launch Gradio interface
if __name__ == "__main__":
    demo_interface = build_interface()
    demo_interface.launch()
