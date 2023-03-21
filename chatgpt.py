import openai
import gradio as gr

openai.api_key = "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"

messages = [
    {"role": "system", "content": "You are a helpful and kind AI Assistant."},
]

def openai_chat(input):
    if input:
        messages.append({"role": "user", "content": input})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return reply

def chatbot(input, history=[]):
    output = openai_chat(input)
    history.append((input, output))
    return history, history


inputs = gr.inputs.Textbox(lines=7, label="Chat with AI")
outputs = gr.outputs.Textbox(label="Reply")

gr.Interface(fn=chatbot, inputs=["text",'state'], outputs=["chatbot",'state'], title="AI Chatbot",
             description="Ask anything you want",
             theme="compact").launch(debug=True)