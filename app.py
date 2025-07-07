import os
from huggingface_hub import login
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

MODEL_REPO = "vinayabc1824/gemma-chatbot"  # Change to your repo

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForCausalLM.from_pretrained(MODEL_REPO, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

def answer_question(question):
    prompt = f"### Question: {question}\n### Answer: "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the answer part
    if '### Answer:' in answer:
        answer = answer.split('### Answer:')[-1].strip()
    return answer

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Ask a question"),
    outputs=gr.Textbox(label="Answer"),
    title="Gemma QA Chatbot",
    description="Ask any question and get an answer from a Gemma-2B-IT model fine-tuned on NQ-Open."
)

if __name__ == "__main__":
    demo.launch() 