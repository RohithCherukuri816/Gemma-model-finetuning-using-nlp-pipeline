import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from huggingface_hub import login

# 1. Login to Hugging Face (set your token as HF_TOKEN env var)
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError('Please set your Hugging Face token in the HF_TOKEN environment variable.')
login(token=hf_token)

# 2. Load dataset (1000 samples)
dataset = load_dataset('google-research-datasets/nq_open', split='train[:1000]')

def preprocess(example):
    question = example['question']
    answers = example['answer']
    answer = answers[0] if answers else 'Unknown'
    return {
        'text': f"### Question: {question}\n### Answer: {answer}"
    }

dataset = dataset.map(preprocess)

# 3. Load tokenizer and model
model_name = 'google/gemma-2b-it'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

# 4. Apply LoRA (PEFT)
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)

# 5. Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
    output_dir="./gemma-qa-lora",
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)

# 6. SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    formatting_func=lambda x: x['text']
)

# 7. Train
trainer.train()

# 8. Push to Hugging Face Hub
repo_id = "vinayabc1824/gemma-chatbot"  
trainer.model.push_to_hub(repo_id, use_temp_dir=False)
tokenizer.push_to_hub(repo_id, use_temp_dir=False) 