# Gemma Chatbot

A QA chatbot powered by a fine-tuned Gemma-2B-IT language model, designed for answering open-domain questions. This project demonstrates model fine-tuning, deployment with Gradio, and a web interface for user interaction.

## Features
- **Fine-tuning**: Uses PEFT/LoRA to fine-tune Gemma-2B-IT on the NQ-Open dataset.
- **Gradio Interface**: Interactive QA chatbot UI for direct user queries.
- **Flask Backend**: REST API and web interface for chatbot access.
- **Visualization**: Jupyter notebook for analyzing chatbot usage and model performance.
- **Hugging Face Hub Integration**: Model and tokenizer are pushed to the Hugging Face Hub.

## Project Structure
```
├── app.py                # Gradio app for QA chatbot
├── backend.py            # Flask backend serving HTML and API
├── finetune.py           # Script for fine-tuning the model
├── requirements.txt      # Python dependencies
├── index.html            # Landing page for the chatbot demo
├── templates/
│   └── first.html        # Web chat interface template
└── myenv/                # (Optional) Python virtual environment
```

## Setup
1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd gemma_chatbot
   ```
2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set your Hugging Face token**
   - Obtain a token from [Hugging Face](https://huggingface.co/settings/tokens)
   - Set it as an environment variable:
     ```bash
     export HF_TOKEN=your_token_here  # On Windows: set HF_TOKEN=your_token_here
     ```

## Usage
### 1. Fine-tune the Model
Edit `finetune.py` if needed, then run:
```bash
python finetune.py
```
This will fine-tune the model on 1000 samples from NQ-Open and push it to the Hugging Face Hub.

### 2. Launch the Gradio Chatbot
```bash
python app.py
```
This starts a Gradio web interface for the chatbot.

### 3. Run the Flask Backend
```bash
python backend.py
```
This serves a web page (`first.html`) and a REST API for chatbot queries.

### 4. Deployment
The Finetuned Gemma model was deployed inside [Hugging Face](https://huggingface.co/spaces/vinayabc1824/gemma-chatbot) spaces.

## Web Interfaces
- **index.html**: Main landing page with information and an embedded chatbot demo.
- **templates/first.html**: Simple chat UI for interacting with the backend API.

## Requirements
See `requirements.txt`:
```
transformers
peft
trl
datasets
gradio
torch
huggingface_hub
```

## Credits
- Model: [Gemma-2B-IT](https://huggingface.co/google/gemma-2b-it)
- Dataset: [NQ-Open](https://huggingface.co/datasets/google-research-datasets/nq_open)
- Demo: [Hugging Face Space](https://huggingface.co/spaces/vinayabc1824/gemma-chatbot)

## License
This project is for educational and research purposes. See individual files for license details. 
