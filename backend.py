from flask import Flask, request, jsonify, render_template
from gradio_client import Client
import time
import requests

app = Flask(__name__)

# Step 1: Wait for HF space to be ready
def wait_for_space_ready(space_url, retries=10, delay=5):
    for i in range(retries):
        try:
            print(f"Checking if Hugging Face Space is ready... (Attempt {i + 1})")
            res = requests.get(space_url, timeout=10)
            if res.status_code == 200:
                print("✅ Hugging Face Space is ready.")
                return
        except Exception as e:
            print(f"❌ Space not ready yet: {e}")
        time.sleep(delay)
    raise Exception("❌ Space did not become available in time.")

wait_for_space_ready("https://vinayabc1824-gemma-chatbot.hf.space")
client = Client("vinayabc1824/gemma-chatbot")

# Route to serve HTML page
@app.route("/", methods=["GET"])
def home():
    return render_template("first.html")

# Route to handle question
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    print(f"❓ Question received: {question}")
    try:
        answer = client.predict(question, api_name="/predict")
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
