import os

from flask import Flask, render_template, request, jsonify
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from flask import Flask, render_template, request, jsonify
import constant

app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("base.html")

# Load the OpenAI API Key (You may need to replace 'YOUR_API_KEY' with the actual API key)
os.environ["OPENAI_API_KEY"] = constant.API_KEY

# Load the TextLoader and create the index with the ChatOpenAI model
loader = TextLoader('data.txt')
index = VectorstoreIndexCreator().from_loaders([loader])

@app.post("/predict")
def predict():
    text = request.get_json().get("message")

    # Get the response from the ChatGPT model
    response = index.query(text, llm=ChatOpenAI())

    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)