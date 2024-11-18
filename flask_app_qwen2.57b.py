import os
import warnings
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from flask_caching import Cache
from concurrent.futures import ThreadPoolExecutor
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms.base import LLM
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import threading
from typing import Optional

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# Flask app initialization
app = Flask(__name__)
app.secret_key = 'your_random_secret_key'
app.permanent_session_lifetime = timedelta(hours=1)
socketio = SocketIO(app, cors_allowed_origins="*")
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)
executor = ThreadPoolExecutor(max_workers=30)

# SQLite database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

# Define database model for user queries
class UserQuery(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    query = db.Column(db.String(500), nullable=False)
    response = db.Column(db.String(500), nullable=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# Define database model for visitor stats
class VisitorStats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    visitor_count = db.Column(db.Integer, nullable=False, default=0)
    question_count = db.Column(db.Integer, nullable=False, default=0)

# Create database tables
with app.app_context():
    db.create_all()

# Initialize Hugging Face's transformers model
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto"
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Embeddings initialization
embedding_model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Embeddings initialized with GPU support.")

# Default prompt template
default_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, politely say you don't know without guessing.

Context: {context}
Question: {question}

Helpful answer:"""
prompt = PromptTemplate(template=default_prompt_template, input_variables=['context', 'question'])

# Set the default vector store directory
VECTOR_STORE_DIR = "stores/admin_cosine"

# Initialize vector store
try:
    load_vector_store = Chroma(
        persist_directory=VECTOR_STORE_DIR,
        embedding_function=embeddings
    )
    print(f"Vector store loaded from {VECTOR_STORE_DIR}")
except Exception as e:
    print(f"Error loading vector store from {VECTOR_STORE_DIR}: {e}")
    load_vector_store = None

model_lock = threading.Lock()

# Define a custom LLM wrapper for LangChain
class CustomLLM(LLM):
    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        with model_lock:
            # Use your model inference function here
            return model_inference(prompt)

    @property
    def _identifying_params(self):
        return {"name": "Custom LLM"}

    @property
    def _llm_type(self):
        return "custom_llm"  # A string identifier for your custom LLM


# Model inference function
def model_inference(prompt):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

custom_llm = CustomLLM()

def async_inference(query):
    if load_vector_store is not None:
        chain_type_kwargs = {"prompt": prompt}
        qa = RetrievalQA.from_chain_type(
            llm=custom_llm,
            chain_type="stuff",
            retriever=load_vector_store.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs,
            verbose=True
        )
        response = qa.invoke(query)
        answer = response['result']
        doc = response['source_documents'][0].metadata['source']
        return {"query": query, "answer": answer, "doc": doc}
    else:
        return {"error": "Vector store not properly initialized. Please check your setup."}

@app.route('/')
def index():
    stats = VisitorStats.query.first()
    if not stats:
        stats = VisitorStats(visitor_count=1, question_count=0)
        db.session.add(stats)
    else:
        stats.visitor_count += 1
    db.session.commit()
    return render_template('index.html', visitor_count=stats.visitor_count, question_count=stats.question_count)

@app.route('/clear_cache')
def clear_cache():
    cache.clear()
    return "Cache cleared successfully!"

@app.route('/get_response', methods=['POST'])
def get_response():
    query = request.form.get('query')

    stats = VisitorStats.query.first()
    if stats:
        stats.question_count += 1
        db.session.commit()

    user_query = UserQuery(query=query)
    db.session.add(user_query)
    db.session.commit()

    future = executor.submit(async_inference, query)
    result = future.result()

    user_query.response = result["answer"]
    db.session.commit()

    response_data = {
        "answer": result["answer"],
        "doc": result["doc"]
    }

    return jsonify(response_data)

@socketio.on('connect')
def on_connect():
    emit('message', {'data': 'Connected'})

@socketio.on('disconnect')
def on_disconnect():
    print('Client disconnected')

@socketio.on('message')
def handle_message(data):
    user_query = UserQuery(query=data)
    db.session.add(user_query)
    db.session.commit()

    future = executor.submit(async_inference, data)
    result = future.result()

    user_query.response = result["answer"]
    db.session.commit()

    emit('message', {'data': result['answer']})

if __name__ == '__main__':
    socketio.run(app, debug=True, use_reloader=True, host='127.0.0.1', port=6501)
