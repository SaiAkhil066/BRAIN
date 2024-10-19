import warnings
from flask import Flask, render_template, redirect, url_for, request, jsonify, session, send_from_directory
from flask_socketio import SocketIO, emit
from flask_caching import Cache
from concurrent.futures import ThreadPoolExecutor
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from datetime import timedelta
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import logging
import threading

import torch


warnings.filterwarnings(
    "ignore",
    message="Torch was not compiled with flash attention",
)

# Suppress specific warning
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
app = Flask(__name__)
app.secret_key = 'your_random_secret_key'
app.permanent_session_lifetime = timedelta(hours=1)
socketio = SocketIO(app, cors_allowed_origins="*")
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)
executor = ThreadPoolExecutor(max_workers=30)

# Configure the Flask application to use SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)


user_visitor_counts = {}


# Define a model for storing user queries and responses
class UserQuery(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False)
    query = db.Column(db.String(500), nullable=False)
    response = db.Column(db.String(500), nullable=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
# Create the database tables before running the application
with app.app_context():
    db.create_all()

# Initialize LLM and other components as in the original code
local_llm = "openhermes-2.5-mistral-7b.Q8_0.gguf"
config = {
    'config.config.max_new_tokens': 200, #400 to 200
    'repetition_penalty': 0.3, #0.3 to 0.1
    'temperature': 0.0,
    'top_k': 50, #50 to 20
    'top_p': 0.9, #0.9 to 0.5
    'stream': True,
    'threads': 30,
    'config.config.context_length': 4096
}

# CUDA Check
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

llm = CTransformers(
    model=local_llm,
    model_type="llama",  # Change from "ollama" to "llama"
    lib="cuda",
    **config        
)


print("LLM Initialized with GPU support.")

# Add the mappings of usernames to vector store paths
user_vector_stores = {
    'admin': 'stores/admin_cosine',
    'training': 'stores/training_cosine',
    'manpower': 'stores/manpower_cosine',
    'promotion': 'stores/promotion_cosine',
    'record': 'stores/record_cosine',
    'test': 'stores/test_cosine',
    'english': 'stores/english_cosine',
    'dictionary': 'stores/dict_cosine',
    'policy' : 'stores/policy_cosine',


}

load_vector_store = None  # Declare load_vector_store as a global variable

# Add this dictionary at the beginning of your file
prompt_templates = {
    'admin': """Use the following pieces of information to answer the data strictly stored from the only admin's database to the user's question. Understand the User's question and search from the pieces of information you have, reply with the accurate answer. Acknowledge the User's Question and Use "Sir" for every response.
If you don't know the answer, just say that you don't know sir and sorry politely, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
""",
    'training': """Use the following pieces of information to answer the data strictly stored from the only training's database to the user's question. Understand the User's question and search from the pieces of information you have, reply shortly with good meaning. Acknowledge the User's Question and Use "Sir" for every response.
If you don't know the answer, just say that you don't know sir and sorry politely, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
""",
    'manpower': """Use the following pieces of information to answer the data strictly stored from the only manpower's database to the user's question. Understand the User's question and search from the pieces of information you have, reply shortly with good meaning. Acknowledge the User's Question and Use "Sir" for every response.
If you don't know the answer, just say that you don't know sir and sorry politely, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
""",
    'promotion': """Use the following pieces of information to answer the data strictly stored from the only promotion's database to the user's question. Understand the User's question and search from the pieces of information you have, reply shortly with good meaning. Acknowledge the User's Question and Use "Sir" for every response.
If you don't know the answer, just say that you don't know sir and sorry politely, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
""",
    'record': """Use the following pieces of information to answer the data strictly stored from the only record's database to the user's question. Understand the User's question and search from the pieces of information you have, reply shortly with good meaning. Acknowledge the User's Question and Use "Sir" for every response.
If you don't know the answer, just say that you don't know sir and sorry politely, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
""",
    'test': """Use the following pieces of information to answer the data strictly stored from the only test's database to the user's question. Understand the User's question and search from the pieces of information you have, reply shortly with good meaning. Acknowledge the User's Question and Use "Sir" for every response.
If you don't know the answer, just say that you don't know sir and sorry politely, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
""",
    'english': """Use the following pieces of information to answer the data strictly stored from the only english's database to the user's question. Understand the User's question and search from the pieces of information you have, You're an English language expert and behave like that, reply shortly with good meaning. Acknowledge the User's Question and Use "Sir" for every response.
If you don't know the answer, just say that you don't know sir and sorry politely, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
""",
    'dictionary': """Use the following pieces of information to answer the data strictly stored from the only dict's database to the user's question. Understand the User's question, Ignore the spelling mistakes, correct them and search from the pieces of information you have, reply shortly with good meaning. Acknowledge the User's Question and Use "Sir" for every response.
If you don't know the answer, just say that you don't know sir and sorry politely, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
""",
    'policy': """Use the following pieces of information to answer the data strictly stored from the only policy's database to the user's question. Understand the User's question, Ignore the spelling mistakes, correct them and search from the pieces of information you have, reply with the accurate answers. Acknowledge the User's Question and Use "Sir" for every response.
If you don't know the answer, just say that you don't know sir and sorry politely, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
""",
# Add more roles as needed
}

@app.route('/dictionary')
def dictionary():
    session['username'] = 'dictionary'
    return redirect(url_for('index'))



# Add this line before the app.before_request function
prompt = None
# Modify the before_request function to set the appropriate vector store and prompt template


@app.before_request
def before_request():
    global load_vector_store, prompt

    # Check which device is being used
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")

    if 'username' in session:
        username = session['username']
        if username in user_vector_stores:
            store_path = user_vector_stores[username]
            load_vector_store = Chroma(persist_directory=store_path, embedding_function=embeddings)
            print(f"User logged in: {username}")

            # Set the prompt template based on the user's role
            prompt_template = prompt_templates.get(username, """Use the following pieces of information to answer the data strictly stored from database to the user's question. Understand the User's question and search from the pieces of information you have, reply shortly with good meaning. Acknowledge to the User's Question and Use "Sir" for every response.
                If you don't know the answer, just say that you don't know politely, don't try to make up an answer.

                Context: {context}
                Question: {question}

                Only return the helpful answer below and nothing else.
                Helpful answer:
                """)
            prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])



model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Embeddings initialized with GPU support.")
# prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])


model_lock = threading.Lock()

def print_gpu_stats():
    print(f"Allocated GPU memory: {torch.cuda.memory_allocated()} bytes")
    print(f"Cached GPU memory: {torch.cuda.memory_reserved()} bytes")
    
def async_inference(query, username):
    with model_lock:
        # CUDA Check
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Running on CPU")

        if load_vector_store is not None:
            chain_type_kwargs = {"prompt": prompt}
            qa = RetrievalQA.from_chain_type(
                llm=llm,
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





@app.route('/pdfs/<filename>')
def pdfs(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'pdfs'), filename)

@app.route('/')
def index():
    # Check if the user is logged in
    if 'username' in session:
        # Increment visitor count for the current user
        username = session['username']
        user_visitor_counts.setdefault(username, 0)
        user_visitor_counts[username] += 1
        # Check if the session has expired
        if session.permanent and session.modified:
            return redirect(url_for('login'))
        return render_template('index.html', username=session['username'], visitor_count=user_visitor_counts[username])
    else:
        # Redirect to the login page if not logged in
        return redirect(url_for('login'))

@app.route('/logout')
def user_logout():
    # Remove the username from the session if it exists
    session.pop('username', None)
    # Redirect to the login page after logout
    return redirect(url_for('login'))

# Clear Cache route
@app.route('/clear_cache')
def clear_cache():
    cache.clear()
    return "Cache cleared successfully! You can confirm that the cache is cleared by checking your application behavior or logs."
print("cache cleared")

# Update the route to pass the username explicitly
@app.route('/get_response', methods=['POST'])
def get_response():
    query = request.form.get('query')
    username = session.get('username', 'Anonymous')

    # Log the user query
    user_query = UserQuery(username=username, query=query)
    db.session.add(user_query)
    db.session.commit()

    # Asynchronously process the query
    future = executor.submit(async_inference, query, username)

    # Wait for the result
    result = future.result()

    # Log the user response
    user_query.response = result["answer"]
    db.session.commit()

    response_data = {
        "answer": result["answer"],
        "doc": result["doc"]
    }

    return jsonify(response_data)


# Batch processing route
@app.route('/get_responses', methods=['POST'])
def get_responses():
    queries = request.form.getlist('queries')

    # Asynchronously process queries
    futures = [executor.submit(async_inference, query) for query in queries]

    # Collect results
    results = [future.result() for future in futures]

    return jsonify(results)

# Define a dictionary of valid usernames and corresponding passwords
valid_credentials = {
    'admin': 'admin',
    'training': 'training',
    'manpower': 'manpower',
    'promotion': 'promotion',
    'record' : 'record',
    'test' : 'test',
    'english' : 'english',
    'dictionary' : 'dictionary',
    'policy' : 'policy',

}

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in valid_credentials and valid_credentials[username] == password:
            session['username'] = username
            session.permanent = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials. Please try again.")
    return render_template('login.html')


@socketio.on('connect')
def on_connect():
    emit('message', {'data': 'Connected'})

@socketio.on('disconnect')
def on_disconnect():
    print('Client disconnected')

@socketio.on('message')
def handle_message(data):
    username = session.get('username', 'Anonymous')
    user_query = UserQuery(username=username, query=data)
    db.session.add(user_query)
    db.session.commit()

    future = executor.submit(async_inference, data, username)
    result = future.result()

    user_query.response = result["answer"]
    db.session.commit()

    emit('message', {'data': result['answer']})



@socketio.on('submit_query')
def handle_query(data):
    try:
        queries = data['data']['queries']

        # Your logic to handle multiple queries
        results = []
        for query in queries:
            chain_type_kwargs = {"prompt": prompt}
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=load_vector_store.as_retriever(search_kwargs={"k": 1}),
                return_source_documents=True,
                chain_type_kwargs=chain_type_kwargs,
                verbose=True
            )
            response = qa(query)
            answer = response['result']
            doc = response['source_documents'][0].metadata['source']
            results.append({"query": query, "answer": answer, "doc": doc})

        # Emit the results back to the client using WebSocket
        emit('response', {'data': results})

    except Exception as e:
        # Emit an error back to the client
        emit('response', {'error': str(e)})

@app.route('/logout')
def logout():
    # Remove the username from the session if it exists
    session.pop('username', None)
    # Redirect to the login page after logout
    return redirect(url_for('login'))

if __name__ == '__main__':
    socketio.run(app, debug=True, use_reloader=True, host='127.0.0.1', port=6500)
