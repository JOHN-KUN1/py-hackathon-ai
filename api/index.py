from flask import Flask, request, jsonify
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.gemini import Gemini
from llama_index.core import StorageContext, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.embeddings.gemini import GeminiEmbedding
from dotenv import load_dotenv
import os
load_dotenv()

app = Flask(__name__)

api_key = os.getenv('API_KEY')
# Initialize your LLM and required components
def initialize_llm():
    documents = SimpleDirectoryReader("./data").load_data()
    db = chromadb.PersistentClient(path="./embeddings/chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")

    Settings.llm = Gemini(
        api_key=api_key,
        model="models/gemini-pro",
        temperature=0.9,
    )
    Settings.node_parser = SentenceSplitter(chunk_size=1200, chunk_overlap=20)
    Settings.embed_model = GeminiEmbedding(
        model_name="models/embedding-001", api_key=api_key, title="this is a document"
    )

    vector_store = ChromaVectorStore(chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    template = (
        "We have provided context information below which will serve as a user's transaction history."
        "With \"Description\" being the category that the user has spent his or her money on. Debit means that the money leaves the users account and credit means money that comes into the users account.\n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please answer the question: {query_str}\n"
    )

    qa_template = PromptTemplate(template)

    query_engine = index.as_query_engine(text_qa_template=qa_template)
    return query_engine

query_engine = initialize_llm()

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query = data.get('query', '')

    if query:
        response = query_engine.query(query)
        return jsonify({"response": str(response)}), 200
    else:
        return jsonify({"error": "No query provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)
