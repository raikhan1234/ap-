import streamlit as st
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ollama import Ollama

# Initialize FAISS index (using L2 distance and 512-dimensional vectors, you can adjust the dimensions)
dimension = 512  # Adjust according to the embedding size of your model
index = faiss.IndexFlatL2(dimension)

# Initialize Ollama model
model = Ollama("llama2")

st.title("Document Q&A with LLM")

# File upload section
uploaded_files = st.file_uploader("Upload .txt files", type=["txt"], accept_multiple_files=True)

if uploaded_files:
    document_vectors = []
    document_ids = []
    
    for file in uploaded_files:
        content = file.read().decode("utf-8")
        st.write(f"### {file.name}")
        st.text_area("File Content", content, height=200)

        # Split content into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(content)

        # Convert chunks into vectors and store in FAISS
        for i, chunk in enumerate(chunks):
            # Convert chunk text to a vector (use a real embedding model here)
            # For example, you can replace this with a model like OpenAI embeddings or Sentence-BERT
            vector = np.random.rand(dimension).astype('float32')  # Placeholder random vector
            
            document_vectors.append(vector)
            document_ids.append(f"{file.name}_chunk_{i}")
    
    # Convert the document vectors into a numpy array and add them to the FAISS index
    document_vectors = np.array(document_vectors)
    index.add(document_vectors)
    st.success("Files processed and stored in FAISS!")

# Ask a question
user_query = st.text_input("Ask a question:")
if st.button("Submit"):
    if user_query.strip():
        # Convert the user query into a vector (same as document chunks, use an actual embedding model)
        query_vector = np.random.rand(dimension).astype('float32')  # Placeholder random vector

        # Search in FAISS for the most similar documents
        distances, indices = index.search(np.array([query_vector]), k=5)

        # Gather the results
        context = " ".join([document_ids[i] for i in indices[0]])

        with st.spinner("Generating response..."):
            response = model.generate(f"{user_query}\n{context}")
            st.write(f"**Answer:** {response}")





