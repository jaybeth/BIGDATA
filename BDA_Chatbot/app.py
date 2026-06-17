import os
import faiss
import numpy as np
import streamlit as st
import google.generativeai as genai

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ==========================
# GEMINI API KEY
# ==========================
genai.configure(api_key="AQ.Ab8RN6KTRXUy_IRpvnWfkIWhjxg85zp_c92LYIjf7l6Ljelb8w")

# ==========================
# PDF FILES
# ==========================
PDF_FILES = [
    "U20IT701 BDA UNIT1.pdf",
    "U20IT701 BDA UNIT222.pdf",
    "U20IT701 BDA UNIT3.pdf",
    "U20IT701 BDA UNIT4444.pdf",
    "U20IT701 BDA UNIT5555.pdf"
]

# ==========================
# LOAD EMBEDDING MODEL
# ==========================
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================
# EXTRACT PDF TEXT
# ==========================
def extract_text_from_pdfs():

    full_text = ""

    for pdf in PDF_FILES:

        reader = PdfReader(pdf)

        for page in reader.pages:

            text = page.extract_text()

            if text:
                full_text += text + "\n"

    return full_text

# ==========================
# CHUNKING
# ==========================
def chunk_text(text, chunk_size=1000):

    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    return chunks

# ==========================
# VECTOR STORE
# ==========================
@st.cache_resource
def create_vector_db():

    text = extract_text_from_pdfs()

    chunks = chunk_text(text)

    embeddings = embedding_model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index, chunks

# ==========================
# RETRIEVAL
# ==========================
def retrieve_context(question, index, chunks, top_k=5):

    q_embedding = embedding_model.encode([question])

    distances, indices = index.search(
        np.array(q_embedding),
        top_k
    )

    context = ""

    for idx in indices[0]:
        context += chunks[idx] + "\n\n"

    return context

# ==========================
# GEMINI ANSWER
# ==========================
def ask_gemini(question, context):

    model = genai.GenerativeModel(
        "gemini-2.5-flash"
    )

    prompt = f"""
You are a Big Data Analytics tutor.

Answer ONLY from the provided context.

Context:
{context}

Question:
{question}

If answer is not present in context,
say:
'Answer not found in uploaded PDFs.'
"""

    response = model.generate_content(prompt)

    return response.text

# ==========================
# STREAMLIT UI
# ==========================
st.set_page_config(
    page_title="BDA PDF Chatbot",
    page_icon="🤖"
)

st.title("🤖 Big Data Analytics PDF Chatbot")

st.write("Ask questions from all 5 uploaded PDFs")

index, chunks = create_vector_db()

question = st.text_input(
    "Enter your question"
)

if question:

    with st.spinner("Searching PDFs..."):

        context = retrieve_context(
            question,
            index,
            chunks
        )

        answer = ask_gemini(
            question,
            context
        )

    st.success(answer)
