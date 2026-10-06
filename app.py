import os
import uuid

import streamlit as st

from dotenv import load_dotenv
from pypdf import PdfReader

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


# Load environment variables

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


# Create OpenAI client

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)


# Create Pinecone client

pinecone_client = Pinecone(
    api_key=PINECONE_API_KEY
)


# Create Pinecone index if it does not exist

existing_indexes = pinecone_client.list_indexes().names()


if PINECONE_INDEX_NAME not in existing_indexes:

    pinecone_client.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


index = pinecone_client.Index(
    PINECONE_INDEX_NAME
)


# Page configuration

st.set_page_config(
    page_title="AI Document Q&A",
    page_icon="📄"
)


st.title("📄 AI Document Q&A Assistant")

st.write(
    "Upload a document and ask questions about its content."
)


# --------------------------------------------------
# Extract text from PDF
# --------------------------------------------------

def extract_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# --------------------------------------------------
# Split text into chunks
# --------------------------------------------------

def split_text(text, chunk_size=1000, overlap=200):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end - overlap

    return chunks


# --------------------------------------------------
# Generate embedding
# --------------------------------------------------

def generate_embedding(text):

    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


# --------------------------------------------------
# Store document chunks in Pinecone
# --------------------------------------------------

def store_chunks(chunks):

    vectors = []

    document_id = str(uuid.uuid4())


    for i, chunk in enumerate(chunks):

        embedding = generate_embedding(chunk)


        vectors.append(
            {
                "id": f"{document_id}-{i}",

                "values": embedding,

                "metadata": {
                    "text": chunk,
                    "chunk_number": i
                }
            }
        )


    index.upsert(
        vectors=vectors
    )


# --------------------------------------------------
# Retrieve relevant chunks
# --------------------------------------------------

def retrieve_context(question, top_k=5):

    question_embedding = generate_embedding(
        question
    )


    result = index.query(
        vector=question_embedding,
        top_k=top_k,
        include_metadata=True
    )


    context = []


    for match in result["matches"]:

        metadata = match.get("metadata", {})

        text = metadata.get("text", "")

        if text:
            context.append(text)


    return context


# --------------------------------------------------
# Generate answer using LLM
# --------------------------------------------------

def generate_answer(question, context):

    combined_context = "\n\n".join(context)


    prompt = f"""
You are an AI document assistant.

Answer the user's question using only the
information provided in the document context.

If the answer is not present in the context,
say that the information is not available
in the uploaded document.

Document Context:

{combined_context}

Question:

{question}
"""


    response = openai_client.chat.completions.create(

        model="gpt-4o-mini",

        messages=[
            {
                "role": "system",
                "content":
                    "You answer questions using retrieved document context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0
    )


    return response.choices[0].message.content


# --------------------------------------------------
# Upload document
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)


if uploaded_file is not None:

    if st.button("Process Document"):

        with st.spinner("Processing document..."):

            # Step 1: Extract text

            text = extract_text(
                uploaded_file
            )


            if not text.strip():

                st.error(
                    "Could not extract text from this PDF."
                )

            else:

                # Step 2: Chunk text

                chunks = split_text(text)


                # Step 3: Generate embeddings
                # Step 4: Store in Pinecone

                store_chunks(chunks)


                st.session_state["document_processed"] = True

                st.success(
                    f"Document processed successfully. "
                    f"Created {len(chunks)} chunks."
                )


# --------------------------------------------------
# Question answering
# --------------------------------------------------

if st.session_state.get(
    "document_processed",
    False
):

    st.subheader("Ask a Question")

    question = st.text_input(
        "Enter your question"
    )


    if st.button("Ask"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner("Generating answer..."):

                # Retrieve relevant chunks

                context = retrieve_context(
                    question
                )


                # Generate answer

                answer = generate_answer(
                    question,
                    context
                )


            st.subheader("Answer")

            st.write(answer)


            # Show retrieved context

            with st.expander(
                "View Retrieved Context"
            ):

                for i, chunk in enumerate(context):

                    st.write(
                        f"**Chunk {i + 1}**"
                    )

                    st.write(chunk)

                    st.divider()
