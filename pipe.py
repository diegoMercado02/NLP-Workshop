# app.py

import streamlit as st
import PyPDF2
import nltk
import tempfile
import os
import subprocess
import asyncio
import edge_tts
import numpy as np
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# Set page configuration
st.set_page_config(page_title="PDF to Podcast Converter", layout="wide")

st.title("PDF to Podcast Converter with Embeddings and Dialogue")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
    return text

# Function to split text into chunks
def split_text_into_chunks(text, max_chunk_size=500):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Function to generate embeddings
def generate_embeddings(chunks):
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(chunks)
    return embeddings

# Function to select relevant chunks based on embeddings
def select_relevant_chunks(chunks, embeddings, query_embedding, top_n=5):
    similarities = np.dot(embeddings, query_embedding)
    top_indices = similarities.argsort()[-top_n:][::-1]
    selected_chunks = [chunks[i] for i in top_indices]
    return selected_chunks

# Function to generate podcast script with dialogue
def generate_podcast_script(chunks):
    dialogues = []
    system_message = """You are an expert podcast script writer. Your task is to:
    1. Convert technical content into engaging dialogue
    2. Maintain accuracy while making content accessible
    3. Create natural transitions between topics
    4. Include relevant examples and analogies
    5. Keep a consistent tone throughout the conversation"""

    # Initialize the language model
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')

    for i, chunk in enumerate(chunks):
        prompt = f"""
{system_message}

Convert this content into a natural podcast dialogue:

CONTENT:
{chunk}

REQUIREMENTS:
1. Use 'Host' and 'Expert' as speakers
2. Format as:
   **Host:** [dialogue]
   **Expert:** [dialogue]
3. Include:
   - 1-2 clarifying questions from the Host
   - Real-world examples or analogies
   - Natural transitions
4. Maintain technical accuracy while being conversational
5. Keep responses concise (2-3 sentences per speaker turn)
"""

        # Generate dialogue
        try:
            response = generator(prompt, max_length=500, do_sample=True, temperature=0.7, num_return_sequences=1)
            dialogue = response[0]['generated_text']
            # Extract the dialogue part after the prompt
            dialogue = dialogue[len(prompt):].strip()
            dialogues.append(dialogue)
        except Exception as e:
            st.error(f"Error generating dialogue for chunk {i}: {str(e)}")
            continue

    # Create a conclusion
    conclusion_prompt = """
Create a conclusion that:
1. Summarizes key points
2. Provides a memorable takeaway
3. Includes a call to action
4. Thanks the audience
Keep it under 1 minute when spoken.
"""

    try:
        conclusion_response = generator(conclusion_prompt, max_length=150, do_sample=True, temperature=0.7, num_return_sequences=1)
        conclusion = conclusion_response[0]['generated_text']
        conclusion = conclusion[len(conclusion_prompt):].strip()
        dialogues.append(conclusion)
    except Exception as e:
        st.error(f"Error generating conclusion: {str(e)}")

    return '\n\n'.join(dialogues)

# Function to generate audio asynchronously
async def generate_audio_segment(text, voice, output_path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

# Function to concatenate audio files using FFmpeg
def concatenate_audio(file_list, output_path):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for audio_file in file_list:
            f.write(f"file '{audio_file}'\n")
        concat_list = f.name

    try:
        command = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_list,
            '-c', 'copy',
            output_path
        ]
        subprocess.run(command, capture_output=True, check=True)
    finally:
        os.unlink(concat_list)

# Function to generate audio from dialogue text
def generate_audio(dialogue_text):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_files = []
            lines = dialogue_text.splitlines()

            for i, line in enumerate(lines):
                if not line.strip():
                    continue

                if line.startswith("**Host:**") or line.startswith("**Expert:**"):
                    speaker, _, text = line.partition(':')
                    text = text.strip()
                    speaker = speaker.strip('*').strip()

                    if not text:
                        continue

                    # Choose voice based on speaker
                    if speaker == "Host":
                        voice = "en-US-GuyNeural"    # Male voice for Host
                    elif speaker == "Expert":
                        voice = "en-US-JennyNeural"  # Female voice for Expert
                    else:
                        continue  # Skip unknown speakers

                    temp_path = os.path.join(temp_dir, f'segment_{i}.mp3')
                    asyncio.run(generate_audio_segment(text, voice, temp_path))
                    audio_files.append(temp_path)

            if audio_files:
                output_path = os.path.join(temp_dir, 'combined.mp3')
                concatenate_audio(audio_files, output_path)

                # Read the final audio file
                with open(output_path, 'rb') as f:
                    return f.read()
            else:
                st.error("No audio files were generated.")
                return None

    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

# Main functionality
if uploaded_file is not None:
    raw_text = extract_text_from_pdf(uploaded_file)
    st.subheader("Extracted Text")
    st.write(raw_text)

    text_chunks = split_text_into_chunks(raw_text)
    chunk_embeddings = generate_embeddings(text_chunks)

    # Get user input for query or topic
    query = st.text_input("Enter a topic or keywords to focus on (optional):", value="")
    if query:
        embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        query_embedding = embedder.encode([query])[0]
        selected_chunks = select_relevant_chunks(text_chunks, chunk_embeddings, query_embedding)
    else:
        # Select chunks that best represent the content
        centroid = np.mean(chunk_embeddings, axis=0)
        selected_chunks = select_relevant_chunks(text_chunks, chunk_embeddings, centroid)

    st.subheader("Selected Text Chunks for Podcast Script")
    for idx, chunk in enumerate(selected_chunks):
        st.markdown(f"**Chunk {idx+1}:** {chunk}")

    podcast_script = generate_podcast_script(selected_chunks)
    st.subheader("Generated Podcast Script")
    st.write(podcast_script)

    audio_bytes = generate_audio(podcast_script)
    if audio_bytes:
        st.audio(audio_bytes, format='audio/mp3')
        st.download_button(
            label="Download Podcast",
            data=audio_bytes,
            file_name="podcast.mp3",
            mime="audio/mpeg"
        )
    else:
        st.error("Failed to generate audio.")
else:
    st.info("Please upload a PDF file to proceed.")
