# %% [markdown]
# # PDF to Podcast Generator
# This notebook converts PDF documents into engaging podcast-style audio content using AI.

# %% [markdown]
# ## Import Required Libraries
# First, we'll import all the libraries we need for processing PDFs, generating text, and creating audio.

# %%
import streamlit as st
import requests
import os
import tempfile
import numpy as np
import torch
import subprocess
import soundfile as sf
from io import BytesIO
from dotenv import load_dotenv
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from pypdf import PdfReader

# %% [markdown]
# ## Load Environment Variables
# We need to load our API keys and other settings from the environment.

# %%
load_dotenv()

# %% [markdown]
# ## Set Up Constants
# These are the main settings we'll use throughout the program.

# %%
XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-beta"
AVERAGE_SPEAKING_SPEED_WPM = 150

# %% [markdown]
# ## PDF Text Extraction
# This function gets all the text from a PDF file and organizes it by page.

# %%
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with better error handling and metadata."""
    try:
        reader = PdfReader(pdf_path)
        text_data = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            text_data.append({
                'text': text,
                'page': page_num + 1,
                'chars': len(text)
            })
        return text_data
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

# %% [markdown]
# ## Text Chunking
# We break down the text into smaller, manageable pieces while keeping sentences together.

# %%
def chunk_text(text_data, chunk_size=500, overlap=50):
    """Improved chunking with semantic boundaries and metadata preservation."""
    chunks = []
    for page_data in text_data:
        text = page_data['text']
        words = text.split()

        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space

            # Check if chunk is complete at sentence boundary
            if current_size >= chunk_size and word[-1] in '.!?':
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'page': page_data['page'],
                    'size': current_size
                })
                # Keep overlap words for context
                overlap_words = current_chunk[-int(overlap/5):]  # Approximate words for overlap
                current_chunk = overlap_words
                current_size = sum(len(word) + 1 for word in overlap_words)

        # Add remaining chunk if it's substantial
        if current_size > overlap:
            chunks.append({
                'text': ' '.join(current_chunk),
                'page': page_data['page'],
                'size': current_size
            })

    return chunks

# %% [markdown]
# ## File Processing
# This function handles the uploaded PDF file safely.

# %%
def process_uploaded_file(uploaded_file):
    """Process uploaded file and create temporary file."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
        return None

# %% [markdown]
# ## Word Count Management
# These functions help control the length of the final podcast.

# %%
def calculate_max_words(max_minutes):
    return max_minutes * AVERAGE_SPEAKING_SPEED_WPM

def trim_chunks_to_max_words(chunks, max_words):
    try:
        total_words = 0
        trimmed_chunks = []
        for chunk in chunks:
            chunk_word_count = len(chunk.split())
            if total_words + chunk_word_count > max_words and trimmed_chunks:
                break
            trimmed_chunks.append(chunk)
            total_words += chunk_word_count
        if not trimmed_chunks and chunks:
            trimmed_chunks.append(chunks[0])  # Ensure at least one chunk
        st.write(f"Trimmed to {len(trimmed_chunks)} chunks")  # Debugging info
        return trimmed_chunks
    except Exception as e:
        st.error(f"Error trimming chunks: {str(e)}")
        return []

# %% [markdown]
# ## Script Generation
# This function turns our text into a natural conversation between a host and expert.

# %%
def generate_podcast_script(chunks, format_type):
    try:
        dialogues = []
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {XAI_API_KEY}'
        }

        # Initial system message to set context
        system_message = {
            "role": "system",
            "content": """You are an expert podcast script writer. Your task is to:
                            1. Convert technical content into engaging dialogue
                            2. Maintain accuracy while making content accessible
                            3. Create natural transitions between topics
                            4. Include relevant examples and analogies
                            5. Keep a consistent tone throughout the conversation"""
        }

        for i, chunk in enumerate(chunks):
            # Extract metadata if available
            page_num = 'unknown'  # Update if page number is available

            if format_type == "Podcast":
                messages = [
                    system_message,
                    {
                        "role": "user",
                        "content": f"""
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
                    }
                ]
            else:  # Short-form Video Script
                messages = [
                    system_message,
                    {
                        "role": "user",
                        "content": f"""
                        Convert this content into a short-form video script:

                        CONTENT:
                        {chunk}

                        REQUIREMENTS:
                        1. Use 'Host' and 'Expert' as speakers
                        2. Format as:
                           **Host:** [dialogue]
                           **Expert:** [dialogue]
                        3. Keep it engaging and fast-paced
                        4. Focus on one key point or insight
                        5. Include a hook in the first 3 seconds
                        6. Total length should be under 60 seconds when spoken
                        """
                    }
                ]

            data = {
                'model': MODEL,
                'messages': messages,
                'temperature': 0.7,
                'max_tokens': 1000,
                'top_p': 0.9,
                'frequency_penalty': 0.2,
                'presence_penalty': 0.2
            }

            st.write(f"Processing chunk {i+1}/{len(chunks)}")
            response = requests.post(XAI_API_URL, headers=headers, json=data)

            if response.status_code == 200:
                result = response.json()
                dialogue = result['choices'][0]['message']['content']
                dialogues.append(dialogue)

                # Add transition prompt if not the last chunk
                if i < len(chunks) - 1:
                    transition_prompt = {
                        "role": "user",
                        "content": "Generate a smooth transition to the next topic that maintains flow and engagement."
                    }
                    data['messages'] = [system_message, transition_prompt]
                    transition_response = requests.post(XAI_API_URL, headers=headers, json=data)
                    if transition_response.status_code == 200:
                        transition = transition_response.json()['choices'][0]['message']['content']
                        dialogues.append(transition)
            else:
                st.error(f"API error for chunk {i+1}: {response.status_code} - {response.text}")

        # Generate conclusion
        conclusion_prompt = {
            "role": "user",
            "content": """Create a conclusion that:
                1. Summarizes key points
                2. Provides a memorable takeaway
                3. Includes a call to action
                4. Thanks the audience
                Keep it under 1 minute when spoken."""
        }

        data['messages'] = [system_message, conclusion_prompt]
        conclusion_response = requests.post(XAI_API_URL, headers=headers, json=data)
        if conclusion_response.status_code == 200:
            conclusion = conclusion_response.json()['choices'][0]['message']['content']
            dialogues.append(conclusion)

        return '\n\n'.join(dialogues)

    except Exception as e:
        st.error(f"Error generating podcast script: {str(e)}")
        return ""

# %% [markdown]
# ## Audio Processing
# These functions handle creating and combining audio files.

# %%
def concatenate_audio(file_list, output_path):
    """Concatenate audio files using FFmpeg"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for audio_file in file_list:
            f.write(f"file '{audio_file}'\n")
        concat_list = f.name

    try:
        command = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', concat_list,
            '-c:a', 'pcm_s16le',
            '-ar', '16000',
            output_path
        ]
        subprocess.run(command, capture_output=True, check=True)
    finally:
        os.unlink(concat_list)

# %% [markdown]
# ## Text-to-Speech Generation
# This function converts our script into spoken audio using different voices.

# %%
import asyncio
import edge_tts

def generate_audio(dialogue_text):
    try:
        st.write("Starting generate_audio function")
        st.write(f"Input dialogue_text length: {len(dialogue_text)}")

        # Create a temporary directory to store audio files
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_files = []
            lines = dialogue_text.splitlines()
            st.write(f"Number of lines split: {len(lines)}")

            for i, line in enumerate(lines):
                st.write(f"Processing line {i}: {line}")
                if not line.strip():
                    continue

                if line.startswith("**Host:**") or line.startswith("**Expert:**"):
                    speaker, _, text = line.partition(':')
                    st.write(f"Raw speaker: {speaker}")
                    text = text.strip().lstrip("*").strip()
                    speaker = speaker.strip().lstrip("*").strip()
                    st.write(f"Processed speaker: {speaker}")
                    st.write(f"Text: {text}")

                    if not text:
                        continue

                    # Choose voice based on speaker
                    if speaker == "Host":
                        voice = "en-US-GuyNeural"    # Male voice for Host
                    elif speaker == "Expert":
                        voice = "en-US-JennyNeural"  # Female voice for Expert
                    else:
                        st.error(f"Unknown speaker '{speaker}' in line {i}")
                        continue

                    try:
                        temp_path = os.path.join(temp_dir, f'segment_{i}.mp3')

                        # Generate speech asynchronously
                        communicate = edge_tts.Communicate(text, voice)
                        asyncio.run(communicate.save(temp_path))

                        audio_files.append(temp_path)

                    except Exception as inner_e:
                        st.error(f"Error generating audio for line {i}: {str(inner_e)}")
                        continue

            if audio_files:
                # Convert mp3 files to wav format for concatenation
                wav_files = []
                for mp3_file in audio_files:
                    wav_file = mp3_file.replace('.mp3', '.wav')
                    command = [
                        'ffmpeg', '-y', '-i', mp3_file,
                        '-ar', '16000',
                        '-ac', '1',
                        '-c:a', 'pcm_s16le',
                        wav_file
                    ]
                    subprocess.run(command, capture_output=True, check=True)
                    wav_files.append(wav_file)

                output_path = os.path.join(temp_dir, 'combined.wav')
                concatenate_audio(wav_files, output_path)

                # Read the final audio file
                with open(output_path, 'rb') as f:
                    return f.read()

            return None

    except Exception as e:
        st.error(f"Error in generate_audio: {str(e)}")
        return None

# %% [markdown]
# ## Streamlit Web Interface
# This creates our user interface for uploading PDFs and generating podcasts.

# %%
def main():
    st.title("PDF to Podcast Generator")
    st.write("Upload a PDF to generate a dialogue script and convert it to audio with distinct voices.")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf", accept_multiple_files=False)

    # Dropdown for content format
    content_format = st.selectbox("Select Content Format", ["Podcast", "Short-form Video Script (TikTok, Reels, Shorts)"])

    # Input for max podcast length
    max_length = st.number_input("Maximum Length (minutes)", min_value=1, max_value=60, value=5, step=1)

    # Button to generate script and audio
    generate_button = st.button("Generate Script and Audio")

    if generate_button and uploaded_file is not None:
        try:
            with st.spinner("Processing PDF and creating script..."):
                # Save uploaded file temporarily
                temp_pdf_path = process_uploaded_file(uploaded_file)

                if temp_pdf_path:
                    # Extract text from PDF
                    text_data = extract_text_from_pdf(temp_pdf_path)
                    if not text_data:
                        st.error("Failed to extract text from the PDF.")
                        return

                    # Create text chunks
                    chunks_data = chunk_text(text_data)
                    chunks = [chunk['text'] for chunk in chunks_data]

                    if chunks:
                        st.success("Successfully created text chunks!")

                        # Trim chunks to fit desired podcast length
                        max_words = calculate_max_words(max_length)
                        trimmed_chunks = trim_chunks_to_max_words(chunks, max_words)

                        # Generate the script
                        script = generate_podcast_script(trimmed_chunks, content_format)

                        if script:
                            st.subheader("Generated Script")
                            st.text_area("Script", script, height=400)

                            st.write("Generating audio...")
                            audio_data = generate_audio(script)
                            if audio_data:
                                # Play audio
                                st.audio(audio_data, format="audio/wav")

                                # Download button with key to prevent app reset
                                st.download_button(
                                    "Download Complete Podcast Audio",
                                    audio_data,
                                    file_name="complete_podcast.wav",
                                    mime="audio/wav",
                                    key="download_button"
                                )
                            else:
                                st.error("Failed to generate audio.")
                        else:
                            st.error("Failed to generate script.")
                    else:
                        st.error("No text chunks were created from the PDF.")

                    # Cleanup temporary file
                    os.unlink(temp_pdf_path)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# %% [markdown]
# ## Run the Application

# %%
if __name__ == "__main__":
    main()
