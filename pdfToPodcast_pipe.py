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
from transformers import pipeline, GPTNeoForCausalLM, GPT2Tokenizer
from pypdf import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
import asyncio
import edge_tts

nltk.download('punkt')

load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-beta"
AVERAGE_SPEAKING_SPEED_WPM = 150

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def chunk_text(text, max_length=1024):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_podcast_script(text):
    model_name = 'EleutherAI/gpt-neo-125M'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
    pad_token_id = tokenizer.eos_token_id
    system_message = """You are an expert podcast script writer. Your task is to:
    1. Convert technical content into engaging dialogue
    2. Maintain accuracy while making content accessible
    3. Create natural transitions between topics
    4. Include relevant examples and analogies
    5. Keep a consistent tone throughout the conversation"""

    chunks = chunk_text(text)
    podcast_script = ""
    for chunk in chunks:
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
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        attention_mask = input_ids.ne(pad_token_id).long().to(device)
        response = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=150, pad_token_id=pad_token_id)
        podcast_script += tokenizer.decode(response[0], skip_special_tokens=True) + " "
    return podcast_script

def generate_audio(dialogue_text):
    try:
        st.write("Starting generate_audio function")
        st.write(f"Input dialogue_text length: {len(dialogue_text)}")

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

                    if speaker == "Host":
                        voice = "en-US-GuyNeural"
                    elif speaker == "Expert":
                        voice = "en-US-JennyNeural"
                    else:
                        st.error(f"Unknown speaker '{speaker}' in line {i}")
                        continue

                    try:
                        temp_path = os.path.join(temp_dir, f'segment_{i}.mp3')
                        communicate = edge_tts.Communicate(text, voice)
                        asyncio.run(communicate.save(temp_path))
                        audio_files.append(temp_path)

                    except Exception as inner_e:
                        st.error(f"Error generating audio for line {i}: {str(inner_e)}")
                        continue

            if audio_files:
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

                with open(output_path, 'rb') as f:
                    return f.read()

            return None

    except Exception as e:
        st.error(f"Error in generate_audio: {str(e)}")
        return None

def concatenate_audio(wav_files, output_path):
    data = []
    for wav_file in wav_files:
        audio, samplerate = sf.read(wav_file)
        data.append(audio)
    combined = np.concatenate(data)
    sf.write(output_path, combined, samplerate)

st.title("PDF to Podcast Converter with Embeddings and Dialogue")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Extracting text from PDF...")
    raw_text = extract_text_from_pdf(uploaded_file)
    progress_bar.progress(25)
    
    if raw_text:
        status_text.text("Generating podcast script...")
        podcast_script = generate_podcast_script(raw_text)
        progress_bar.progress(50)
        
        st.subheader("Generated Podcast Script")
        st.write(podcast_script)
        
        status_text.text("Generating audio...")
        audio_data = generate_audio(podcast_script)
        progress_bar.progress(75)
        
        if audio_data:
            status_text.text("Audio generation complete!")
            st.audio(audio_data, format='audio/wav')
            progress_bar.progress(100)
        else:
            status_text.text("Failed to generate audio.")
            progress_bar.progress(0)
    else:
        status_text.text("Failed to extract text from PDF.")
        progress_bar.progress(0)
else:
    st.info("Please upload a PDF file to proceed.")