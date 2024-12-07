# %% [markdown]
#  # PDF to Podcast Generator
#  This notebook converts PDF documents into engaging podcast-style audio content using AI. The goial was to create a tool that allows me understanding pdf by just listening for example while biking. This pdf can be fromn the internet as it can be my notes for a presentation, which is really fun to listen to compared to reading them.

# %%


# %% [markdown]
#  ## Import Required Libraries
#  First, we'll import all the libraries we need for processing PDFs, generating text, and creating audio.

# %%
import streamlit as st
import requests
import os
import tempfile
import subprocess
from dotenv import load_dotenv
from pypdf import PdfReader
import tensorflow as tf
from transformers import pipeline
import edge_tts
import asyncio


# %% [markdown]
#  ## Load Environment Variables
#  We need to load the API key from the environment.

# %%
load_dotenv()

# %% [markdown]
#  ## Set Up Constants
#  These are the main settings we'll use throughout the program I decided on grok-beta as XAI has a open testing for up to 25 euros of credits, which was plenty for this project.

# %%
XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-beta"
AVERAGE_SPEAKING_SPEED_WPM = 150

# %% [markdown]
#  ## PDF Text Extraction
#  This function gets all the text from a PDF file and organizes it by page.

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



# %%

def summarize_text(text):
    """Summarize text using a Hugging Face pipeline."""
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Error summarizing text: {str(e)}")
        return None

# Summarize each chunk of text
def summarize_text_data(text_data):
    summarized_text_data = []
    for item in text_data:
        summary = summarize_text(item['text'])
        if summary:
            summarized_text_data.append({
                'text': summary,
                'page': item['page'],
                'chars': len(summary)
            })
    return summarized_text_data

# %% [markdown]
#  ## File Processing
#  This function handles the uploaded PDF file safely.

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
#  ## Script Generation
#  This function turns our text into a natural conversation between a host and expert. I use the same formatting for both but change the label, this way its easy to process the audio.

# %%
def generate_podcast_script(summarized_text_data, first_page_text):
    try:
        dialogues = []
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {XAI_API_KEY}'
        }

        system_message = {
            "role": "system",
            "content": """You are an expert podcast script writer. Your task is to:
                    1. Convert technical content into engaging dialogue
                    2. Maintain accuracy while making content accessible
                    3. Create natural transitions between topics
                    4. Include relevant examples and analogies
                    5. Keep a consistent tone throughout the conversation
                    REQUIREMENTS:
                    1. Use 'Host' and 'Expert' as speakers
                    2. Always Format as:
                       **Host:** [dialogue]
                       **Expert:** [dialogue]"""
        }

        # Generate introduction
        introduction_prompt = {
            "role": "user",
            "content": f"""Create an engaging introduction for the podcast that includes:
                1. A brief overview of the content
                2. An introduction of the host 'Diego' as Host 
                3. An introduction of an author, if no writters name refer in the dialog as expert
                3. A hook to capture the audience's attention
                4. Mention that this is an AI-generated podcast
                CONTENT:
                {first_page_text}
                TEMPLATE:
                Host: Welcome to "Diego's AI podcasts," your go-to podcast for diving into complex topics with the help of artificial intelligence! I'm your host, Diego, and today we have a special guest who is an expert in [Expertise Area]. Joining us is [Expert Name], a [brief description of expertise]. They've been at the forefront of [mention relevant achievements or contributions]. Today, we'll be exploring [brief overview of podcast topic], delving deep into [specific aspect of the topic]. Get ready to dive into the fascinating world of [Theme/Topic]!
                Expert: Thank you, Diego! I'm thrilled to be here and discuss [specific aspect of the topic]. It's an exciting time in [Expertise Area], and I'm looking forward to sharing insights and exploring new ideas with you.  Let's get started!"""
        }
        data = {
            'model': MODEL,
            'messages': [system_message, introduction_prompt],
            'temperature': 0.7,
            'max_tokens': 1000,
            'top_p': 0.9,
            'frequency_penalty': 0.2,
            'presence_penalty': 0.2
        }
        response = requests.post(XAI_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            introduction = response.json()['choices'][0]['message']['content']
            dialogues.append(introduction)
        else:
            st.error(f"API error for introduction: {response.status_code} - {response.text}")

        # Generate Q&A section
        for i, item in enumerate(summarized_text_data):
            qna_prompt = {
                "role": "user",
                "content": f"""Create a Q&A section based on the following content:
                CONTENT:
                {item['text']}
                REQUIREMENTS:
                1. Use 'Host' and 'Expert' as speakers
                2. Format as:
                   **Host:** [dialogue]
                   **Expert:** [dialogue]
                3. Include:
                   - 1-2 clarifying questions from the Host
                   - Real-world examples or analogies
                   - Natural transitions
                   - Relevant technical details
                   - Paraphrase the questions and answers in a podcast-friendly way, not a literal Q&A
                   - 
                4. Maintain technical accuracy while being conversational
                5. Keep responses concise (2-3 sentences per speaker turn)
                6. Do not use phrases like today we are talking about, the topic of today is, etc.
                    REQUIREMENTS:
                    1. Use 'Host' and 'Expert' as speakers
                    2. Always Format as:
                       **Host:** [dialogue]
                       **Expert:** [dialogue]"""
            }
            data['messages'] = [system_message, qna_prompt]
            response = requests.post(XAI_API_URL, headers=headers, json=data)
            if response.status_code == 200:
                qna = response.json()['choices'][0]['message']['content']
                dialogues.append(qna)
            else:
                st.error(f"API error for Q&A section {i+1}: {response.status_code} - {response.text}")

        # Generate conclusion
        conclusion_prompt = {
            "role": "user",
            "content": f"""Create a conclusion that:
                1. Summarizes most relevant points from the following Q&A sections:
                {''.join(dialogues[1:])} while still keeping in the context of the cover page: {first_page_text}
                2. Provides a memorable takeaway
                3. Includes a call to action
                4. Thanks the audience
                5. In the dialog always refer to the host by their name and the expert as expert
                Keep it under 1 minute when spoken.
                REQUIREMENTS:
                    1. Use 'Host' and 'Expert' as speakers
                    2. Always Format as:
                       **Host:** [dialogue]
                       **Expert:** [dialogue]"""
        }
        data['messages'] = [system_message, conclusion_prompt]
        response = requests.post(XAI_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            conclusion = response.json()['choices'][0]['message']['content']
            dialogues.append(conclusion)
        else:
            st.error(f"API error for conclusion: {response.status_code} - {response.text}")

        return '\n\n'.join(dialogues)

    except Exception as e:
        st.error(f"Error generating podcast script: {str(e)}")
        return ""
    
# %% [markdown]
#  ## Audio Processing
#  These functions handle creating and combining audio files. This way it can create 2 of them then concatenate them and create a new one

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
#  ## Text-to-Speech Generation
#  This function converts our script into spoken audio using different voices. I tried gtts but it doesn't allow for 2 voices, and some of the more complex ones require more setup. Edge_tts was the best one I could find with multiple voices, this still sounds a bit robotic but still can be listened to.

# %%

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
#  ## Streamlit Web Interface
#  This creates our user interface for uploading PDFs and generating podcasts.


# %%
def main():
    st.title("PDF to Podcast Generator")
    st.write("Upload a PDF to generate a dialogue script and convert it to audio with distinct voices.")

    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"], accept_multiple_files=False)

    generate_button = st.button("Generate Script and Audio")

    if generate_button and uploaded_file is not None:
        try:
            with st.spinner("Processing PDF and creating script..."):
                temp_pdf_path = process_uploaded_file(uploaded_file)

                if temp_pdf_path:
                    text_data = extract_text_from_pdf(temp_pdf_path)
                    if not text_data:
                        st.error("Failed to extract text from the PDF.")
                        return

                    st.subheader("Extracted Text Data")
                    st.write(text_data)

                    summarized_text_data = summarize_text_data(text_data)
                    if not summarized_text_data:
                        st.error("Failed to summarize text from the PDF. Pages may be too long.")
                        return

                    st.subheader("Summarized Text Data")
                    st.write(summarized_text_data)

                    first_page_text = text_data[0]['text'] if text_data else summarized_text_data[0:4]['text']

                    script = generate_podcast_script(summarized_text_data, first_page_text)

                    if script:
                        st.subheader("Generated Script")
                        st.text_area("Script", script, height=400)

                        st.write("Generating audio...")
                        audio_data = generate_audio(script)
                        if audio_data:
                            st.session_state['audio_data'] = audio_data
                            st.audio(audio_data, format="audio/wav")

                            st.download_button(
                                "Download Complete Podcast Audio",
                                st.session_state['audio_data'],
                                file_name="complete_podcast.wav",
                                mime="audio/wav",
                                key="download_button"
                            )
                        else:
                            st.error("Failed to generate audio.")
                    else:
                        st.error("Failed to generate script.")

                    os.unlink(temp_pdf_path)

        except KeyboardInterrupt:
            st.warning("Process interrupted by user.")
            tf.keras.backend.clear_session()
            st.stop()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# %% [markdown]
# This file is originally a .py so I can run it with streamlit, but i converted it into a notebook to add explanations

if __name__ == "__main__":
    main()
