import asyncio
import io

import numpy as np
from scipy.io import wavfile

import streamlit as st
from openai import OpenAI

from agents import Agent
from agents.voice import AudioInput, SingleAgentVoiceWorkflow, VoicePipeline

from chat_memory import ChatMemory

SAMPLE_RATE = 24000


def get_openai_client() -> OpenAI:
    """Get OpenAI client with API key from Streamlit secrets."""
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def transcribe_audio(audio_bytes: bytes) -> str:
    client = get_openai_client()
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    return transcript.text


def transcribe_audio_array(audio_array: np.ndarray, sample_rate: int) -> str:
    """Transcribe a numpy audio array by converting to WAV first."""
    wav_bytes = numpy_to_wav_bytes(audio_array, sample_rate)
    return transcribe_audio(wav_bytes)


def numpy_to_wav_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio array to WAV bytes for Streamlit playback."""
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio_array.astype(np.int16))
    buffer.seek(0)
    return buffer.read()


def build_instructions(history: list[dict]) -> str:
    base = "You're a mock dating advice assistant Dr. Steve S. Stevenson. Ask what the user wants advice on, be polite and nice, always respond in English."
    if not history:
        return base
    history_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in history
    )
    return f"{base}\n\nConversation history:\n{history_text}"


async def run_voice_agent(history: list[dict], audio_bytes: bytes) -> tuple[str, np.ndarray | None]:
    # Convert audio bytes to numpy array
    audio_file = io.BytesIO(audio_bytes)
    sample_rate, audio_data = wavfile.read(audio_file)
    audio_data = audio_data.astype(np.int16)
    
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]  # mono

    agent = Agent(
        name="Dating Assistant Dr. Steve S. Stevenson",
        instructions=build_instructions(history),
        model="gpt-4o-mini",
    )

    pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))
    audio_input = AudioInput(buffer=audio_data)
    result = await pipeline.run(audio_input)

    response_text = ""
    audio_chunks = []

    async for event in result.stream():
        if event.type == "voice_stream_event_audio":
            audio_chunks.append(event.data)
        elif event.type == "voice_stream_event_transcript":
            response_text = event.text

    # Combine audio chunks as numpy arrays
    if audio_chunks:
        response_audio = np.concatenate(audio_chunks)
    else:
        response_audio = None
        
    return response_text, response_audio


def apply_custom_css():
    """Apply custom CSS for layout: bordered chat container and fixed bottom input."""
    st.markdown("""
        <style>
        /* Main container adjustments */
        .main .block-container {
            padding-bottom: 120px;
        }
        
        /* Chat container styling */
        .chat-container {
            border: 1px solid #4a4a4a;
            border-radius: 10px;
            padding: 1rem;
            height: 60vh;
            overflow-y: auto;
            margin-bottom: 1rem;
            background-color: rgba(50, 50, 50, 0.2);
        }
        
        /* Fixed bottom input area */
        .fixed-bottom {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: var(--background-color);
            padding: 1rem 2rem;
            border-top: 1px solid #4a4a4a;
            z-index: 1000;
        }
        
        /* Adjust for sidebar */
        [data-testid="stSidebar"][aria-expanded="true"] ~ .main .fixed-bottom {
            left: 300px;
        }
        </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Dating Voice Demo", page_icon="🎙️", layout="centered")
    apply_custom_css()
    
    st.title("🎙️ Dating Voice Demo")

    # Initialize in-memory storage (persists during session)
    if "memory" not in st.session_state:
        st.session_state.memory = ChatMemory()
    
    memory = st.session_state.memory
    
    if "messages" not in st.session_state:
        st.session_state.messages = memory.get_history()
    
    # Store audio responses separately for playback
    if "audio_responses" not in st.session_state:
        st.session_state.audio_responses = {}
    
    # Track which message index should autoplay (only the latest)
    if "autoplay_idx" not in st.session_state:
        st.session_state.autoplay_idx = None
    
    # Use a key counter to reset the audio widget after processing
    if "audio_key" not in st.session_state:
        st.session_state.audio_key = 0

    # Chat container with border
    chat_container = st.container(height=450, border=True)
    
    with chat_container:
        if not st.session_state.messages:
            st.caption("Start a conversation by recording a message below.")
        
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                # Show audio player for assistant messages if available
                if msg["role"] == "assistant" and idx in st.session_state.audio_responses:
                    # Only autoplay the most recent response
                    should_autoplay = (idx == st.session_state.autoplay_idx)
                    st.audio(
                        st.session_state.audio_responses[idx], 
                        format="audio/wav",
                        autoplay=should_autoplay
                    )
        
        # Clear autoplay after rendering so it doesn't replay on rerun
        st.session_state.autoplay_idx = None

    # Spacer to push content above the fixed input
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
    
    # Audio input at the bottom
    audio_value = st.audio_input("🎤 Record a message", key=f"audio_input_{st.session_state.audio_key}")

    if audio_value:
        audio_bytes = audio_value.read()

        # Transcribe user audio
        user_text = transcribe_audio(audio_bytes)

        # Save user message
        memory.add_message("user", user_text)
        st.session_state.messages.append({"role": "user", "content": user_text})

        # Get agent response
        with st.spinner("Thinking..."):
            history = memory.get_history()
            response_text, response_audio = asyncio.run(
                run_voice_agent(history, audio_bytes)
            )
        
        # If no transcript was returned, transcribe the response audio
        if not response_text and response_audio is not None:
            response_text = transcribe_audio_array(response_audio, SAMPLE_RATE)
        
        # Fallback if still no text
        if not response_text:
            response_text = "(Audio response)"

        # Save assistant message
        memory.add_message("assistant", response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        # Store audio for this message index and mark for autoplay
        if response_audio is not None:
            wav_bytes = numpy_to_wav_bytes(response_audio, SAMPLE_RATE)
            msg_idx = len(st.session_state.messages) - 1
            st.session_state.audio_responses[msg_idx] = wav_bytes
            st.session_state.autoplay_idx = msg_idx  # Mark this one for autoplay
        
        # Increment key to reset the audio input widget
        st.session_state.audio_key += 1
        st.rerun()


if __name__ == "__main__":
    main()
