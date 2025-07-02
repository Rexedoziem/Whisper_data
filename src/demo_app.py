"""
Demo application for Whisper Nigerian Pidgin transcription
"""

import streamlit as st
import numpy as np
import librosa
import tempfile
import os
from pathlib import Path
import logging

# Import our modules
from inference_engine import WhisperPidginInference, PidginPostProcessor
from training_pipeline import WhisperPidginPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PidginTranscriptionDemo:
    def __init__(self):
        self.inference_engine = None
        self.post_processor = PidginPostProcessor()
        
    def load_model(self, model_path: str = None):
        """Load the inference model"""
        try:
            if model_path and Path(model_path).exists():
                self.inference_engine = WhisperPidginInference(model_path)
                st.success(f"Loaded fine-tuned model from {model_path}")
            else:
                self.inference_engine = WhisperPidginInference()
                st.info("Using base Whisper model (fine-tuned model not found)")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.inference_engine = None
    
    def transcribe_audio(self, audio_data, sample_rate=16000):
        """Transcribe audio data"""
        if self.inference_engine is None:
            return {"error": "Model not loaded"}
        
        try:
            # Resample if necessary
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            
            # Transcribe
            result = self.inference_engine.transcribe(audio_data)
            
            # Post-process
            normalized_text = self.post_processor.normalize_pidgin(result["text"])
            result["normalized_text"] = normalized_text
            
            return result
        except Exception as e:
            return {"error": str(e)}

def main():
    st.set_page_config(
        page_title="Nigerian Pidgin Speech Transcription",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 Nigerian Pidgin Speech Transcription")
    st.markdown("Fine-tuned Whisper model for transcribing Nigerian Pidgin English")
    
    # Initialize demo app
    if 'demo_app' not in st.session_state:
        st.session_state.demo_app = PidginTranscriptionDemo()
    
    demo_app = st.session_state.demo_app
    
    # Sidebar for model management
    with st.sidebar:
        st.header("Model Settings")
        
        # Model loading
        model_path = st.text_input(
            "Model Path (optional)",
            placeholder="models/whisper-pidgin-final",
            help="Path to fine-tuned model directory"
        )
        
        if st.button("Load Model"):
            demo_app.load_model(model_path if model_path else None)
        
        st.markdown("---")
        
        # Training section
        st.header("Training")
        if st.button("Start Training Pipeline"):
            with st.spinner("Training model... This may take a while"):
                try:
                    pipeline = WhisperPidginPipeline(
                        base_model="openai/whisper-small",
                        output_dir="models/whisper-pidgin-streamlit"
                    )
                    
                    # Quick training config for demo
                    config = {
                        "num_train_epochs": 2,
                        "per_device_train_batch_size": 1,
                        "per_device_eval_batch_size": 1,
                        "learning_rate": 1e-4,
                        "warmup_steps": 10,
                        "logging_steps": 5,
                        "save_steps": 50,
                        "eval_steps": 50
                    }
                    
                    results = pipeline.run_full_pipeline(
                        use_sample_data=True,
                        augment_data=False,
                        training_config=config
                    )
                    
                    st.success("Training completed!")
                    st.json(results["evaluation_results"])
                    
                except Exception as e:
                    st.error(f"Training failed: {e}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Audio Input")
        
        # Audio upload
        uploaded_file = st.file_uploader(
            "Upload Audio File",
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Upload an audio file to transcribe"
        )
        
        # Audio recording (placeholder - would need additional setup)
        st.markdown("**Record Audio** (Feature coming soon)")
        st.info("Audio recording functionality would require additional browser permissions and setup")
        
        # Sample audio generation for demo
        if st.button("Generate Sample Audio"):
            st.info("Generating synthetic audio for demonstration...")
            # Create synthetic audio
            duration = 3  # seconds
            sample_rate = 16000
            t = np.linspace(0, duration, duration * sample_rate)
            frequency = 440  # A4 note
            audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            st.audio(audio_data, sample_rate=sample_rate)
            st.session_state.demo_audio = audio_data
            st.session_state.demo_sample_rate = sample_rate
    
    with col2:
        st.header("Transcription Results")
        
        # Process uploaded audio
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                # Load audio
                audio_data, sample_rate = librosa.load(tmp_path, sr=None)
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                # Display audio
                st.audio(uploaded_file)
                
                # Transcribe
                if st.button("Transcribe Uploaded Audio"):
                    with st.spinner("Transcribing..."):
                        result = demo_app.transcribe_audio(audio_data, sample_rate)
                        
                        if "error" in result:
                            st.error(f"Transcription error: {result['error']}")
                        else:
                            st.success("Transcription completed!")
                            
                            # Display results
                            st.markdown("**Original Transcription:**")
                            st.write(result.get("text", "No transcription"))
                            
                            st.markdown("**Normalized Pidgin:**")
                            st.write(result.get("normalized_text", "No normalization"))
                            
                            st.markdown("**Confidence Score:**")
                            confidence = result.get("confidence", 0)
                            st.progress(confidence)
                            st.write(f"{confidence:.2%}")
                            
                            st.markdown("**Processing Time:**")
                            st.write(f"{result.get('processing_time', 0):.2f} seconds")
                
            except Exception as e:
                st.error(f"Error processing audio file: {e}")
        
        # Process demo audio
        if hasattr(st.session_state, 'demo_audio'):
            if st.button("Transcribe Sample Audio"):
                with st.spinner("Transcribing sample audio..."):
                    result = demo_app.transcribe_audio(
                        st.session_state.demo_audio,
                        st.session_state.demo_sample_rate
                    )
                    
                    if "error" in result:
                        st.error(f"Transcription error: {result['error']}")
                    else:
                        st.info("Sample audio transcribed (note: this is synthetic audio)")
                        st.write(f"Transcription: {result.get('text', 'No transcription')}")
    
    # Information section
    st.markdown("---")
    st.header("About Nigerian Pidgin Transcription")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Common Pidgin Phrases:**
        - "How you dey?" - How are you?
        - "I dey fine" - I am fine
        - "Wetin dey happen?" - What's happening?
        - "Make we go" - Let's go
        - "Abeg help me" - Please help me
        """)
    
    with col2:
        st.markdown("""
        **Model Features:**
        - Fine-tuned Whisper base model
        - Specialized for Nigerian Pidgin
        - Maintains English capabilities
        - Real-time transcription ready
        - Post-processing normalization
        """)
    
    with col3:
        st.markdown("""
        **Technical Details:**
        - Base Model: OpenAI Whisper
        - Sample Rate: 16kHz
        - Max Duration: 30 seconds
        - Languages: English + Pidgin
        - Framework: PyTorch + Transformers
        """)

if __name__ == "__main__":
    main()