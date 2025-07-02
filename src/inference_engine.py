"""
Inference engine for the fine-tuned Whisper Pidgin model
"""

import torch
import torchaudio
import librosa
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from typing import Union, List, Dict, Optional
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperPidginInference:
    def __init__(
        self,
        model_path: str = "models/whisper-pidgin",
        device: str = "auto"
    ):
        self.model_path = Path(model_path)
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model and processor
        self.load_model()
        
        logger.info(f"Inference engine initialized on {self.device}")
    
    def load_model(self):
        """
        Load the fine-tuned model and processor
        """
        try:
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_path)
            self.processor = WhisperProcessor.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.warning(f"Could not load fine-tuned model: {e}")
            logger.info("Loading base Whisper model instead...")
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            self.model.to(self.device)
            self.model.eval()
    
    def preprocess_audio(self, audio_input: Union[str, np.ndarray]) -> torch.Tensor:
        """
        Preprocess audio for inference
        """
        if isinstance(audio_input, str):
            # Load from file
            audio, sr = librosa.load(audio_input, sr=16000)
        else:
            # Use provided array
            audio = audio_input
        
        # Ensure audio is the right length (max 30 seconds for Whisper)
        max_length = 16000 * 30  # 30 seconds at 16kHz
        if len(audio) > max_length:
            audio = audio[:max_length]
        
        return audio
    
    def transcribe(
        self,
        audio_input: Union[str, np.ndarray],
        language: str = "en",
        task: str = "transcribe",
        return_timestamps: bool = False,
        chunk_length: int = 30
    ) -> Dict[str, Union[str, List]]:
        """
        Transcribe audio to Nigerian Pidgin text
        """
        start_time = time.time()
        
        # Preprocess audio
        audio = self.preprocess_audio(audio_input)
        
        # Process with Whisper processor
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_features"],
                max_length=225,
                num_beams=5,
                temperature=0.0,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode transcription
        transcription = self.processor.batch_decode(
            generated_ids.sequences,
            skip_special_tokens=True
        )[0]
        
        # Calculate confidence score
        confidence = self.calculate_confidence(generated_ids.scores)
        
        processing_time = time.time() - start_time
        
        result = {
            "text": transcription,
            "confidence": confidence,
            "processing_time": processing_time,
            "language": language,
            "task": task
        }
        
        if return_timestamps:
            result["timestamps"] = self.extract_timestamps(generated_ids)
        
        return result
    
    def calculate_confidence(self, scores: List[torch.Tensor]) -> float:
        """
        Calculate confidence score from generation scores
        """
        if not scores:
            return 0.0
        
        # Average probability across all tokens
        probs = []
        for score in scores:
            prob = torch.softmax(score, dim=-1)
            max_prob = torch.max(prob, dim=-1)[0]
            probs.extend(max_prob.cpu().numpy())
        
        return float(np.mean(probs)) if probs else 0.0
    
    def extract_timestamps(self, generated_ids) -> List[Dict]:
        """
        Extract word-level timestamps (simplified implementation)
        """
        # This is a simplified implementation
        # In practice, you'd need more sophisticated alignment
        tokens = generated_ids.sequences[0]
        timestamps = []
        
        for i, token_id in enumerate(tokens):
            if token_id != self.processor.tokenizer.pad_token_id:
                token = self.processor.tokenizer.decode([token_id])
                timestamps.append({
                    "word": token,
                    "start": i * 0.02,  # Rough estimate
                    "end": (i + 1) * 0.02
                })
        
        return timestamps
    
    def batch_transcribe(
        self,
        audio_files: List[str],
        **kwargs
    ) -> List[Dict]:
        """
        Transcribe multiple audio files
        """
        results = []
        
        for audio_file in audio_files:
            try:
                result = self.transcribe(audio_file, **kwargs)
                result["file"] = audio_file
                results.append(result)
                logger.info(f"Transcribed: {audio_file}")
            except Exception as e:
                logger.error(f"Error transcribing {audio_file}: {e}")
                results.append({
                    "file": audio_file,
                    "error": str(e),
                    "text": "",
                    "confidence": 0.0
                })
        
        return results
    
    def real_time_transcribe(self, audio_stream, chunk_duration: float = 1.0):
        """
        Real-time transcription from audio stream
        """
        # This would be implemented for real-time applications
        # Placeholder for streaming transcription
        pass

class PidginPostProcessor:
    """
    Post-processing for Nigerian Pidgin transcriptions
    """
    
    def __init__(self):
        # Common corrections and normalizations
        self.corrections = {
            "dey": "dey",
            "de": "dey",
            "they": "dey",
            "wetin": "wetin",
            "what": "wetin",
            "abeg": "abeg",
            "please": "abeg",
            "sabi": "sabi",
            "know": "sabi",
        }
        
        # Pidgin grammar patterns
        self.patterns = [
            (r'\bI dey\b', 'I dey'),
            (r'\bYou dey\b', 'You dey'),
            (r'\bWe dey\b', 'We dey'),
        ]
    
    def normalize_pidgin(self, text: str) -> str:
        """
        Normalize transcribed text to standard Pidgin forms
        """
        normalized = text.lower()
        
        # Apply corrections
        for wrong, correct in self.corrections.items():
            normalized = normalized.replace(wrong, correct)
        
        # Apply pattern corrections
        import re
        for pattern, replacement in self.patterns:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        return normalized
    
    def add_punctuation(self, text: str) -> str:
        """
        Add basic punctuation to transcribed text
        """
        # Simple punctuation rules
        text = text.strip()
        if text and not text.endswith(('.', '!', '?')):
            # Add period if it seems like a statement
            if any(word in text.lower() for word in ['dey', 'be', 'na']):
                text += '.'
            else:
                text += '.'
        
        return text

if __name__ == "__main__":
    # Test inference engine
    inference = WhisperPidginInference()
    
    # Create a dummy audio array for testing
    dummy_audio = np.random.randn(16000 * 2)  # 2 seconds of audio
    
    result = inference.transcribe(dummy_audio)
    print(f"Transcription: {result['text']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Processing time: {result['processing_time']:.2f}s")
    
    # Test post-processor
    post_processor = PidginPostProcessor()
    normalized = post_processor.normalize_pidgin("I de fine today")
    print(f"Normalized: {normalized}")