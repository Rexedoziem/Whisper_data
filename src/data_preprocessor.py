"""
Data preprocessing for Whisper fine-tuning
Handles audio processing and dataset preparation
"""

import torch
import torchaudio
import librosa
import numpy as np
from datasets import Dataset, Audio
from transformers import WhisperProcessor
from typing import List, Dict, Any
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperDataPreprocessor:
    def __init__(self, model_name: str = "openai/whisper-small"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.sampling_rate = 16000
        
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio file
        """
        try:
            # Load audio with librosa
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
            return audio
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            return np.array([])
    
    def prepare_dataset(self, manifest_path: str) -> Dataset:
        """
        Prepare dataset from manifest file
        """
        data = []
        
        # For demonstration, we'll create synthetic audio data
        # In practice, you'd load real audio files
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                
                # Create synthetic audio (replace with real audio loading)
                duration = max(len(item['text']) * 0.1, 1.0)  # Rough estimate
                synthetic_audio = np.random.randn(int(duration * self.sampling_rate)) * 0.1
                
                data.append({
                    'audio': {
                        'array': synthetic_audio,
                        'sampling_rate': self.sampling_rate
                    },
                    'text': item['text']
                })
        
        # Create HuggingFace dataset
        dataset = Dataset.from_list(data)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        
        logger.info(f"Created dataset with {len(dataset)} samples")
        return dataset
    
    def preprocess_function(self, examples):
        """
        Preprocessing function for the dataset
        """
        # Extract audio arrays
        audio_arrays = [x["array"] for x in examples["audio"]]
        
        # Process audio inputs
        inputs = self.processor(
            audio_arrays,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Process text labels
        labels = self.processor.tokenizer(
            examples["text"],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Replace padding token id's of the labels by -100
        labels["input_ids"] = labels["input_ids"].masked_fill(
            labels["attention_mask"] == 0, -100
        )
        
        inputs["labels"] = labels["input_ids"]
        
        return inputs
    
    def create_data_collator(self):
        """
        Create data collator for training
        """
        from transformers import DataCollatorForSeq2Seq
        
        return DataCollatorForSeq2Seq(
            tokenizer=self.processor.tokenizer,
            model=None,
            padding=True,
            return_tensors="pt"
        )

class PidginAugmentor:
    """
    Data augmentation techniques for Nigerian Pidgin
    """
    
    def __init__(self):
        # Common Pidgin variations and synonyms
        self.pidgin_variations = {
            'dey': ['de', 'they'],
            'wetin': ['what', 'whatin'],
            'make': ['let', 'mek'],
            'chop': ['eat', 'food'],
            'sabi': ['know', 'understand'],
            'wahala': ['problem', 'trouble'],
            'oya': ['come on', 'let\'s go'],
            'abeg': ['please', 'beg'],
        }
    
    def augment_text(self, text: str, num_variations: int = 3) -> List[str]:
        """
        Create text variations using Pidgin synonyms
        """
        variations = [text]
        
        for _ in range(num_variations):
            augmented = text
            for pidgin_word, alternatives in self.pidgin_variations.items():
                if pidgin_word in augmented.lower():
                    alternative = np.random.choice(alternatives)
                    augmented = augmented.replace(pidgin_word, alternative)
            
            if augmented != text and augmented not in variations:
                variations.append(augmented)
        
        return variations
    
    def add_noise_to_audio(self, audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """
        Add noise to audio for augmentation
        """
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise
    
    def speed_change(self, audio: np.ndarray, speed_factor: float = 1.1) -> np.ndarray:
        """
        Change audio speed for augmentation
        """
        return librosa.effects.time_stretch(audio, rate=speed_factor)

if __name__ == "__main__":
    # Test preprocessing
    preprocessor = WhisperDataPreprocessor()
    
    # Create sample manifest for testing
    sample_manifest = "data/raw/training_manifest.jsonl"
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    with open(sample_manifest, 'w', encoding='utf-8') as f:
        f.write('{"audio_filepath": "", "text": "How you dey?", "duration": 2.0}\n')
        f.write('{"audio_filepath": "", "text": "I dey fine o", "duration": 1.5}\n')
    
    dataset = preprocessor.prepare_dataset(sample_manifest)
    print(f"Dataset created with {len(dataset)} samples")
    
    # Test augmentation
    augmentor = PidginAugmentor()
    variations = augmentor.augment_text("How you dey today?")
    print(f"Text variations: {variations}")