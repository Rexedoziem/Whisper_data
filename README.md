# Whisper Nigerian Pidgin Fine-tuning System

A comprehensive Python-based system for fine-tuning OpenAI's Whisper model to transcribe Nigerian Pidgin English (Naija). This system preserves Whisper's original multilingual capabilities while adding specialized support for Nigerian Pidgin.

## Features

- **Complete Training Pipeline**: End-to-end system from data collection to model deployment
- **Data Augmentation**: Specialized augmentation techniques for Nigerian Pidgin
- **Fine-tuning**: Preserves original Whisper knowledge while adding Pidgin capabilities
- **Post-processing**: Normalization and correction of Pidgin transcriptions
- **Inference Engine**: Fast, production-ready transcription with confidence scoring
- **Demo Application**: Streamlit-based web interface for testing
- **Batch Processing**: Support for transcribing multiple audio files

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Training Pipeline

```bash
# Quick training with sample data
python run_training.py --epochs 5 --batch-size 2 --use-sample-data

# Full training with custom data
python run_training.py --epochs 20 --batch-size 8 --data-dir /path/to/your/data
```

### 3. Run Inference

```bash
# Transcribe a single audio file
python run_inference.py --audio-file audio.wav --model-path models/whisper-pidgin

# Batch transcribe multiple files
python run_inference.py --audio-dir /path/to/audio/files --output-file results.json

# Interactive testing
python run_inference.py
```

### 4. Launch Demo App

```bash
streamlit run src/demo_app.py
```

## System Architecture

### Core Components

1. **Data Collector** (`src/data_collector.py`)
   - Collects audio-text pairs from various sources
   - Supports YouTube video extraction
   - Creates training manifests

2. **Data Preprocessor** (`src/data_preprocessor.py`)
   - Audio preprocessing and normalization
   - Dataset preparation for Whisper
   - Data augmentation techniques

3. **Trainer** (`src/whisper_trainer.py`)
   - Fine-tuning pipeline with LoRA support
   - Specialized for Nigerian Pidgin
   - Preserves original Whisper capabilities

4. **Inference Engine** (`src/inference_engine.py`)
   - Fast transcription with confidence scoring
   - Batch processing support
   - Real-time transcription ready

5. **Training Pipeline** (`src/training_pipeline.py`)
   - Orchestrates the complete training process
   - Handles evaluation and testing
   - Saves training artifacts

## Nigerian Pidgin Support

### Common Phrases Supported

- "How you dey?" → How are you?
- "I dey fine o" → I am fine
- "Wetin dey happen?" → What's happening?
- "Make we go" → Let's go
- "Abeg help me" → Please help me
- "Na so e be" → That's how it is
- "You sabi am well well" → You know it very well

### Vocabulary Extensions

The system adds common Pidgin words to the tokenizer:
- dey, wetin, abeg, wahala, oya, sabi
- chop, waka, palava, gbege, katakata
- shakara, ginger, pepper, scatter

### Post-processing Features

- **Normalization**: Standardizes Pidgin spellings
- **Punctuation**: Adds appropriate punctuation
- **Code-switching**: Handles English-Pidgin mixing

## Training Configuration

### Default Settings

```python
training_config = {
    "num_train_epochs": 10,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "learning_rate": 1e-5,
    "warmup_steps": 500,
    "logging_steps": 25,
    "save_steps": 1000,
    "eval_steps": 1000
}
```

### Custom Configuration

Create a JSON config file:

```json
{
    "num_train_epochs": 20,
    "per_device_train_batch_size": 8,
    "learning_rate": 5e-6,
    "warmup_steps": 1000,
    "gradient_accumulation_steps": 2
}
```

Use with:
```bash
python run_training.py --config-file config.json
```

## Data Requirements

### Training Data Format

The system expects audio-text pairs in JSONL format:

```json
{"audio_filepath": "path/to/audio.wav", "text": "How you dey?", "duration": 2.0}
{"audio_filepath": "path/to/audio2.wav", "text": "I dey fine o", "duration": 1.5}
```

### Data Collection

1. **YouTube Videos**: Extract audio from Pidgin content
2. **Recorded Conversations**: Natural Pidgin speech
3. **Radio/TV Shows**: Nigerian media content
4. **Synthetic Data**: Generated using TTS systems

### Recommended Dataset Size

- **Minimum**: 10 hours of transcribed audio
- **Good**: 50-100 hours
- **Excellent**: 200+ hours

## Model Performance

### Evaluation Metrics

- **WER (Word Error Rate)**: Primary metric for transcription accuracy
- **Confidence Score**: Model's certainty in predictions
- **Processing Time**: Speed of transcription

### Expected Performance

With sufficient training data:
- **WER**: 15-25% on Nigerian Pidgin
- **English WER**: Maintains original Whisper performance
- **Processing**: Real-time capable (1x speed or faster)

## Advanced Usage

### Custom Model Training

```python
from src.training_pipeline import WhisperPidginPipeline

pipeline = WhisperPidginPipeline(
    base_model="openai/whisper-medium",
    output_dir="models/custom-pidgin"
)

results = pipeline.run_full_pipeline(
    use_sample_data=False,
    augment_data=True,
    training_config={
        "num_train_epochs": 15,
        "learning_rate": 2e-5
    }
)
```

### Inference Integration

```python
from src.inference_engine import WhisperPidginInference

# Initialize
inference = WhisperPidginInference("models/whisper-pidgin")

# Transcribe
result = inference.transcribe("audio.wav")
print(f"Transcription: {result['text']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Batch Processing

```python
# Process multiple files
audio_files = ["file1.wav", "file2.wav", "file3.wav"]
results = inference.batch_transcribe(audio_files)

for result in results:
    print(f"{result['file']}: {result['text']}")
```

## Technical Details

### Model Architecture

- **Base**: OpenAI Whisper (small/medium/large)
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Vocabulary**: Extended with Pidgin tokens
- **Languages**: English + Nigerian Pidgin

### System Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ for models and data
- **Python**: 3.8+

### Dependencies

- PyTorch 2.0+
- Transformers 4.35+
- Datasets 2.14+
- Librosa 0.10+
- Streamlit (for demo app)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch-size 2`
   - Use gradient accumulation
   - Try smaller model: `--base-model openai/whisper-small`

2. **Poor Transcription Quality**
   - Increase training epochs
   - Add more training data
   - Improve audio quality
   - Check data augmentation settings

3. **Slow Training**
   - Use GPU acceleration
   - Increase batch size if memory allows
   - Use mixed precision training (fp16)

### Performance Optimization

```python
# Training optimizations
training_config = {
    "fp16": True,  # Mixed precision
    "dataloader_num_workers": 4,  # Parallel data loading
    "gradient_accumulation_steps": 2,  # Effective larger batch size
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd whisper-pidgin

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
python -m pytest tests/
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- OpenAI for the Whisper model
- Hugging Face for the Transformers library
- Nigerian Pidgin language community
- Contributors and testers

## Citation

If you use this system in your research, please cite:

```bibtex
@software{whisper_pidgin_2024,
  title={Whisper Nigerian Pidgin Fine-tuning System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/whisper-pidgin}
}
```

## Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Review the troubleshooting section

---

**Note**: This system is designed for research and educational purposes. For production use, ensure you have appropriate rights to your training data and comply with relevant regulations.