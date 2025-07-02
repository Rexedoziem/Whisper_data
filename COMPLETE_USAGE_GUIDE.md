# Complete Usage Guide: Whisper Nigerian Pidgin Training System

## 🚀 Quick Start (Recommended)

### Option 1: Automatic Setup and Training
```bash
# Clone or download the project
# Navigate to project directory

# Run complete setup and quick training
python setup_and_run.py --quick-start
```

This will:
1. Set up virtual environment
2. Install all dependencies
3. Create sample data
4. Train a basic model
5. Test the model

### Option 2: Step-by-Step Setup

#### Step 1: Initial Setup
```bash
python setup_and_run.py --setup
```

#### Step 2: Activate Virtual Environment
```bash
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

#### Step 3: Choose Your Data Collection Method

**Method A: Use Sample Data (Fastest)**
```bash
python setup_and_run.py --collect-data
```

**Method B: Collect from YouTube**
```bash
# Edit youtube_urls_example.txt with real URLs
python src/enhanced_data_collector.py --urls-file youtube_urls_example.txt
```

**Method C: Single YouTube Video**
```bash
python src/enhanced_data_collector.py --url "https://youtube.com/watch?v=YOUR_VIDEO_ID"
```

#### Step 4: Train the Model
```bash
python setup_and_run.py --train
```

#### Step 5: Test Your Model
```bash
python setup_and_run.py --inference
```

#### Step 6: Run Demo App
```bash
python setup_and_run.py --demo
```

## 📊 Understanding the Data Collection Process

### Automatic vs Manual Transcription

The system handles both scenarios intelligently:

#### Automatic Extraction (When Available)
- ✅ Videos with existing subtitles/captions
- ✅ Auto-generated YouTube captions
- ✅ Manual subtitles added by creators
- ⚡ Fast processing, ready for training

#### Manual Transcription (When Needed)
- 📝 Videos without subtitles
- 🎯 Higher quality, human-verified transcriptions
- 🔄 Iterative improvement process
- 👥 Can involve multiple transcribers

### The Bootstrap Strategy

1. **Phase 1: Foundation** (10-20 hours)
   - Manually transcribe diverse Pidgin content
   - Focus on clear, representative speech
   - Include various contexts and speakers

2. **Phase 2: Initial Model** 
   - Train first model on foundation data
   - Achieve basic Pidgin recognition

3. **Phase 3: Assisted Transcription**
   - Use model to help transcribe more content
   - Human verification and correction
   - Faster than pure manual transcription

4. **Phase 4: Iterative Improvement**
   - Retrain with expanded dataset
   - Better accuracy enables more automation
   - Continuous improvement cycle

## 🛠 Advanced Usage

### Custom Training Configuration

Create a custom config file:
```json
{
  "model": {
    "base_model": "openai/whisper-medium",
    "output_dir": "models/my-pidgin-model"
  },
  "training": {
    "num_train_epochs": 15,
    "per_device_train_batch_size": 8,
    "learning_rate": 2e-5,
    "warmup_steps": 1000
  },
  "data": {
    "use_sample_data": false,
    "augment_data": true
  }
}
```

Train with custom config:
```bash
python setup_and_run.py --train --config my_config.json
```

### Batch YouTube Processing

Create a file with URLs:
```bash
# youtube_urls.txt
https://youtube.com/watch?v=video1
https://youtube.com/watch?v=video2
https://youtube.com/watch?v=video3
```

Process all URLs:
```bash
python src/enhanced_data_collector.py --urls-file youtube_urls.txt
```

### Manual Transcription Workflow

1. **Generate transcription queue:**
```bash
python src/enhanced_data_collector.py --urls-file youtube_urls.txt
```

2. **Check manual transcription queue:**
```bash
# File created: data/raw/manual_transcription_queue.json
```

3. **Use the web interface for transcription:**
```bash
npm run dev  # Start the React app
# Navigate to the Transcription tab
```

4. **Or transcribe programmatically:**
```python
import json

# Load queue
with open('data/raw/manual_transcription_queue.json', 'r') as f:
    queue = json.load(f)

# Add transcriptions
for item in queue:
    # Listen to item['audio_path']
    # Add transcription to item['transcript']
    item['transcript'] = "Your Pidgin transcription here"
    item['status'] = 'transcribed'

# Save updated queue
with open('data/raw/manual_transcription_queue.json', 'w') as f:
    json.dump(queue, f, indent=2)
```

### Testing and Evaluation

**Test with specific audio file:**
```bash
python setup_and_run.py --inference --audio-file path/to/your/audio.wav
```

**Batch testing:**
```bash
python run_inference.py --audio-dir path/to/audio/folder --output-file results.json
```

**Interactive testing:**
```bash
python run_inference.py  # No arguments for interactive mode
```

## 📁 Project Structure

```
whisper-pidgin/
├── src/
│   ├── components/           # React components
│   ├── data_collector.py     # Basic data collection
│   ├── enhanced_data_collector.py  # Advanced data collection
│   ├── data_preprocessor.py  # Data preprocessing
│   ├── whisper_trainer.py    # Model training
│   ├── training_pipeline.py  # Complete pipeline
│   ├── inference_engine.py   # Model inference
│   └── demo_app.py          # Streamlit demo
├── data/
│   ├── raw/                 # Raw collected data
│   ├── processed/           # Processed training data
│   └── metadata/            # Data metadata
├── models/                  # Trained models
├── logs/                    # Training logs
├── config.json             # Default configuration
├── setup_and_run.py        # Main setup and run script
├── run_training.py         # Training script
├── run_inference.py        # Inference script
└── requirements.txt        # Python dependencies
```

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in config
"per_device_train_batch_size": 2
```

**2. YouTube Download Errors**
```bash
# Update yt-dlp
pip install --upgrade yt-dlp
```

**3. Audio Processing Issues**
```bash
# Install FFmpeg
# Windows: Download from https://ffmpeg.org/
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg
```

**4. Virtual Environment Issues**
```bash
# Remove and recreate
rm -rf venv
python setup_and_run.py --setup
```

### Performance Optimization

**For faster training:**
- Use GPU if available
- Increase batch size (if memory allows)
- Use smaller base model (whisper-small vs whisper-large)
- Enable mixed precision training

**For better accuracy:**
- More diverse training data
- Longer training (more epochs)
- Data augmentation
- Manual verification of transcriptions

## 📈 Monitoring Progress

### Training Logs
```bash
# View training logs
tail -f training.log

# View tensorboard (if installed)
tensorboard --logdir models/your-model/runs
```

### Data Collection Reports
```bash
# Check collection progress
cat data/raw/collection_report.json
```

### Model Evaluation
```bash
# Check evaluation results
cat models/your-model/evaluation_results.json
```

## 🎯 Best Practices

### Data Collection
1. **Quality over Quantity**: Better to have 10 hours of high-quality transcriptions than 50 hours of poor quality
2. **Diversity**: Include different speakers, contexts, and Pidgin variations
3. **Clear Audio**: Prioritize content with clear speech and minimal background noise
4. **Balanced Content**: Mix formal and informal speech, different topics

### Training
1. **Start Small**: Begin with whisper-small model for faster iteration
2. **Monitor Overfitting**: Use validation set to check model performance
3. **Save Checkpoints**: Regular saving prevents loss of progress
4. **Experiment**: Try different learning rates and batch sizes

### Evaluation
1. **Human Evaluation**: Always have native speakers evaluate the output
2. **Context Testing**: Test with various types of Pidgin content
3. **Error Analysis**: Understand what types of errors the model makes
4. **Iterative Improvement**: Use evaluation results to improve training data

## 🚀 Production Deployment

### Model Export
```bash
# Export trained model
python -c "
from src.inference_engine import WhisperPidginInference
inference = WhisperPidginInference('models/your-model')
# Model is ready for production use
"
```

### API Integration
```python
# Example API usage
from src.inference_engine import WhisperPidginInference

# Initialize once
model = WhisperPidginInference('models/your-model')

# Use for multiple requests
def transcribe_audio(audio_file):
    result = model.transcribe(audio_file)
    return {
        'text': result['text'],
        'confidence': result['confidence']
    }
```

This guide covers everything from basic setup to advanced usage. The system is designed to be flexible and handle both automatic and manual transcription workflows effectively.