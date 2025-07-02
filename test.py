import whisper

# Load the Whisper model
model = whisper.load_model("medium")  # or "large" for better accuracy

# Path to your audio file
audio_path = "C:/Users/User/Desktop/finetune_whisper/data/raw/audio/No Be Me.wav"

# Transcribe the audio
result = model.transcribe(audio_path)

# Print the transcription
print("Original transcription:")
print(result["text"])
