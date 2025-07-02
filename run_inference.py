#!/usr/bin/env python3
"""
Script to run inference with the trained Whisper Nigerian Pidgin model
"""

import argparse
import logging
from pathlib import Path
import json
import numpy as np

from src.inference_engine import WhisperPidginInference, PidginPostProcessor

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description="Run inference with Whisper Pidgin model")
    
    parser.add_argument(
        "--model-path",
        default="models/whisper-pidgin",
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--audio-file",
        help="Audio file to transcribe"
    )
    parser.add_argument(
        "--audio-dir",
        help="Directory containing audio files to transcribe"
    )
    parser.add_argument(
        "--output-file",
        help="Output file for transcription results"
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code"
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task to perform"
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include timestamps in output"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Apply Pidgin normalization"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Initialize inference engine
    logger.info(f"Loading model from: {args.model_path}")
    inference_engine = WhisperPidginInference(args.model_path)
    
    # Initialize post-processor
    post_processor = PidginPostProcessor() if args.normalize else None
    
    results = []
    
    if args.audio_file:
        # Single file transcription
        logger.info(f"Transcribing: {args.audio_file}")
        
        result = inference_engine.transcribe(
            args.audio_file,
            language=args.language,
            task=args.task,
            return_timestamps=args.timestamps
        )
        
        if post_processor:
            result["normalized_text"] = post_processor.normalize_pidgin(result["text"])
            result["text_with_punctuation"] = post_processor.add_punctuation(result["normalized_text"])
        
        results.append({
            "file": args.audio_file,
            **result
        })
        
        # Print result
        print(f"\nFile: {args.audio_file}")
        print(f"Transcription: {result['text']}")
        if post_processor:
            print(f"Normalized: {result['normalized_text']}")
            print(f"With punctuation: {result['text_with_punctuation']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        
        if args.timestamps and "timestamps" in result:
            print("\nTimestamps:")
            for ts in result["timestamps"][:10]:  # Show first 10
                print(f"  {ts['start']:.2f}s - {ts['end']:.2f}s: {ts['word']}")
    
    elif args.audio_dir:
        # Batch transcription
        audio_dir = Path(args.audio_dir)
        audio_files = []
        
        # Find audio files
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
            audio_files.extend(audio_dir.glob(ext))
        
        if not audio_files:
            logger.error(f"No audio files found in {audio_dir}")
            return
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Batch transcribe
        batch_results = inference_engine.batch_transcribe(
            [str(f) for f in audio_files],
            language=args.language,
            task=args.task,
            return_timestamps=args.timestamps
        )
        
        # Process results
        for result in batch_results:
            if "error" not in result and post_processor:
                result["normalized_text"] = post_processor.normalize_pidgin(result["text"])
                result["text_with_punctuation"] = post_processor.add_punctuation(result["normalized_text"])
            
            results.append(result)
            
            # Print result
            print(f"\nFile: {result['file']}")
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Transcription: {result['text']}")
                if post_processor:
                    print(f"Normalized: {result.get('normalized_text', '')}")
                print(f"Confidence: {result['confidence']:.2%}")
    
    else:
        # Interactive mode with sample phrases
        logger.info("Running in interactive mode with sample phrases")
        
        sample_phrases = [
            "How you dey?",
            "I dey fine o",
            "Wetin you wan chop?",
            "Make we go market",
            "Abeg help me",
            "Na so e be",
            "You sabi am well well"
        ]
        
        print("\nTesting with sample phrases (using synthetic audio):")
        print("=" * 50)
        
        for phrase in sample_phrases:
            # Create synthetic audio for demonstration
            dummy_audio = np.random.randn(16000 * 2)  # 2 seconds
            
            result = inference_engine.transcribe(dummy_audio)
            
            if post_processor:
                result["normalized_text"] = post_processor.normalize_pidgin(result["text"])
            
            results.append({
                "expected": phrase,
                "transcription": result["text"],
                "normalized": result.get("normalized_text", ""),
                "confidence": result["confidence"]
            })
            
            print(f"Expected: {phrase}")
            print(f"Got: {result['text']}")
            if post_processor:
                print(f"Normalized: {result.get('normalized_text', '')}")
            print(f"Confidence: {result['confidence']:.2%}")
            print("-" * 30)
    
    # Save results if output file specified
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_path}")
    
    logger.info("Inference completed!")

if __name__ == "__main__":
    main()