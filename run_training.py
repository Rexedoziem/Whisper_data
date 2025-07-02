#!/usr/bin/env python3
"""
Main script to run the Whisper Nigerian Pidgin training pipeline
"""

import argparse
import logging
from pathlib import Path
import json

from src.training_pipeline import WhisperPidginPipeline

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Train Whisper model for Nigerian Pidgin")
    
    # Model arguments
    parser.add_argument(
        "--base-model",
        default="openai/whisper-medium",
        help="Base Whisper model to fine-tune"
    )
    parser.add_argument(
        "--output-dir",
        default="models/whisper-pidgin",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps"
    )
    
    # Data arguments
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        default=True,
        help="Use sample data for training"
    )
    parser.add_argument(
        "--augment-data",
        action="store_true",
        default=True,
        help="Apply data augmentation"
    )
    
    # Other arguments
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--config-file",
        help="JSON config file with training parameters"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load config from file if provided
    training_config = {}
    if args.config_file and Path(args.config_file).exists():
        with open(args.config_file, 'r') as f:
            training_config = json.load(f)
        logger.info(f"Loaded config from {args.config_file}")
    
    # Override with command line arguments
    training_config.update({
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "logging_steps": max(10, args.warmup_steps // 10),
        "save_steps": max(100, args.epochs * 50),
        "eval_steps": max(100, args.epochs * 50)
    })
    
    logger.info("Starting Whisper Nigerian Pidgin training pipeline")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Training config: {training_config}")
    
    try:
        # Initialize pipeline
        pipeline = WhisperPidginPipeline(
            base_model=args.base_model,
            output_dir=args.output_dir,
            data_dir=args.data_dir
        )
        
        # Run training
        results = pipeline.run_full_pipeline(
            use_sample_data=args.use_sample_data,
            augment_data=args.augment_data,
            training_config=training_config
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {results['model_path']}")
        logger.info(f"Final evaluation WER: {results['evaluation_results'].get('eval_wer', 'N/A')}")
        
        # Save final results
        results_file = Path(args.output_dir) / "final_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()