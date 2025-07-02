#!/usr/bin/env python3
"""
Complete setup and run script for Whisper Nigerian Pidgin Training System
This script handles everything from installation to running the training pipeline
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WhisperPidginSetup:
    def __init__(self):
        self.project_root = Path.cwd()
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"

    def get_python_command(self):
        """Get the python command for the system"""
        # This assumes python is available in the system's PATH
        return sys.executable # Use the current Python executable

    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.models_dir,
            Path("logs"),
            Path("outputs")
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")

    def create_config_file(self):
        """Create default configuration file"""
        config = {
            "model": {
                "base_model": "openai/whisper-medium",
                "output_dir": "models/whisper-pidgin",
                "language": "en"
            },
            "training": {
                "num_train_epochs": 10,
                "per_device_train_batch_size": 4,
                "per_device_eval_batch_size": 4,
                "learning_rate": 1e-5,
                "warmup_steps": 500,
                "logging_steps": 25,
                "save_steps": 1000,
                "eval_steps": 1000
            },
            "data": {
                "data_dir": "data",
                "use_sample_data": True,
                "augment_data": True
            }
        }

        config_path = self.project_root / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Configuration file created: {config_path}")

    def setup_complete(self):
        """Complete setup process (directory and config creation)"""
        logger.info("Starting Whisper Pidgin setup...")

        self.create_directories()
        self.create_config_file()

        logger.info("Setup completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Ensure all necessary Python dependencies are installed manually (e.g., pip install -r requirements.txt).")
        logger.info("2. Run training: python setup_and_run.py --train")
        logger.info("3. Run inference: python setup_and_run.py --inference")

        return True

class WhisperPidginRunner:
    def __init__(self, setup_instance):
        self.setup = setup_instance
        self.python_cmd = setup_instance.get_python_command()

    def collect_youtube_data(self, urls_file=None):
        """Collect data from YouTube URLs"""
        logger.info("Starting YouTube data collection...")

        if urls_file and Path(urls_file).exists():
            cmd = [self.python_cmd, "src/data_collector.py", "--urls-file", urls_file]
        else:
            # Use sample data
            cmd = [self.python_cmd, "src/data_collector.py", "--sample"]

        try:
            subprocess.run(cmd, check=True)
            logger.info("Data collection completed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Data collection failed: {e}")
            return False

    def run_training(self, config_file="config.json"):
        """Run the training pipeline"""
        logger.info("Starting training pipeline...")

        cmd = [
            self.python_cmd, "run_training.py",
            "--config-file", config_file
        ]

        try:
            subprocess.run(cmd, check=True)
            logger.info("Training completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            return False

    def run_inference(self, audio_file=None, model_path=None):
        """Run inference with the trained model"""
        logger.info("Starting inference...")

        cmd = [self.python_cmd, "run_inference.py"]

        if model_path:
            cmd.extend(["--model-path", model_path])

        if audio_file:
            cmd.extend(["--audio-file", audio_file])

        try:
            subprocess.run(cmd, check=True)
            logger.info("Inference completed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Inference failed: {e}")
            return False

    def run_demo_app(self):
        """Run the Streamlit demo application"""
        logger.info("Starting demo application...")

        try:
            # Streamlit is now assumed to be pre-installed
            # subprocess.run([self.setup.get_pip_command(), "install", "streamlit"], check=True) # Removed

            # Run streamlit app
            cmd = [self.python_cmd, "-m", "streamlit", "run", "src/demo_app.py"]
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Demo app failed: {e}")
            return False

    def quick_start(self):
        """Quick start with sample data"""
        logger.info("Running quick start with sample data...")

        # Step 1: Collect sample data
        if not self.collect_youtube_data():
            return False

        # Step 2: Run training with quick config
        quick_config = {
            "model": {
                "base_model": "openai/whisper-small",
                "output_dir": "models/whisper-pidgin-quick"
            },
            "training": {
                "num_train_epochs": 3,
                "per_device_train_batch_size": 2,
                "learning_rate": 5e-5,
                "warmup_steps": 50
            },
            "data": {
                "use_sample_data": True,
                "augment_data": True
            }
        }

        quick_config_path = "quick_config.json"
        with open(quick_config_path, 'w') as f:
            json.dump(quick_config, f, indent=2)

        if not self.run_training(quick_config_path):
            return False

        # Step 3: Test inference
        if not self.run_inference(model_path="models/whisper-pidgin-quick"):
            return False

        logger.info("Quick start completed successfully!")
        return True

def main():
    parser = argparse.ArgumentParser(description="Whisper Nigerian Pidgin Setup and Runner")

    parser.add_argument("--setup", action="store_true", help="Run initial setup (create directories and config file)")
    parser.add_argument("--train", action="store_true", help="Run training pipeline")
    parser.add_argument("--inference", action="store_true", help="Run inference")
    parser.add_argument("--demo", action="store_true", help="Run demo application")
    parser.add_argument("--quick-start", action="store_true", help="Quick start with sample data, training, and inference")
    parser.add_argument("--collect-data", action="store_true", help="Collect data from YouTube")

    parser.add_argument("--config", default="config.json", help="Configuration file")
    parser.add_argument("--audio-file", help="Audio file for inference")
    parser.add_argument("--model-path", help="Path to trained model")
    parser.add_argument("--urls-file", help="File containing YouTube URLs")

    args = parser.parse_args()

    # Initialize setup
    setup = WhisperPidginSetup()
    runner = WhisperPidginRunner(setup)

    # The --setup flag will now only create directories and config, assuming dependencies are met.
    if args.setup:
        if not setup.setup_complete():
            sys.exit(1)

    if args.quick_start:
        if not runner.quick_start():
            sys.exit(1)
    elif args.collect_data:
        if not runner.collect_youtube_data(args.urls_file):
            sys.exit(1)
    elif args.train:
        if not runner.run_training(args.config):
            sys.exit(1)
    elif args.inference:
        if not runner.run_inference(args.audio_file, args.model_path):
            sys.exit(1)
    elif args.demo:
        runner.run_demo_app()
    else:
        # Default: show help and run quick start if no specific command is given
        parser.print_help()
        print("\n" + "="*50)
        print("QUICK START GUIDE")
        print("="*50)
        print("1. Run initial setup (creates directories and config files):")
        print("    python setup_and_run.py --setup")
        print("\n2. Quick training with sample data:")
        print("    python setup_and_run.py --quick-start")
        print("\n3. Run demo application:")
        print("    python setup_and_run.py --demo")
        print("\n4. Custom training:")
        print("    python setup_and_run.py --train --config your_config.json")
        print("\n5. Test inference:")
        print("    python setup_and_run.py --inference --audio-file your_audio.wav")

if __name__ == "__main__":
    main()
