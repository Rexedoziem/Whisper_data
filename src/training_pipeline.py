"""
Complete training pipeline for Whisper Nigerian Pidgin fine-tuning
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

from src.enhanced_data_collector import EnhancedPidginDataCollector
from src.data_preprocessor import WhisperDataPreprocessor, PidginAugmentor
from src.whisper_trainer import PidginSpecificTrainer
from src.inference_engine import WhisperPidginInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperPidginPipeline:
    """
    Complete pipeline for training Whisper on Nigerian Pidgin
    """
    
    def __init__(
        self,
        base_model: str = "openai/whisper-medium",
        output_dir: str = "models/whisper-pidgin-final",
        data_dir: str = "data"
    ):
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_collector = EnhancedPidginDataCollector(str(self.data_dir / "raw"))
        self.preprocessor = WhisperDataPreprocessor(self.base_model)
        self.augmentor = PidginAugmentor()
        self.trainer = PidginSpecificTrainer(
            model_name=self.base_model,
            output_dir=str(self.output_dir)
        )
        
        logger.info(f"Pipeline initialized with base model: {self.base_model}")
    
    def collect_data(self, use_sample: bool = True) -> str:
        """
        Step 1: Collect training data
        """
        logger.info("Step 1: Collecting training data...")
        
        if use_sample:
            # Use sample data for demonstration
            sample_data = self.data_collector.create_sample_dataset()
            manifest_path = self.data_collector.prepare_training_manifest(sample_data)
        else:
            # In practice, you would collect real audio data here
            # Example: YouTube videos, recorded conversations, etc.
            video_urls = []  # Add your video URLs
            transcripts = []  # Add corresponding transcripts
            
            collected_data = self.data_collector.collect_youtube_data(video_urls, transcripts)
            manifest_path = self.data_collector.prepare_training_manifest(collected_data)
        
        logger.info(f"Data collection completed. Manifest: {manifest_path}")
        return manifest_path
    
    def preprocess_data(self, manifest_path: str, augment: bool = True):
        """
        Step 2: Preprocess and augment data
        """
        logger.info("Step 2: Preprocessing data...")
        
        # Load dataset
        dataset = self.preprocessor.prepare_dataset(manifest_path)
        
        # Apply augmentation if requested
        if augment:
            logger.info("Applying data augmentation...")
            augmented_data = []
            
            for item in dataset:
                # Original item
                augmented_data.append(item)
                
                # Text variations
                text_variations = self.augmentor.augment_text(item['text'], num_variations=2)
                for variation in text_variations[1:]:  # Skip original
                    augmented_item = item.copy()
                    augmented_item['text'] = variation
                    augmented_data.append(augmented_item)
            
            # Create augmented dataset
            from datasets import Dataset
            dataset = Dataset.from_list(augmented_data)
            dataset = dataset.cast_column("audio", dataset.features["audio"])
        
        # Split dataset
        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']
        
        # Apply preprocessing
        train_dataset = train_dataset.map(
            self.preprocessor.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        eval_dataset = eval_dataset.map(
            self.preprocessor.preprocess_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        
        logger.info(f"Preprocessing completed. Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
        return train_dataset, eval_dataset
    
    def train_model(
        self,
        train_dataset,
        eval_dataset,
        training_config: Optional[Dict[str, Any]] = None
    ):
        """
        Step 3: Fine-tune the model
        """
        logger.info("Step 3: Training model...")
        
        # Default training configuration
        default_config = {
            "num_train_epochs": 10,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "learning_rate": 1e-5,
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500
        }
        
        if training_config:
            default_config.update(training_config)
        
        # Create data collator
        data_collator = self.preprocessor.create_data_collator()
        
        # Save training configuration
        self.trainer.save_training_config(default_config)
        
        # Start training
        trainer = self.trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            **default_config
        )
        
        logger.info("Training completed!")
        return trainer
    
    def evaluate_model(self, eval_dataset, data_collator):
        """
        Step 4: Evaluate the trained model
        """
        logger.info("Step 4: Evaluating model...")
        
        results = self.trainer.evaluate_model(eval_dataset, data_collator)
        
        # Save evaluation results
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {results_path}")
        return results
    
    def test_inference(self, test_phrases: Optional[list] = None):
        """
        Step 5: Test inference with the trained model
        """
        logger.info("Step 5: Testing inference...")
        
        # Initialize inference engine
        inference = WhisperPidginInference(model_path=str(self.output_dir))
        
        # Default test phrases
        if test_phrases is None:
            test_phrases = [
                "How you dey?",
                "I dey fine o",
                "Wetin you wan chop?",
                "Make we go market"
            ]
        
        # Test with synthetic audio (in practice, use real audio)
        import numpy as np
        
        results = []
        for phrase in test_phrases:
            # Create dummy audio for testing
            dummy_audio = np.random.randn(16000 * 2)  # 2 seconds
            
            result = inference.transcribe(dummy_audio)
            results.append({
                "expected": phrase,
                "transcribed": result["text"],
                "confidence": result["confidence"]
            })
            
            logger.info(f"Expected: {phrase}")
            logger.info(f"Transcribed: {result['text']}")
            logger.info(f"Confidence: {result['confidence']:.2f}")
            logger.info("---")
        
        # Save test results
        test_results_path = self.output_dir / "inference_test_results.json"
        with open(test_results_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Inference testing completed. Results saved to {test_results_path}")
        return results
    
    def run_full_pipeline(
        self,
        use_sample_data: bool = True,
        augment_data: bool = True,
        training_config: Optional[Dict[str, Any]] = None
    ):
        """
        Run the complete training pipeline
        """
        logger.info("Starting full Whisper Pidgin training pipeline...")
        
        try:
            # Step 1: Collect data
            manifest_path = self.collect_data(use_sample=use_sample_data)
            
            # Step 2: Preprocess data
            train_dataset, eval_dataset = self.preprocess_data(manifest_path, augment=augment_data)
            
            # Step 3: Train model
            trainer = self.train_model(train_dataset, eval_dataset, training_config)
            
            # Step 4: Evaluate model
            data_collator = self.preprocessor.create_data_collator()
            evaluation_results = self.evaluate_model(eval_dataset, data_collator)
            
            # Step 5: Test inference
            inference_results = self.test_inference()
            
            logger.info("Pipeline completed successfully!")
            
            return {
                "model_path": str(self.output_dir),
                "evaluation_results": evaluation_results,
                "inference_results": inference_results
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

if __name__ == "__main__":
    # Run the complete pipeline
    pipeline = WhisperPidginPipeline(
        base_model="openai/whisper-small",
        output_dir="models/whisper-pidgin-demo"
    )
    
    # Configuration for quick demo training
    demo_config = {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "learning_rate": 5e-5,
        "warmup_steps": 50,
        "logging_steps": 5,
        "save_steps": 100,
        "eval_steps": 100
    }
    
    results = pipeline.run_full_pipeline(
        use_sample_data=True,
        augment_data=True,
        training_config=demo_config
    )
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED!")
    print("="*50)
    print(f"Model saved to: {results['model_path']}")
    print(f"Final WER: {results['evaluation_results'].get('eval_wer', 'N/A')}")
    print("\nInference test results:")
    for result in results['inference_results'][:3]:  # Show first 3
        print(f"  Expected: {result['expected']}")
        print(f"  Got: {result['transcribed']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print()