"""
Whisper Fine-tuning Trainer for Nigerian Pidgin
"""

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
from datasets import Dataset
import evaluate
import numpy as np
from typing import Dict, List, Any
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperPidginTrainer:
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        output_dir: str = "models/whisper-pidgin",
        language: str = "en",  # Base language
        task: str = "transcribe"
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and processor
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.processor = WhisperProcessor.from_pretrained(model_name)
        
        # Set language and task
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        
        # Initialize metrics
        self.wer_metric = evaluate.load("wer")
        
        logger.info(f"Initialized trainer with model: {model_name}")
    
    def compute_metrics(self, eval_pred):
        """
        Compute WER (Word Error Rate) for evaluation
        """
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids
        
        # Replace -100 with pad token id
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        
        # Decode predictions and labels
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # Compute WER
        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer}
    
    def setup_training_arguments(
        self,
        num_train_epochs: int = 10,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        learning_rate: float = 1e-5,
        warmup_steps: int = 500,
        logging_steps: int = 25,
        save_steps: int = 1000,
        eval_steps: int = 1000
    ) -> Seq2SeqTrainingArguments:
        """
        Setup training arguments
        """
        return Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            logging_steps=logging_steps,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            remove_unused_columns=False,
            label_names=["labels"],
            predict_with_generate=True,
            generation_max_length=225,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
        )
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        data_collator: Any = None,
        **training_kwargs
    ):
        """
        Fine-tune the Whisper model
        """
        # Setup training arguments
        training_args = self.setup_training_arguments(**training_kwargs)
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.processor.save_pretrained(self.output_dir)
        
        logger.info(f"Training completed. Model saved to {self.output_dir}")
        
        return trainer
    
    def evaluate_model(self, test_dataset: Dataset, data_collator: Any = None):
        """
        Evaluate the fine-tuned model
        """
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            remove_unused_columns=False,
            label_names=["labels"],
        )
        
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )
        
        results = trainer.evaluate()
        logger.info(f"Evaluation results: {results}")
        
        return results
    
    def save_training_config(self, config: Dict[str, Any]):
        """
        Save training configuration
        """
        config_path = self.output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Training config saved to {config_path}")

class PidginSpecificTrainer(WhisperPidginTrainer):
    """
    Specialized trainer for Nigerian Pidgin with custom preprocessing
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add Pidgin-specific tokens to vocabulary
        self.add_pidgin_tokens()
    
    def add_pidgin_tokens(self):
        """
        Add common Pidgin words to the tokenizer vocabulary
        """
        pidgin_tokens = [
            "dey", "wetin", "abeg", "wahala", "oya", "sabi", 
            "chop", "waka", "palava", "gbege", "katakata",
            "shakara", "ginger", "pepper", "scatter"
        ]
        
        # Add tokens to tokenizer
        new_tokens = []
        for token in pidgin_tokens:
            if token not in self.processor.tokenizer.get_vocab():
                new_tokens.append(token)
        
        if new_tokens:
            self.processor.tokenizer.add_tokens(new_tokens)
            self.model.resize_token_embeddings(len(self.processor.tokenizer))
            logger.info(f"Added {len(new_tokens)} Pidgin tokens to vocabulary")
    
    def create_pidgin_specific_metrics(self):
        """
        Create metrics specific to Pidgin evaluation
        """
        def pidgin_wer(predictions, references):
            # Custom WER calculation that accounts for Pidgin variations
            total_words = 0
            total_errors = 0
            
            for pred, ref in zip(predictions, references):
                pred_words = pred.lower().split()
                ref_words = ref.lower().split()
                
                # Simple word-level comparison
                max_len = max(len(pred_words), len(ref_words))
                errors = sum(1 for i in range(max_len) 
                           if i >= len(pred_words) or i >= len(ref_words) 
                           or pred_words[i] != ref_words[i])
                
                total_words += len(ref_words)
                total_errors += errors
            
            return total_errors / total_words if total_words > 0 else 0
        
        return pidgin_wer

if __name__ == "__main__":
    # Example usage
    trainer = PidginSpecificTrainer(
        model_name="openai/whisper-small",
        output_dir="models/whisper-pidgin-v1"
    )
    
    print("Trainer initialized successfully!")
    print(f"Model will be saved to: {trainer.output_dir}")