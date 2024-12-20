import argparse
import evaluate
import gc
import json
import logging
import os
import pandas as pd
import psutil
import string
import time
import torch
import types
from codecarbon import OfflineEmissionsTracker
from datasets import load_dataset, config
from datetime import datetime
from pathlib import Path
from peft import get_peft_model, LoraConfig, PeftModel, TaskType
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorForLanguageModeling
from transformers import TrainerCallback
from torch import nn
import torch.distributed as dist
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from support.peconfig import PEConfig

def setup_emissions_tracker(project_name="qlora_training"):
    """
    Configure and create an emissions tracker instance.
    
    Returns:
        EmissionsTracker: Configured tracker instance
    """
    # experiment_id = uuid.uuid4()
    tracker = OfflineEmissionsTracker(
        project_name=project_name,
        tracking_mode='machine',
        measure_power_secs=120, 
    )
    return tracker

class CacheCleanupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()

class ProfilingCallback(TrainerCallback):
    def __init__(self, log_dir: str = "logs/profiling_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.steps = []
        self.start_time = time.time()
        self.rank = int(os.environ.get("LOCAL_RANK", 0))
        
        logging.basicConfig(
            filename=self.log_dir / "profiler_rank{self.rank}.log",
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start = time.time()
        torch.cuda.synchronize()
        
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        duration = time.time() - self.step_start
        
        profile = {
            'step': state.global_step,
            'duration': duration,
            'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,
            'gpu_memory_cached': torch.cuda.memory_reserved() / 1024**3,
            'cpu_memory_used': psutil.Process().memory_info().rss / 1024**3
        }
        
        self.steps.append(profile)
        
        if state.global_step % 10 == 0:
            self._log_step(profile)
            self._save_metrics()
    
    def _log_step(self, profile):
        if self.rank == 0:
            logging.info(
                f"Step {profile['step']}: "
                f"Duration: {profile['duration']:.2f}s, "
                f"GPU Allocated: {profile['gpu_memory_allocated']:.2f}GB, "
                f"GPU Cached: {profile['gpu_memory_cached']:.2f}GB, "
                f"CPU Used: {profile['cpu_memory_used']:.2f}GB"
            )
    
    def _save_metrics(self):
        with open(self.log_dir / 'metrics.json', 'w') as f:
            json.dump(self.steps, f, indent=2)
            
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        avg_step_time = sum(s['duration'] for s in self.steps) / len(self.steps)
        max_gpu_mem = max(s['gpu_memory_allocated'] for s in self.steps)
        max_cpu_mem = max(s['cpu_memory_used'] for s in self.steps)
        
        logging.info(
            f"\nTraining Summary:\n"
            f"Total Time: {total_time:.2f}s\n"
            f"Average Step Time: {avg_step_time:.2f}s\n"
            f"Peak GPU Memory: {max_gpu_mem:.2f}GB\n"
            f"Peak CPU Memory: {max_cpu_mem:.2f}GB"
        )


def setup_ddp():
    """Setup distributed data parallel (DDP) backend."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    print(f"DDP initialized on rank {rank} with {world_size} processes.")
    return rank, world_size


def save_adapter(model, config):
    print("Saving adapter...")
    config.adapter_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.adapter_path)
    print(f"Adapter saved in {config.adapter_path}.")

def setup_training(model, tokenizer, config, world_size, rank):
    """ Sets up training arguments and trainer for QLoRA fine-tuning on BoolQ. """

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    torch.backends.cudnn.benchmark = True

    if hasattr(model.config, 'use_memory_efficient_attention'):
        model.config.use_memory_efficient_attention = True

    if hasattr(model.config, 'use_flash_attention_2'):
        model.config.use_flash_attention_2 = True

    train_dataset = load_dataset("allenai/c4", 'en', split="train", streaming=True)
    
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples['text'],
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        return model_inputs
    
    # Preprocess the dataset
    processed_train = train_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=32,
        remove_columns=train_dataset.column_names,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = torch.exp(loss).item()
        return {'perplexity': perplexity}

    training_args = TrainingArguments(
        output_dir=config.adapter_path,
        resume_from_checkpoint=False,
        learning_rate=3e-4,
        max_steps=5000,
        warmup_steps=100,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=max(4 // world_size, 1),
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        optim="adamw_bnb_8bit",
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        ddp_backend='nccl',
        hub_token=config.auth_token,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False
        ),
        compute_metrics=compute_metrics,
        callbacks=[ProfilingCallback(), CacheCleanupCallback()]
    )
    
    return trainer, training_args


def add_adapter(model):
    """ Adds a QLoRA adapter for fine-tuning """

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    base_config = {
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": ["gate_proj", "down_proj", "up_proj"],
        "task_type": "CAUSAL_LM"
    }

    model_name = model.config.model_type.lower()

    # default 
    rank = 8
    if "llama" in model_name:
        # 321b = 4
        # 70b models = 8
        if "321" or "323" in model_name:
            rank = 4
        if "38" or "70" in model_name:
            rank = 8 
        config = {
            **base_config,
            "r": rank,
            "lora_alpha": rank,
        }
    elif "qwen" in model_name:
        # 2505b = 4 
        if "2505b" in model_name:
            rank = 4
        if "70" in model_name:
            rank = 8
        config = {
            **base_config,
            "r": rank,
            "lora_alpha": rank,
        }
    else:
        raise ValueError(f"what model is this lol {model_name}")

    lora_config = LoraConfig(**config)
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False

    return model

def prune_model(model, config, n):
    """Prunes n consecutive layers from the model and updates KV cache indexing."""
    if n == 0:
        return model
        
    # read and filter similarity CSV
    df = pd.read_csv(config.csv_path)
    filtered_df = df[
        (df['layers_to_prune'] == n) & 
        (df['metric'] == 'exp')
    ].sort_values('similarity', ascending=False)
    
    if filtered_df.empty:
        raise ValueError(f"No matching entries found for {n} layers to prune")
    
    # get starting layer index and layers to prune
    start_layer = filtered_df.iloc[0]['starting_layer']
    layers_to_prune = range(start_layer, start_layer + n)
    
    # Update model config
    model.config.num_hidden_layers -= n
    
    # Prune layers
    model.model.layers = nn.ModuleList([
        layer for i, layer in enumerate(model.model.layers)
        if i not in layers_to_prune
    ])
    
    # Store which layers were pruned for KV cache handling
    model.pruned_layers = list(layers_to_prune)
    
    # Modify forward pass to handle KV cache
    original_forward = model.forward
    
    def new_forward(self, *args, **kwargs):
        if 'past_key_values' in kwargs and kwargs['past_key_values'] is not None:
            past_kv = kwargs['past_key_values']
            kwargs['past_key_values'] = tuple(
                kv for i, kv in enumerate(past_kv) 
                if i not in self.pruned_layers
            )
        return original_forward(*args, **kwargs)
    
    model.forward = types.MethodType(new_forward, model)
    
    print(f"Pruned architecture: \n{model}")
    return model


def load_adapter(model, config):
    try:
        adapter_model = PeftModel.from_pretrained(
            model,
            config.adapter_path,
            token=config.auth_token
        )
        return adapter_model, True
    except:
        return model, False

def load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        token=config.auth_token,
        trust_remote_code=True,
        use_fast=True,
        padding_side="left",
        # device_map="auto"
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
 

def load_model(model_name, auth_token):
    torch.cuda.empty_cache()
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=auth_token,
        trust_remote_code=True,
        quantization_config=quant_config,
        # device_map="auto",
    )
    return model

def parse_arguments():
    parser = argparse.ArgumentParser(description="Distributed evaluation of pruned LLM")
    parser.add_argument("--model", type=str, required=True, help="Model identifier from config")
    parser.add_argument("--task", type=str, required=True, help="Task identifier from config")
    parser.add_argument("--nltp", type=int, required=True, help="Number of layers to prune")
    return parser.parse_args()

    
def main():
    args = parse_arguments()
    args.n = args.nltp
    config = PEConfig(args)
    
    torch.cuda.empty_cache()
    
    rank, world_size = setup_ddp()

    try:
        tokenizer = load_tokenizer(config)
        model = load_model(config.model_name, config.auth_token)
        
        model = prune_model(model, config, args.n)
        model = add_adapter(model)

        device = torch.device(f"cuda:{rank}")
        model = model.to(device)

        if rank == 0:
            print("Setting up training...")
        trainer, training_args = setup_training(model, tokenizer, config, world_size, rank)        

        if rank == 0:
            print("Training...")
        tracker = setup_emissions_tracker()
        tracker.start()
        
        trainer.train()

        emissions = tracker.stop()
        print(emissions)
            
        save_adapter(model, config)
    finally:
        torch.cuda.empty_cache()
        _ = tracker.stop()
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
