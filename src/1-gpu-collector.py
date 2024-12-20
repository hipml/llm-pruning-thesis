import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
from support.peconfig import PEConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DistributedEnvironment:
    """
    handles distributed training setup
    """

    def __init__(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.is_distributed = False
        self.is_main = True
    
    def setup(self):
        """Initialize distributed environment"""
        if self.local_rank != -1:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl")
            dist.barrier()
            self.is_distributed = True
            self.is_main = (self.local_rank == 0)
            return True
        return False

    def cleanup(self):
        """Cleanup distributed environment"""
        if dist.is_initialized():
            dist.destroy_process_group()

    def is_main_process(self):
        """Check if this is the main process"""
        return not dist.is_initialized() or dist.get_rank() == 0

    @property
    def rank(self):
        """ Get local rank """
        return self.local_rank

    @property
    def world_size(self):
        """ Get world size """
        return int(os.environ.get("WORLD_SIZE", 1))

class ModelHandler:
    """Handles all model-related operations"""

    def __init__(self, config, rank):
        self.config = config
        self.local_rank = rank
        self._initialize_model()

    def _initialize_model(self):
        """Init our model and tokenizer with distributed support"""

        if self.local_rank != -1:
            dist.barrier()
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            token=self.config.auth_token,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            token=self.config.auth_token,
            quantization_config=quant_config,
            trust_remote_code=True,
        )

        if self.local_rank != -1:
            dist.barrier()
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )

        print("Quantized model loaded")

    def tokenize_text(self, text):
        """Tokenize text"""
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
        ).input_ids.to(self.local_rank if self.local_rank != -1 else self.model.device)

class DatasetHandler:
    """Handles all dataset-related operations"""
    
    def __init__(self, config, rank, world_size):
        self.config = config
        self.local_rank = rank
        self.world_size = world_size
        self._initialize_dataloader()

    def _initialize_dataloader(self):
        """Initialize the loader with distributed sampler support"""
        # if self.config.dataset_name == "boolq":
        #     dataset = load_dataset(
        #         "super_glue",
        #         "boolq",
        #         split=f"train[:{self.config.num_rows}]",
        #         trust_remote_code=True
        #     )
        # else:
        #     dataset = load_dataset(
        #         self.config.dataset_name,
        #         split=f"train[:{self.config.num_rows}]"
        #     )

        dataset = load_dataset(
            self.config.task,
            split=f"train[:{self.config.train_size}]",
            trust_remote_code=True,
        )

        # Create distributed sampler if using distributed training
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.local_rank
        ) if self.local_rank != -1 else None

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=1,
            pin_memory=True
        )

class HiddenStatesCollector:
    """Handles collection and storage of hidden states"""
    
    def __init__(self, config, model_handler):
        self.config = config
        self.model_handler = model_handler
        dist_env = DistributedEnvironment()
        if dist_env.is_main_process():
            os.makedirs(config.tensor_dir, exist_ok=True)

    def process_batch(self, batch):
        """Process a single batch and return hidden states by layer"""
        try:
            torch.cuda.empty_cache()
            inputs = []

            # BoolQ
            if 'passage' in batch and 'question' in batch:
                for passage, question in zip(batch['passage'], batch['question']):
                    prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer: "
                    inputs.append(self.model_handler.tokenize_text(prompt))
            else:
                raise ValueError("Unsupported batch format.")

            inputs = torch.stack(inputs, dim=0).squeeze(1)

            with torch.no_grad():
                if isinstance(self.model_handler.model, DDP):
                    print("DDP")
                    outputs = self.model_handler.model.module(inputs, output_hidden_states=True)
                else:
                    print("Else")
                    outputs = self.model_handler.model(inputs, output_hidden_states=True)

            return outputs.hidden_states
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return None
        finally:
            torch.cuda.empty_cache()

    def save_hidden_states(self, hidden_states, batch_index):
        final_token_hidden_states = [hs[:, -1, :].detach() for hs in hidden_states]
        hidden_states_dict = {}

        batch_size = hidden_states[0].shape[0]
        print("Saving hidden states....")
        for example_index in range(batch_size):
            example_dict = {}
            for layer_idx, layer_hidden_state in enumerate(final_token_hidden_states):
                example_dict[f"layer_{layer_idx}"] = layer_hidden_state[example_index].cpu()
            hidden_states_dict[f"layer_{example_index}"] = example_dict

        file_name = f"{self.config.formatted_name}_{self.config.task}_batch_{batch_index}.pt"
        output_path = os.path.join(self.config.tensor_dir, file_name)

        torch.save(hidden_states_dict, output_path)
        print("Hidden states saved!")

def parse_arguments():
    """Parse command line arguments and return Config object."""
    parser = argparse.ArgumentParser(description="LLM Hidden States Collection")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--task", type=str, default="boolq")
    args = parser.parse_args()

    return PEConfig(args)
    
   
def run_collection(config):
    """Main collection process"""
    dist_env = DistributedEnvironment()
    is_distributed = dist_env.setup()
    
    try:
        if is_distributed:
            dist.barrier()
            
        model_handler = ModelHandler(config, dist_env.rank)

        if is_distributed:
            dist.barrier()
            
        dataset_handler = DatasetHandler(config, dist_env.rank, dist_env.world_size)
        collector = HiddenStatesCollector(config, model_handler)

        for batch_index, batch in enumerate(dataset_handler.dataloader):
            # are we in the main process?
            if dist_env.is_main:
                logger.info(f"Processing batch {batch_index}")
            
            hidden_states = collector.process_batch(batch)

            if hidden_states is not None:
                collector.save_hidden_states(hidden_states, batch_index)
                torch.cuda.empty_cache()
    finally:
        if is_distributed:
            dist.barrier()
            dist_env.cleanup()

def main():
    """Hidden State Collector with multi-GPU support"""
    try:
        config = parse_arguments()
        run_collection(config)
        dist_env = DistributedEnvironment()
        if dist_env.is_main:
            logger.info("Processing completed")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
