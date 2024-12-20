from pathlib import Path
import yaml
from typing import Dict, Any, Optional

class PEConfig:
    """Configuration manager for PE experiments with enhanced model and task handling."""
    
    def __init__(self, args):
        self.model_id = getattr(args, 'model', None)
        self.task = getattr(args, 'task', None)
        self.n = getattr(args, 'n', None) 
        self.family = getattr(args, 'family', None)
        
        config_path = Path(__file__).parent / "experiment_config.yaml"
        with open(config_path) as f:
            self._config = yaml.safe_load(f)
            
        if self.model_id not in self._config["models"] and self.model_id is not None:
            raise ValueError(f"Unknown model: {self.model_id}")
        if self.task not in self._config["tasks"]:
            raise ValueError(f"Unknown task: {self.task}")
            
        self._model_info = self._config["models"].get(self.model_id, {}) if self.model_id else {}
        self._task_info = self._config["tasks"][self.task]
        self._defaults = self._config.get("defaults", {})

        self.auth_token = self.get_hf_key()

    def get_hf_key(self) -> str:
        """Get HuggingFace auth token."""
        try:
            with open(".huggingface_token", "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError("Please create a .huggingface_token file with your HF token")
    
    @property
    def model_name(self) -> str:
        """Get full HuggingFace model name (e.g., 'meta-llama/Llama-3.2-3B')."""
        return self._model_info["huggingface_name"]

    @property
    def pretty_print_name(self) -> str:
        """ meta-llama/Llama-3.2-3B -> Llama 3.2 3B """
        def format_string(s):
            return ' '.join(s.split('/')[-1].split('-'))
        return format_string(self._model_info["huggingface_name"])
    
    @property
    def formatted_name(self) -> str:
        """Get formatted model name (e.g., 'llama323b')."""
        return self._model_info["formatted_name"]
    
    @property
    def family_name(self) -> str:
        """Get model family name (e.g., 'meta-llama', 'Qwen')."""
        return self._model_info["family_name"]
    
    @property
    def task_description(self) -> str:
        """Get task description."""
        return self._task_info["description"]
    
    @property
    def train_size(self) -> int:
        """Get training set size."""
        return self._task_info["train_size"]
    
    @property
    def eval_size(self) -> int:
        """Get evaluation set size."""
        return self._task_info["eval_size"]
    
    @property
    def batch_size(self) -> int:
        """Get batch size for processing."""
        return self._task_info["batch_size"]
    
    @property
    def max_length(self) -> int:
        """Get max sequence length."""
        return self._task_info["max_length"]
    
    @property
    def output_dir(self) -> Path:
        """Get base output directory."""
        return Path(self._defaults["output_dir"])
    
    @property
    def tensor_dir(self) -> Path:
        """Get directory for tensor outputs."""
        return Path(self._defaults["tensor_output_dir"])
    
    @property
    def viz_dir(self) -> Path:
        """Get directory for visualization outputs."""
        return Path(self._defaults["viz_output_dir"])

    @property
    def eval_dir(self) -> Path:
        """ Store evaluation results. """
        return Path(self._defaults["eval_dir"])
    
    def get_tensor_path(self, suffix: Optional[str] = None) -> Path:
        base_name = f"{self.formatted_name}_{self.task}"
        if suffix:
            base_name = f"{base_name}_{suffix}"
        return self.output_dir / f"{base_name}.pt"
    
    def get_viz_path(self, suffix: Optional[str] = None) -> Path:
        base_name = f"{self.formatted_name}_{self.task}"
        if suffix:
            base_name = f"{base_name}_{suffix}"
        return self.viz_dir / f"{base_name}.png"

    @property
    def csv_path(self) -> Path:
        # csv = f"{self.output_dir}/{self.formatted_name}_{self.task}_similarities.csv"
        csv = f"{self.output_dir}/{self.formatted_name}_boolq_similarities.csv"
        return csv

    @property
    def adapter_path(self) -> Path:
        adapter = f"models/adapters/{self.formatted_name}/{self.n}"
        return Path(adapter)

    @property
    def base_adapter_path(self) -> Path:
        adapter = f"models/adapters/{self.formatted_name}/0"
        return Path(adapter)                                           

    def ensure_output_dirs(self) -> None:
        """Ensure all output directories exist."""
        self.output_dir.mkdir(exist_ok=True)
        self.tensor_dir.mkdir(exist_ok=True, parents=True)
        self.viz_dir.mkdir(exist_ok=True, parents=True)

    @property
    def family_models(self):
        """Get formatted names of all models in the specified family."""
        return [info["formatted_name"] for info in self._config["models"].values()
                if info["family_name"] == self.family]

    def __str__(self) -> str:
        """Get string representation of current configuration."""
        return (f"PEConfig(model={self.model_id}, task={self.task})\n"
                f"  Model: {self.model_name} ({self.family_name})\n"
                f"  Task: {self.task_description}\n"
                f"  Batch Size: {self.batch_size}\n"
                f"  Max Length: {self.max_length}\n"
                f"  Train Size: {self.train_size}\n"
                f"  Eval Size: {self.eval_size}")
