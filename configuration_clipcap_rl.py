from transformers import PretrainedConfig
from typing import Optional
@dataclass
class ModelConfig:
    language_model_id: Optional[str] = field(default="microsoft/Phi-3.5-mini-instruct", metadata={"help": "The language model id."})
    prefix_length: Optional[int] = field(default=10, metadata={"help": "The prefix length."})
    max_length: Optional[int] = field(default=20, metadata={"help": "The maximum length."})
    d_model: Optional[int] = field(default=3072, metadata={"help": "The model dimension."})
    d_clip: Optional[int] = field(default=512, metadata={"help": "The CLIP dimension."})
    n_heads: Optional[int] = field(default=8, metadata={"help": "The number of heads."})
    n_layers: Optional[int] = field(default=12, metadata={"help": "The number of layers."})
    d_ff: Optional[int] = field(default=2048, metadata={"help": "The feed forward dimension."})
    dropout: Optional[float] = field(default=0.1, metadata={"help": "The dropout rate."})
    data_dir: Optional[str] = field(default="data", metadata={"help": "The data directory."})
    use_lora: Optional[bool] = field(default=False, metadata={"help": "Whether to use LoRA."})
class ClipCapRLConfig(PretrainedConfig):

    model_type = "clipcap-rl"

    def __init__(
        self, 
        language_model_id: Optional[str] = "microsoft/Phi-3.5-mini-instruct",
        prefix_length: Optional[int] = 10,
        max_length: Optional[int] = 20,
        d_model: Optional[int] = 3072,
        d_clip: Optional[int] = 512,
        n_heads: Optional[int] = 8,
        n_layers: Optional[int] = 12,
        d_ff: Optional[int] = 2048,
        dropout: Optional[float] = 0.1,
        **kwards,
    ):
        
        self.language_model_id = language_model_id
        self.prefix_length = prefix_length
        self.max_length = max_length
        self.d_model = d_model
        self.d_clip = d_clip
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout

        super().__init__(**kwards)
