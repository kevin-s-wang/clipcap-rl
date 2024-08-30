from transformers import PretrainedConfig
from typing import Optional

class ClipCapRLConfig(PretrainedConfig):

    model_type = "clipcap-rl"

    def __init__(
        self, 
        language_model_id: Optional[str] = "gpt2-large",
        prefix_length: Optional[int] = 10,
        max_length: Optional[int] = 20,
        d_model: Optional[int] = 1280,
        d_clip: Optional[int] = 512,
        n_heads: Optional[int] = 8,
        n_layers: Optional[int] = 12,
        d_ff: Optional[int] = 2048,
        dropout: Optional[float] = 0.1,
        use_lora: Optional[bool] = False,
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
        self.use_lora = use_lora

        super().__init__(**kwards)
