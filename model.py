import torch
import torch.nn as nn
from dataclasses import dataclass, field
import math
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False

@dataclass
class ModelConfig:
    language_model_id: Optional[str] = field(default="microsoft/Phi-3.5-mini-instruct", metadata={"help": "The language model id."})
    prefix_length: Optional[int] = field(default=10, metadata={"help": "The prefix length."})
    max_length: Optional[int] = field(default=20, metadata={"help": "The maximum length."})
    d_clip: Optional[int] = field(default=512, metadata={"help": "The CLIP dimension."})
    n_heads: Optional[int] = field(default=8, metadata={"help": "The number of heads."})
    n_layers: Optional[int] = field(default=12, metadata={"help": "The number of layers."})
    d_ff: Optional[int] = field(default=2048, metadata={"help": "The feed forward dimension."})
    dropout: Optional[float] = field(default=0.1, metadata={"help": "The dropout rate."})
    data_dir: Optional[str] = field(default="data", metadata={"help": "The data directory."})
    use_lora: Optional[bool] = field(default=False, metadata={"help": "Whether to use LoRA."})

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    # This is the rank of the decomposed matrices A and B to be learned during fine-tuning. A smaller number will save more GPU memory but might result in worse performance.
    r=32,
    # This is the coefficient for the learned ΔW factor, so the larger number will typically result in a larger behavior change after fine-tuning.
    lora_alpha=64,
    # Drop out ratio for the layers in LoRA adaptors A and B.
    lora_dropout=0.1,
    # We fine-tune all linear layers in the model. It might sound a bit large, but the trainable adapter size is still only **1.16%** of the whole model.
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    # Bias parameters to train. 'none' is recommended to keep the original model performing equally when turning off the adapter.
    bias="none",
)

nf4_config = BitsAndBytesConfig(
    # Load the model with 4-bit quantization
    load_in_4bit=True,
    # Use double quantization
    bnb_4bit_use_double_quant=True,
    # Use 4-bit Normal Float for storing the base model weights in GPU memory
    bnb_4bit_quant_type="nf4",
    # De-quantize the weights to 16-bit (Brain) float before the forward/backward pass
    bnb_4bit_compute_dtype=torch.bfloat16,
)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attns = torch.matmul(attn_probs, V)
        return attns, attn_probs
    
    
    def split_heads(self, x) -> torch.Tensor:
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x) -> torch.Tensor:
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attns, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        out = self.W_o(self.combine_heads(attns))
        return out
    
class PositionWiseFeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.silu = nn.SiLU()
        
    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.silu(self.fc1(x)))
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]
    
class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed
    

class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        
        self.multi_head_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask) -> torch.Tensor:
        
        attns = self.multi_head_attention(x, x, x, mask)
        
        x = self.norm1(x + self.dropout(attns))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class VisionPrefixModel(nn.Module):
    '''A constant prefix that is added to the input embeddings.'''

    def __init__(self, conf: ModelConfig) -> None:
        super().__init__()

        self.conf = conf
        self.in_proj = nn.Linear(self.conf.d_clip,
                                self.conf.prefix_length * self.conf.d_model, bias=False)

        self.encoder_layers = nn.ModuleList([ \
                EncoderLayer(self.conf.d_model, self.conf.n_heads, self.conf.d_ff, self.conf.dropout) \
                    for _ in range(self.conf.n_layers)])        
        self.prefix_const = nn.Parameter(torch.randn(self.conf.prefix_length, self.conf.d_model), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = x.view(-1, self.conf.prefix_length, self.conf.d_model)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat((x, prefix), dim=1)
        # x = F.dropout(self.positional_encoding(x), self.dropout)
        for encoder in self.encoder_layers:
            x = encoder(x, mask=None)
        return x[:, self.conf.prefix_length:]


class ClipCapModel(nn.Module):
    def __init__(self, conf: ModelConfig) -> None:
        super().__init__()
        self.conf = conf
        self.setup()

    def setup(self):
        _model = AutoModelForCausalLM.from_pretrained(self.conf.language_model_id, quantization_config=nf4_config)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.conf.language_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.prefix_encoder = VisionPrefixModel(self.conf)
        if self.conf.use_lora:
            # Enabling gradient checkpointing, to make the training further efficient
            _model.gradient_checkpointing_enable()
            # Set up the model for quantization-aware training e.g. casting layers, parameter freezing, etc.
            _model = prepare_model_for_kbit_training(_model)

            _model = get_peft_model(_model, lora_config)
            _model.print_trainable_parameters()

        self.language_model = _model


    def train(self, mode: bool = True):
        self.prefix_encoder.train(mode)
        if self.conf.use_lora:
            self.language_model.train(mode)
        return self
    
    def eval(self):
        self.train()
        return self
    
    @torch.inference_mode()
    def generate(self, inputs: torch.Tensor,
                    temperature: float = 1.0,
                    top_k: int = 50,
                    max_seq_length: int = 30,
                    num_beams: int = 5,
                    keep_prompt: bool = True,
                    stop_words: List[str] = ['.', '\n']) -> str:
        input_embedds = self.prefix_encoder(inputs).view(-1, self.conf.prefix_length, self.conf.d_model)
        
        inputs_ids = self.language_model(inputs_embeds=input_embedds).logits.argmax(dim=-1)
   
        stop_ids = [self.tokenizer.encode(w)[0] for w in stop_words]
        stop_criteria = KeywordsStoppingCriteria(stop_ids)

        tokens = self.language_model.generate(
                        inputs_ids, 
                        max_new_tokens=max_seq_length, 
                        pad_token_id=self.tokenizer.eos_token_id,
                        early_stopping=True, 
                        num_beams=num_beams,                
                        temperature=temperature,
                        top_k=top_k,
                        stopping_criteria=[stop_criteria],
                        do_sample=True)
        
        if not keep_prompt:
            tokens = tokens[:, self.conf.prefix_length-1: ]
            
        return tokens

    def forward(self, image_embeddings: torch.Tensor, tokens: Optional[torch.Tensor]=None, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        x = self.prefix_encoder(image_embeddings).view(-1, self.conf.prefix_length, self.conf.d_model)

        if tokens is not None:
            token_embeddings = self.language_model.get_input_embeddings()(tokens).squeeze(1)
            x = torch.cat((x, token_embeddings), dim=1)
        
        x = self.language_model(inputs_embeds=x, attention_mask=mask)
        return x