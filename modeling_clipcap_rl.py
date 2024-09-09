from transformers import PreTrainedModel

from configuration_clipcap_rl import ClipCapRLConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from typing import Optional, Any


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

    def __init__(self, conf: ClipCapRLConfig) -> None:
        super().__init__()

        self.conf = conf
        self.in_proj = nn.Linear(self.conf.d_clip,
                                self.conf.prefix_length * self.conf.d_model, bias=False)

        self.encoder_layers = nn.ModuleList([ \
                EncoderLayer(self.conf.d_model, self.conf.n_heads, self.conf.d_ff, self.conf.dropout) \
                    for _ in range(self.conf.n_layers)])        
        self.prefix_const = nn.Parameter(torch.randn(self.conf.prefix_length, self.conf.d_model), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = x.view(-1, self.conf.prefix_length, self.conf.d_model)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat((x, prefix), dim=1)
        # x = F.dropout(self.positional_encoding(x), self.dropout)
        for encoder in self.encoder_layers:
            x = encoder(x, mask=None)
        return x[:, self.conf.prefix_length:]

class ClipCapRLModel(PreTrainedModel):
    config_class = ClipCapRLConfig

    def __init__(self, config: ClipCapRLConfig):
        super().__init__(config)
        self.config = config
        self.setup()

    def setup(self):
        _model = AutoModelForCausalLM.from_pretrained(
                    self.config.language_model_id,
                    use_cache=False,
                )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.language_model_id, clean_up_tokenization_spaces=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(_model)
        self.prefix_encoder = VisionPrefixModel(self.config)
        if self.config.use_lora:
            _model.gradient_checkpointing_enable()

            _model = get_peft_model(_model, 
                        LoraConfig(
                            task_type="CAUSAL_LM",
                            r=32,
                            lora_alpha=64,
                            target_modules=[
                                'q_proj',
                                'k_proj',
                                'v_proj',
                                'o_proj',
                                'gate_proj',
                                'up_proj',
                                'down_proj',
                                'lm_head',
                            ],
                            lora_dropout=0.1,
                            bias="none",
                        )
                    )

        self.language_model = _model

    def train(self, mode: bool=True):
        self.prefix_encoder.train(mode)
        self.language_model.train(mode and self.config.use_lora)
        return self
    
    def eval(self):
        self.train(False)
        return self

    def forward(self, 
        image_embeddings: torch.Tensor, 
        tokens: Optional[torch.Tensor]=None, 
        mask: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        x = self.prefix_encoder(image_embeddings).view(-1, self.config.prefix_length, self.config.d_model)

        if tokens is not None:
            token_embeddings = self.language_model.get_input_embeddings()(tokens).squeeze(1)
            x = torch.cat((x, token_embeddings), dim=1)
        self.language_model.eval()
        x = self.language_model(inputs_embeds=x, attention_mask=mask)
        return x
    
    @torch.inference_mode()
    def generate(
        self,
        image_embeddings: torch.Tensor,
        max_length: Optional[int] = 512,
        max_new_tokens: Optional[int] = 512,
        stop_words = ['.'], 
        temperature: Optional[float]=1.0, 
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        n_beams: Optional[int] = None,
        do_sample: Optional[bool] = True,
        num_return_sequences: Optional[int] = 1,
    ) -> torch.LongTensor:
    
        # Convert image embeddings to input ids
        inputs_embeds = self.prefix_encoder(image_embeddings).view(-1, self.config.prefix_length, self.config.d_model)
        # Transform stop words to token ids
        stop_ids = [self.tokenizer.encode(w)[0] for w in stop_words]
        setence_count = 1
        # Calculate the maximum length of the generated sequence
        max_length = min(max_length - self.config.prefix_length, max_new_tokens)
        new_tokens = None
        for _ in range(max_length):
            _next_token = self.next_token(inputs_embeds, temperature, top_k)
            
            if new_tokens is None:
                new_tokens = _next_token
            else:
                new_tokens = torch.cat((new_tokens, _next_token), dim=1)

            next_token_embed = self.language_model.get_input_embeddings()(_next_token)

            inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)
            if _next_token in stop_ids:
                if setence_count >= num_return_sequences:
                    break
                setence_count += 1
        return new_tokens
    
    def next_token(self, inputs_embeds, temperature=1.0, top_k=None):
        out = self.language_model(inputs_embeds=inputs_embeds)
        next_token_logits = out.logits[:, -1, :]

        if top_k is not None:
            v, i = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits = torch.full_like(next_token_logits, float("-inf")).scatter_(-1, i, v)
        
        if temperature > 0.0:
            probs = F.softmax(next_token_logits / temperature, dim=-1)

            if torch._dynamo.is_compiling():
                # Faster alternative to `torch.multinomial(probs, num_samples=1)` that is also CUDAGraph friendly
                distribution = torch.empty_like(probs).exponential_(1)
                return torch.argmax(probs / distribution, dim=-1, keepdim=True)
            return torch.multinomial(probs, num_samples=1)
        
        return torch.argmax(next_token_logits, dim=-1, keepdim=True)

    @torch.inference_mode()
    def generate_with_critic(self,
        critic_model: Any,
        image_embeddings: torch.Tensor,
        max_length: Optional[int] = 512,
        max_new_tokens: Optional[int] = 77,
        stop_words = ['.'], 
        temperature: Optional[float]=1.0, 
        top_k: Optional[int] = 50):

        # Convert image embeddings to input ids
        inputs_embeds = self.prefix_encoder(image_embeddings).view(-1, self.config.prefix_length, self.config.d_model)
        # Transform stop words to token ids
        stop_ids = [self.tokenizer.encode(w)[0] for w in stop_words]
        # Calculate the maximum length of the generated sequence
        max_length = min(max_length - self.config.prefix_length, max_new_tokens)
        new_tokens = None

        for step in range(max_length):
            out = self.language_model(inputs_embeds=inputs_embeds)
            next_token_logits = out.logits[:, -1, :]

            if top_k is not None:
                v, i = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits = torch.full_like(next_token_logits, float("-inf")).scatter_(-1, i, v)
        
            if temperature > 0.0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)

            if step != 0 and step % 2 == 0:

                sampled_tokens = torch.multinomial(probs, num_samples=top_k)
                sampled_tokens = sampled_tokens.permute(1, 0)
                
                if new_tokens is None:
                    candidates_tokens = sampled_tokens
                else:
                    candidates_tokens = torch.cat(
                        (new_tokens.repeat(sampled_tokens.shape[0], new_tokens.shape[0]), 
                        sampled_tokens), dim=1)
                
                candidates = self.tokenizer.batch_decode(candidates_tokens, skip_special_tokens=True)
                scores = critic_model(image_embeddings, candidates)
                _next_token = sampled_tokens[scores.argmax().item()].unsqueeze(0)
            else:
                _next_token = torch.multinomial(probs, num_samples=1)

            if new_tokens is None:
                new_tokens = _next_token
            else:
                new_tokens = torch.cat((new_tokens, _next_token), dim=1)

            next_token_embed = self.language_model.get_input_embeddings()(_next_token)
            inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)
            if _next_token in stop_ids:
                break

        return new_tokens

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"Trainable params: {trainable_params} | Total params: {all_param} | Trainable %: {100 * trainable_params / all_param}")






