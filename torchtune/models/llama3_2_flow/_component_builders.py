from functools import partial
from typing import List, Optional

from torch import nn
import torch
from einops import rearrange
from torch import Tensor

from torchtune.models.llama3._model_utils import scale_hidden_dim_for_mlp
from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE
from torchtune.modules import TiedLinear
from torchtune.modules import (
    MultiHeadAttention,
    FeedForward,
    FrozenNF4Linear,
    RMSNorm,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
)

from torchtune.modules.common_utils import reparametrize_as_dtype_state_dict_post_hook

from torchtune.modules.peft import DoRALinear, LORA_ATTN_MODULES, LoRALinear

"""
Component builders for the Llama3.2 model and popular variants such as LoRA.

torchtune provides composable building blocks. Builder functions help
stitch these building blocks into higher-level components. This design has
two benefits:
- The building blocks themselves are very flexible. For example, ``MultiHeadAttention``
can take either nn.Linear or nn.LoRALinear for ``q_proj``.
- Builder functions expose a set of configurable params which keep the constructors of
the building blocks simple.
"""


# ------------------ Vanilla Llama3.2 ------------------

def llama3_2(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 500_000,
    intermediate_dim: Optional[int] = None,
    norm_eps: float = 1e-5,
    scale_factor: int = 32,
) -> TransformerDecoder:
    """
    Build the decoder associated with the Llama3.2 model. This includes:
    - Token embeddings
    - num_layers number of TransformerSelfAttentionLayer blocks
    - RMS Norm layer applied to the output of the transformer
    - Final projection into token space

    Args:
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        embed_dim (int): embedding dimension for self-attention
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        rope_base (int): base for the rotary positional embeddings. Default: 500_000
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        intermediate_dim (Optional[int]): intermediate dimension for MLP. If not specified,
            this is computed using :func:`~torchtune.modules.scale_hidden_dim_for_mlp`
        norm_eps (float): epsilon in RMS norms.
        scale_factor (int): scaling factor for RoPE. Default: 32

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 model.
    """
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    rope = Llama3ScaledRoPE(dim=head_dim, max_seq_len=max_seq_len, base=rope_base, scale_factor=scale_factor)
    layers = []
    for _ in range(num_layers):
        self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        hidden_dim = intermediate_dim if intermediate_dim else scale_hidden_dim_for_mlp(embed_dim)
        mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim)
        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        layers.append(layer)
    layers = nn.ModuleList(layers)

    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    output_proj = TiedLinear(tok_embeddings)
    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )

def llama3_mlp(dim: int, hidden_dim: int, quantize_base: bool = False) -> FeedForward:
    """
    Build the MLP layer associated with the Llama model.
    """
    gate_proj = nn.Linear(dim, hidden_dim, bias=False) if not quantize_base else FrozenNF4Linear(dim, hidden_dim, bias=False)
    down_proj = nn.Linear(hidden_dim, dim, bias=False) if not quantize_base else FrozenNF4Linear(hidden_dim, dim, bias=False)
    up_proj = nn.Linear(dim, hidden_dim, bias=False) if not quantize_base else FrozenNF4Linear(dim, hidden_dim, bias=False)
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)


#----------- IMAGE STUFF ---------

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)
    
def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding

class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))
    

class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
    
class ImageAdapter(nn.Module):
    def __init__(self, img_latent_dim, llama_hidden_size, img_size):
        super().__init__()
        self.img_pos_embed = EmbedND(dim=img_latent_dim, theta=10000, axes_dim=[img_size, img_size, 1])
        self.projection = nn.Linear(img_latent_dim, llama_hidden_size)

    def forward(self, img_encoded, img_ids):
        # Apply 2D positional encoding
        img_pos = self.img_pos_embed(img_ids)
        img_with_pos = img_encoded + img_pos

        # Project to Llama's hidden size
        return self.projection(img_with_pos)
    
class LlamaRectifiedFlow(nn.Module):
    def __init__(self, llama_hidden_size):
        super().__init__()
        self.llama = llama3_2(
            vocab_size=128_256,
            num_layers=16,
            num_heads=32,
            num_kv_heads=8,
            embed_dim=2048,
            max_seq_len=131072,
            intermediate_dim=8192,
            attn_dropout=0.0,
            norm_eps=1e-5,
            rope_base=500_000,
            scale_factor=32,
        )
        
        # Add any additional components needed for rectified flow
        self.velocity_projector = nn.Linear(llama_hidden_size, llama_hidden_size)
        
    def forward(self, combined_input, timestep):
        # Process through Llama
        llama_output = self.llama(combined_input)
        
        # Generate velocity field for rectified flow
        velocity = self.velocity_projector(llama_output)
        
        # You might want to add more processing here for the rectified flow
        
        return velocity

class FluxPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

    def rope(self, pos: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim % 2 == 0
        scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
        omega = 1.0 / (self.theta**scale)
        out = torch.einsum("n,d->nd", pos.float(), omega)
        out = torch.stack([torch.cos(out), torch.sin(out)], dim=-1)
        return out.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, h, d = x.shape
        pos = torch.arange(s, device=x.device)
        pe = self.rope(pos, d)  # Shape: [s, d/2, 2]
        
        x_complex = torch.view_as_complex(x.float().reshape(b, s, h, d//2, 2))
        pe_complex = torch.view_as_complex(pe)
        
        x_out = torch.view_as_real(x_complex * pe_complex.unsqueeze(1))
        
        return x_out.reshape(b, s, h, d)
    
class RotaryPositionalEmbeddings2D(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, max_height: int = 64, max_width: int = 64, theta: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.max_height = max_height
        self.max_width = max_width
        self.theta = theta
        self.build_rope_cache()
        self.build_2d_rope_cache()

    def build_rope_cache(self):
        seq_idx = torch.arange(self.max_seq_len, dtype=torch.float)
        theta = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        idx_theta = torch.einsum("i,j->ij", seq_idx, theta)
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def build_2d_rope_cache(self):
        h_idx = torch.arange(self.max_height, dtype=torch.float)
        w_idx = torch.arange(self.max_width, dtype=torch.float)
        theta = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        h_idx_theta = torch.einsum("i,j->ij", h_idx, theta)
        w_idx_theta = torch.einsum("i,j->ij", w_idx, theta)
        
        h_cache = torch.stack([torch.cos(h_idx_theta), torch.sin(h_idx_theta)], dim=-1)
        w_cache = torch.stack([torch.cos(w_idx_theta), torch.sin(w_idx_theta)], dim=-1)
        
        h_cache = h_cache.unsqueeze(1).expand(-1, self.max_width, -1, -1)
        w_cache = w_cache.unsqueeze(0).expand(self.max_height, -1, -1, -1)
        
        cache_2d = torch.cat([h_cache, w_cache], dim=-2)
        self.register_buffer("cache_2d", cache_2d, persistent=False)

    def forward(self, x: torch.Tensor, sot_positions: torch.Tensor, eov_positions: torch.Tensor, img_size: tuple) -> torch.Tensor:
        b, s, h, d = x.shape
        device = x.device
        pe = self.cache[:s].unsqueeze(0).expand(b, -1, -1, -1)
        
        print(f"Input x shape: {x.shape}")
        print(f"pe shape: {pe.shape}")
        print(f"self.cache_2d shape: {self.cache_2d.shape}")
        
        for i in range(b):
            sot = sot_positions[i].item()
            eov = min(eov_positions[i].item(), s)
            print(f"Batch {i}: sot = {sot}, eov = {eov}")
            
            if sot >= 0 and eov > sot:
                img_tokens = eov - sot
                height, width = img_size
                img_len = height * width
                print(f"img_tokens: {img_tokens}, img_size: {img_size}, img_len: {img_len}")
                
                if img_tokens <= img_len:
                    pe_img = self.cache_2d[:height, :width].reshape(img_len, self.dim, 2)[:img_tokens, :pe.shape[2], :]
                    print(f"pe_img shape (if): {pe_img.shape}")
                    print(f"pe[i, sot:eov] shape: {pe[i, sot:eov].shape}")
                    pe[i, sot:eov] = pe_img
                else:
                    pe_img = self.cache_2d[:height, :width].reshape(img_len, self.dim, 2)[:, :pe.shape[2], :]
                    print(f"pe_img shape (else): {pe_img.shape}")
                    print(f"pe[i, sot:sot+img_len] shape: {pe[i, sot:sot+img_len].shape}")
                    pe[i, sot:sot+img_len] = pe_img
                    pe[i, sot+img_len:eov] = self.cache[sot+img_len:eov, :pe.shape[2], :]

        x_complex = torch.view_as_complex(x.float().reshape(b, s, h, d//2, 2))
        pe_complex = torch.view_as_complex(pe)
        
        x_out = torch.view_as_real(x_complex * pe_complex.unsqueeze(2))
        
        return x_out.reshape(b, s, h, d)


class LlamaFluxModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 128_256,
        num_layers: int = 16,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        embed_dim: int = 2048,
        max_seq_len: int = 131072,
        intermediate_dim: int = 8192,
        attn_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        rope_base: int = 500_000,
        scale_factor: int = 32,
        img_latent_dim: int = 16,
        img_size: int = 64,
    ):
        super().__init__()
        
        # Initialize Llama 3.2 1B model
        self.llama = llama3_2(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            intermediate_dim=intermediate_dim,
            attn_dropout=attn_dropout,
            norm_eps=norm_eps,
            rope_base=rope_base,
            scale_factor=scale_factor,
        )
        
        # Image processing components
        self.img_adapter = ImageAdapter(img_latent_dim, embed_dim, img_size)
        
        # Positional encoding for combined input
        self.pos_encoding = RotaryPositionalEmbeddings2D(dim=embed_dim, max_seq_len=max_seq_len, max_height=img_size, max_width=img_size)
        
        # Components for rectified flow
        self.time_embed = MLPEmbedder(in_dim=256, hidden_dim=embed_dim)
        self.vec_embed = MLPEmbedder(in_dim=embed_dim, hidden_dim=embed_dim)
        
        # Separate projectors for image and text
        self.img_projector = nn.Linear(embed_dim, embed_dim)
        self.txt_projector = nn.Linear(embed_dim, embed_dim)
        
        # Final output layer
        self.final_layer = LastLayer(hidden_size=embed_dim, patch_size=1, out_channels=img_latent_dim)

    def forward(self, 
                tokens: Tensor,
                img: Optional[Tensor] = None,
                img_ids: Optional[Tensor] = None,
                timesteps: Optional[Tensor] = None,
                guidance: Optional[Tensor] = None,
                sot_positions: Optional[Tensor] = None,
                eov_positions: Optional[Tensor] = None
    ) -> Tensor:
        # Process text tokens
        text_embeds = self.llama.tok_embeddings(tokens)
        
        # Process image if present
        if img is not None and img_ids is not None:
            img_embeds = self.img_adapter(img, img_ids)
            combined_input = torch.cat([text_embeds, img_embeds], dim=1)
        else:
            combined_input = text_embeds
        
        # Apply positional encoding
        combined_input = self.pos_encoding(combined_input, sot_positions, eov_positions, (img_size, img_size))
        
        # Process through Llama layers
        for layer in self.llama.layers:
            combined_input = layer(combined_input)
        
        # Apply final norm
        combined_input = self.llama.norm(combined_input)
        
        # Rectified flow components
        if timesteps is not None:
            time_embed = self.time_embed(timestep_embedding(timesteps, 256))
            vec_embed = self.vec_embed(self.llama.output(combined_input[:, :tokens.shape[1]]))  # Use only text part for vec_embed
            
            # Combine embeddings
            context_embed = time_embed + vec_embed
            
            # Separate projections for image and text
            text_part = self.txt_projector(combined_input[:, :tokens.shape[1]])
            img_part = self.img_projector(combined_input[:, tokens.shape[1]:])
            
            # Combine projected parts
            velocity = torch.cat([text_part, img_part], dim=1)
            
            # Apply final layer for image output
            output = self.final_layer(velocity, context_embed)
        else:
            # For text-only input, use Llama's output projection
            output = self.llama.output(combined_input)
        
        return output