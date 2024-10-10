from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, List, Tuple, Union

from torch import nn

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
from torchtune.modules.attention_utils import _MaskType

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
    patch_size: int = 16,
    in_channels: int = 3,
    out_channels: int = 3,
) -> nn.Module:
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
    
    class TimestepEmbedder(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.embed_dim = embed_dim
            self.net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.SiLU(),
                nn.Linear(embed_dim * 4, embed_dim)
            )

        def forward(self, t):
            t_embed = self.get_timestep_embedding(t, self.embed_dim)
            return self.net(t_embed)

        @staticmethod
        def get_timestep_embedding(timesteps, embedding_dim):
            half_dim = embedding_dim // 2
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
            emb = timesteps[:, None] * emb[None, :]
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
            if embedding_dim % 2 == 1:  # zero pad
                emb = F.pad(emb, (0, 1), mode='constant')
            return emb

    class FinalLayer(nn.Module):
        """
        The final layer of DiT.
        """
        def __init__(self, hidden_size, patch_size, out_channels):
            super().__init__()
            self.norm_final = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6,
            )
            self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )

        def forward(self, x, c, mask):
            x = x + self.mlp(c).unsqueeze(1) * mask.unsqueeze(2)
            x = self.norm_final(x)
            x = self.linear(x)
            return x

    class Llama3_2WithDiffusion(TransformerDecoder):
        def __init__(
            self,
            tok_embeddings: nn.Embedding,
            layers: Union[nn.Module, List[nn.Module], nn.ModuleList],
            max_seq_len: int,
            num_heads: int,
            head_dim: int,
            norm: nn.Module,
            output: Union[nn.Linear, Callable],
            patch_size: int,
            in_channels: int,
            out_channels: int,
            embed_dim: int,
            num_layers: Optional[int] = None,
            output_hidden_states: Optional[List[int]] = None,
        ):
            super().__init__(
                tok_embeddings=tok_embeddings,
                layers=layers,
                max_seq_len=max_seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
                norm=norm,
                output=output,
                num_layers=num_layers,
                output_hidden_states=output_hidden_states,
            )
            self.t_embedder = TimestepEmbedder(embed_dim)
            self.image_embed = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
            self.final_layer = FinalLayer(embed_dim, patch_size, out_channels)
            self.patch_size = patch_size
            self.in_channels = in_channels
            self.out_channels = out_channels

        def forward(
            self,
            tokens: torch.Tensor,
            t: torch.FloatTensor,
            x_t: torch.FloatTensor,
            image_token_indices: List[Tuple[int, int]],
            cfg_scale: float = 1.0,
            *,
            mask: Optional[_MaskType] = None,
            input_pos: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            batch_size = tokens.shape[0]
            
            # Split input for conditional and unconditional paths
            tokens_uncond, tokens_cond = torch.chunk(tokens, 2, dim=0)
            x_t_uncond, x_t_cond = torch.chunk(x_t, 2, dim=0)
            
            def process_batch(tokens, x_t):
                # Embed timestep
                t_emb = self.t_embedder(t)
                
                # Process input_ids and x_t
                h = self.tok_embeddings(tokens)
                
                x_t_embeds = self.image_embed(x_t.view(x_t.shape[0], -1, self.patch_size * self.patch_size * self.in_channels))
                for i, (start, end) in enumerate(image_token_indices):
                    h[i, start:end] = x_t_embeds[i]

                # Add timestep embedding to all token embeddings
                h += t_emb.unsqueeze(1)

                # Forward pass through Llama layers
                for layer in self.layers:
                    h = layer(
                        h,
                        mask=mask,
                        input_pos=input_pos,
                    )

                h = self.norm(h)

                # Process image tokens
                image_outputs = []
                for i, (start, end) in enumerate(image_token_indices):
                    image_tokens = h[i, start:end]
                    image_output = self.final_layer(image_tokens)
                    image_outputs.append(image_output)

                return torch.stack(image_outputs)

            # Process unconditional and conditional paths
            output_uncond = process_batch(tokens_uncond, x_t_uncond)
            output_cond = process_batch(tokens_cond, x_t_cond)

            # Apply classifier-free guidance
            output = output_uncond + cfg_scale * (output_cond - output_uncond)

            return output

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, num_patches, patch_size**2 * out_channels)
        imgs: (batch, out_channels, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(x.shape[0], h, w, p, p, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        imgs = x.reshape(x.shape[0], self.out_channels, h * p, h * p)
        return imgs
    
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, channels, height, width)
        patches: (batch, num_patches, patch_size**2 * channels)
        """
        p = self.patch_size
        assert x.shape[2] % p == 0 and x.shape[3] % p == 0
        
        b, c, h, w = x.shape
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        patches = x.reshape(b, (h // p) * (w // p), p * p * c)
        return patches
    
    model = Llama3_2WithDiffusion(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=(embed_dim // num_heads),
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
        patch_size=patch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=embed_dim,
    )

    return model

def llama3_mlp(dim: int, hidden_dim: int, quantize_base: bool = False) -> FeedForward:
    """
    Build the MLP layer associated with the Llama model.
    """
    gate_proj = nn.Linear(dim, hidden_dim, bias=False) if not quantize_base else FrozenNF4Linear(dim, hidden_dim, bias=False)
    down_proj = nn.Linear(hidden_dim, dim, bias=False) if not quantize_base else FrozenNF4Linear(hidden_dim, dim, bias=False)
    up_proj = nn.Linear(dim, hidden_dim, bias=False) if not quantize_base else FrozenNF4Linear(dim, hidden_dim, bias=False)
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)