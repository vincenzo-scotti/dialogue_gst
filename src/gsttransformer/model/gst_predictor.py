import torch
import torch.nn as nn

from transformers.models.gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP

from typing import Optional, Union, Tuple


class GSTTBlock(nn.Module):
    # This is a slight modification og GPT-2 block
    def __init__(self, n_out, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.n_out = n_out
        self.hidden_size = hidden_size

        # NOTE the first layer norm is omitted since there is the final one from the transformer block
        self.attn_pooling = GPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        # Prepare dummy decoder hidden states
        pooling_hidden_states = torch.ones(
            (hidden_states.size(0), self.n_out, self.hidden_size), dtype=hidden_states, device=hidden_states.device
        )
        # Apply attention pooling as a cross attention
        attn_pooling_outputs = self.attn_pooling(
            pooling_hidden_states,
            head_mask=head_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        attn_output = attn_pooling_outputs[0]
        # NOTE the residual connection is omitted since we are pooling
        hidden_states = attn_output
        outputs = attn_pooling_outputs[1:]  # add attention pooling outputs if we output attention weights # TODO check if correct

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)


class GSTTransformer(nn.Module):
    def __init__(
            self,
            config: GPT2Config,
            gst_embeds_size: Optional[int] = None,
            gst_scores_shape: Optional[Tuple[int, int]] = None
    ):
        super(GSTTransformer, self).__init__()
        # Hyper-parameters
        self.gst_embeds_size: Optional[int] = gst_embeds_size
        self.gst_scores_shape: Optional[Tuple[int, int]] = gst_scores_shape
        # Attention pooling block
        self.h = GSTTBlock(int() + int(), config)
        # Last layer norm
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        # Heads
        self.gst_embeds_head = nn.Linear(
            config.hidden_size, self.gst_embeds_size
        ) if self.gst_embeds_size is not None else None
        self.gst_scores_head = nn.Linear(
            config.hidden_size, sum(self.gst_scores_shape)
        ) if self.gst_scores_shape is not None else None

    def forward(self, input_embeds, attention_mask, use_cache: Optional[bool] = False, **kwargs):
        # Get hidden vectors
        hidden_outputs = self.h(input_embeds, attention_mask=attention_mask, use_cache=use_cache, **kwargs)
        hidden_states = hidden_outputs[0].squeeze(1)
        # Apply last normalisation
        hidden_states = self.ln_f(hidden_states)
        # Compute outputs
        outputs = {
            'gst_embeds': self.gst_embeds_head(hidden_states[:, 0]) if self.gst_embeds_head is not None else None,
            'gst_scores': self.gst_embeds_head(hidden_states[:, -1]).reshape(
                -1, *self.gst_scores_shape
            ) if self.gst_scores_head is not None else None,
            'cache': hidden_outputs if use_cache else None
        }

        return outputs
