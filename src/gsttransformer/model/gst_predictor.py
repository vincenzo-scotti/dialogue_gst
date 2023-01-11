import torch.nn as nn

from transformers.models.gpt2 import GPT2Config

from typing import Optional, Tuple


class DGST(nn.Module):
    def __init__(
            self,
            config: GPT2Config,
            gst_embeds_size: Optional[int] = None,
            gst_scores_shape: Optional[Tuple[int, int]] = None
    ):
        super(DGST, self).__init__()
        # Hyper-parameters
        self.gst_embeds_size: Optional[int] = gst_embeds_size
        self.gst_scores_shape: Optional[Tuple[int, int]] = gst_scores_shape
        # GRU pooling block
        self.h = nn.GRU(config.hidden_size, config.hidden_size, dropout=config.embd_pdrop, batch_first=True)
        # Last layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Heads
        self.gst_embeds_head = nn.Linear(
            config.hidden_size, self.gst_embeds_size
        ) if self.gst_embeds_size is not None else None
        self.gst_scores_head = nn.Linear(
            config.hidden_size, self.gst_scores_shape[0] * self.gst_scores_shape[1]
        ) if self.gst_scores_shape is not None else None

    def forward(self, input_embeds, use_cache: Optional[bool] = False, **kwargs):
        # Get hidden vectors
        hidden_outputs, last_hidden_state = self.h(input_embeds)
        hidden_states = last_hidden_state.squeeze(0)
        # Apply last normalisation
        hidden_states = self.ln_f(hidden_states)

        # Compute outputs
        outputs = {
            'gst_embeds': self.gst_embeds_head(hidden_states) if self.gst_embeds_head is not None else None,
            'gst_scores': self.gst_scores_head(hidden_states).reshape(
                -1, *self.gst_scores_shape
            ) if self.gst_scores_head is not None else None,
            'cache': (hidden_outputs, last_hidden_state) if use_cache else None
        }

        return outputs
