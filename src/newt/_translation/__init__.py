"""newt._translation — action-format translation layer.

Sits between the policy server's output and the edge client's input.
Invariant client + invariant openpi wire format means the translation
happens here, not in either endpoint.
"""
from newt._translation.widowx_aloha import pad_widowx_to_aloha, slice_aloha_to_widowx

__all__ = ["pad_widowx_to_aloha", "slice_aloha_to_widowx"]
