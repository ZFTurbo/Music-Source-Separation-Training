"""VitMelRoFormer — Vision Transformer style Mel-band RoFormer for music source separation."""

from .bsroformer import BSRoformer

# Alias the ViT-style mel roformer under the requested name.
VitMelRoformer = BSRoformer

__all__ = ["VitMelRoformer", "BSRoformer", "BandSplit"]
