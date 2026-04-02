import argparse

class JIUGEArgument:
    """
    Manager for JIUGE specific CLI arguments.
    Encapsulates parameter definition and normalization logic.
    """
    
    @staticmethod
    def _normalize_pos_type(val: str) -> str:
        """Helper to convert user input (e.g., 'ChatGLM-Rotary') to internal 'chatglm_rotary'."""
        if val is None:
            return 'relative'
        return val.lower().replace("-", "_")

    @classmethod
    def add_jiuge_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add JIUGE specific CLI arguments to the provided parser.
        """
        group = parser.add_argument_group(title='JIUGE FM9G Arguments')

        # Position Bias Type with normalization and choices
        group.add_argument(
            '--pos-bias-type', 
            type=cls._normalize_pos_type, 
            choices=['relative', 'rotary', 'chatglm_rotary'],
            default='relative',
            help='Position bias type. Options: relative, rotary, chatglm-rotary. '
                 'Inputs are case-insensitive and support both "-" and "_".'
        )

        # Skip logic flags
        group.add_argument(
            '--mask-att',
            action='store_true',
            help='If set, the self-attention block in Transformer layers will be skipped.'
        )

        group.add_argument(
            '--mask-ffn',
            action='store_true',
            help='If set, the FFN block in Transformer layers will be skipped.'
        )

        # Future-proofing: add more JIUGE-specific args here
        # group.add_argument('--jiuge-custom-flag', ...)

        return parser