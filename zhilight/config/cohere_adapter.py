# coding=utf-8
import os
from .dev_config import (
    FUSE_QKV,
    DUAL_STREAM,
)


class CohereAdapter: # pylint: disable=missing-class-docstring
    @staticmethod
    def adapt(config: dict): # pylint: disable=missing-function-docstring
        config["eps"] = config["layer_norm_eps"]
        if 'tie_lm_head' not in config:
            config["tie_lm_head"] = True
        if config.get("use_qk_norm", False):
            os.environ[FUSE_QKV] = "0"  # can't fuse because qk norm
        os.environ[DUAL_STREAM] = "0"  # EncoderLayer is different from LlaMA
        os.environ["DEQUANT_DESC_ACT"] = "1"  # dequant attn_out to speedup
