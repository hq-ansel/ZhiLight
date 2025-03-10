# coding=utf-8


from .dev_config import (
    ROPE_CACHE,
    set_env,
)


class LLaMAAdapter: # pylint: disable=missing-class-docstring
    @staticmethod
    def adapt(config: dict): # pylint: disable=missing-function-docstring
        if config["rope_scaling"].get("rope_type", None) == "llama3":
            set_env(ROPE_CACHE, 1)
