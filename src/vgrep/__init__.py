"""vgrep - semantic search over local images."""

import os

# Set before any submodule (and therefore torch/transformers/huggingface_hub)
# is imported. Doing this inside encoder.py is too late: huggingface_hub reads
# these at its own import time and emits an auth warning on every invocation.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
import warnings

warnings.filterwarnings("ignore")
for _name in ("transformers", "huggingface_hub", "torch"):
    logging.getLogger(_name).setLevel(logging.ERROR)

__version__ = "0.1.0.dev0"
