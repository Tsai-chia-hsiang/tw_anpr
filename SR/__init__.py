import os
from pathlib import Path

SR_FILE_DIR = Path(os.path.abspath(__file__)).parent
SR_PRETRAINED_DIR = SR_FILE_DIR/"pretrained"
SR_FINETUNE_DIR = SR_FILE_DIR/"finetune"