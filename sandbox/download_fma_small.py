from datasets import load_dataset
import os

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
data_dir = os.getenv("DATA_DIR")
if not data_dir:
    raise EnvironmentError("DATA_DIR environment variable not set")

ds = load_dataset("rpmon/fma-genre-classification",
                  cache_dir=Path(data_dir) / "fma-small")