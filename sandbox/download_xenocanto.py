from datasets import load_dataset
import os

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
data_dir = os.getenv("DATA_DIR")
if not data_dir:
    raise EnvironmentError("DATA_DIR environment variable not set")

ds = load_dataset("ilyassmoummad/Xeno-Canto-6s-16khz",
                  cache_dir=Path(data_dir) / "xenocanto_6_16khz")