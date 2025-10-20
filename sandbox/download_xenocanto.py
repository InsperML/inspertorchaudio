from datasets import load_dataset
import os
from huggingface_hub import login
login()
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
data_dir = os.getenv("DATA_DIR")
token = os.getenv("HF_TOKEN")
if not data_dir:
    raise EnvironmentError("DATA_DIR environment variable not set")

#ds = load_dataset("ilyassmoummad/Xeno-Canto-6s-16khz",
#                  cache_dir=Path(data_dir) / "xenocanto_6_16khz")

from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ilyassmoummad/Xeno-Canto-6s-16khz",
    cache_dir=Path(data_dir) / "xenocanto_6_16khz",
    repo_type="dataset",
    token=token,
    resume_download=True,
)