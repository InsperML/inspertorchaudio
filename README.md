# inspertorchaudio

Common code for audio experiments with PyTorch at Insper

## Datasets

Check the [datasets page](docs/datasets.md).

## Developer

### Onboarding

#### 0. Install the `uv` package management tool:

  pip install uv

(or: browse to the [Astral website](https://docs.astral.sh/uv/getting-started/installation) and choose another installation method)

#### 1. Clone the repository:

  git clone git@github.com:InsperML/inspertorchaudio.git

#### 2. Syncronize and activate the environment

  cd inspertorchaudio
  uv sync
  source .venv/bin/activate


#### 3. Configure .env

  cp .env_template .env

After that, change the value of `DATA_DIR` to `"/mnt/data/inspertorchaudio/"` (or your favourite data location).

Optionally, just create a new `.env` file:

  echo DATA_DIR = "/mnt/data/inspertorchaudio/" > .env

#### 4. Tun tests to make sure everything works

  pytest -s

Everything should work.

If it is the first time you are using the data directory, some small datasets will automatically be downloaded. This should take between one and ninety minutes, depending on your connection speed.


### Coding rules and style

We are mostly following the Google Style Guide for Python (<https://google.github.io/styleguide/pyguide.html>), but we are not using `pylint` anymore, we are switching to `ruff`.

We are writing commit messages in a style inspired by Conventional Commits (<https://www.conventionalcommits.org/en/v1.0.0/>), but not strictly following it: we are sort of playing by ear in this manner.

### Dependencies and virtual environment

We are using `uv` (<https://docs.astral.sh/uv/>).

- Starting to work with this repo:

In the root directory of the repository, run:

```bash
uv venv
source .venv/bin/activate
uv sync
```

- Regular work after the previous initialization:

```bash
source .venv/bin/activate
```

- After `git pull`, to get new dependencies:

```bash
uv sync
```

- Adding a dependency:

  - If adding a true dependency of the library: `uv add <dependency>`

  - If adding a dependency which is a tool required only to develop the library itself: `uv add --group dev <dependency>`

- Removing a dependency:

  - Same as adding, changing `add` for `remove`

### Formatting and linting

We are using `ruff` (<https://docs.astral.sh/ruff/>).

The `pyproject.toml` file contains settings for line length (90 characters) and usage of single-quotes instead of double-quotes for strings.

It is recommended that you add the ruff extension on vscode.

- Formatting: `ruff format`

- Linting: `ruff check`

### Unit tests

We are using `pytest`.

Tests go into the `tests` folder, and they try to mirror the file structure of the `src` folder.

### Logging

As much as it is convenient, try to use the `logging` module of Python (<https://docs.python.org/3/howto/logging.html>).

In a nutshell:

```Python
import logging

logging.basicConfig(level=logging.DEBUG)

...

# A debug message.
logging.debug('Hello, world!')

```

More advanced logging can be done, e.g. logging to a file, having multiple loggers, etc. For the time being, lets focus on just simple logging. Proper usage of the logging levels is described in <https://docs.python.org/3/howto/logging.html#when-to-use-logging>
