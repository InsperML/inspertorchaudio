# inspertorchaudio

Common code for audio experiments with PyTorch at Insper

## Developer

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
