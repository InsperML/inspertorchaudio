# inspertorchaudio

Common code for audio experiments with PyTorch at Insper

## Developer

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
