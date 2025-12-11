# HoloPASWIN v2

Physics-Aware Swin Transformer for eliminating twin-image artifacts in in-line holography.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

1.  **Install uv** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Sync Dependencies**:
    Navigate to the `holopaswin` directory and run:
    ```bash
    uv sync
    ```
    This will create a virtual environment and install all locked dependencies from `uv.lock`.

3.  **Training**:
    To start training, run:
    ```bash
    uv run src/train.py
    ```

## Development

-   **HOLO-PASWIN v2** builds upon Swin Transformer U-Net architecture.
-   It accepts Hologram Intensity and outputs Complex Object (Phase & Amplitude).
-   Dataset is loaded from efficient Parquet files.

### Installing Git Hooks

This repository includes pre-commit hooks that automatically run code quality checks (ruff and mypy) before each commit. To install them:

```bash
./scripts/install-hooks.sh
```

This will set up the hooks from `.githooks/` to `.git/hooks/`. The hooks will:
- Run `ruff check` on `src/`
- Run `ruff format --check` on `src/`
- Run `mypy --strict` on `src/`

If any check fails, the commit will be blocked. You can bypass the hook with `git commit --no-verify` (not recommended).

