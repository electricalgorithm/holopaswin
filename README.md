# HOLO-PASWIN: In-Line Holographical Physics-Aware SWIN Transformer

A research-grade deep learning project for eliminating the **twin-image problem in in-line holography** using a **physics-aware Swin-UNet architecture** trained with synthetic holograms generated via the Angular Spectrum Method.

Build a modern, high-performing, physics-aware neural architecture capable of reconstructing complex object fields ( $\hat{O} = \hat{O}_{real} + i\hat{O}_{imag}$ ) from inline holograms while robustly suppressing **twin-image artifacts**, **noise**, and **aberrations**.

## Development Setup

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

