<div align="center">

# HoloPASWIN

**Robust Inline Holographic Reconstruction via Physics-Aware Swin Transformers**

[![Paper](https://img.shields.io/badge/arXiv-2603.04926-b31b1b.svg)](https://arxiv.org/abs/2603.04926)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/gokhankocmarli/holopaswin-v3)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/gokhankocmarli/inline-digital-holography-v3)
[![Website](https://img.shields.io/badge/Project-Website-blue)](https://electricalgorithm.github.io/holopaswin/)

<img src="./docs/images/comparison.png" width="800">

*HoloPASWIN recovers clean phase and amplitude mappings from a single intensity hologram, directly eliminating twin-image artifacts.*
</div>

## 📌 At a Glance (The 5-Second Summary)

- **Problem:** Inline digital holography is highly effective but suffers from the **twin-image artifact**, which overlays an out-of-focus duplicate on the reconstructed image.
- **Solution:** **HoloPASWIN** brings the Swin Transformer's global attention to holographic imaging, operating inside a U-Net architecture. By leveraging forward physics models (Angular Spectrum Method), it learns to inherently correct and remove these twin-images.
- **Result:** State-of-the-art phase recovery and high structural fidelity, remaining highly robust across varying noise and distance configurations.

---

## 🏗 Network Architecture

<img src="./docs/images/architecture.png" width="800">

HoloPASWIN effectively processes continuous global diffraction patterns through hierarchical shifted-window attention. It utilizes both frequency-domain constraints and a novel unsupervised physics-based residual loss to guarantee physically consistent reconstructions.

---

## 💻 Installation & Usage

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.

1.  **Install uv** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Sync Dependencies**:
    Navigate to the `holopaswin` directory and run:
    ```bash
    uv sync
    ```
    This creates a virtual environment and installs all locked dependencies from `uv.lock`.

3.  **Training**:
    To start training the model, run:
    ```bash
    uv run src/train.py
    ```

### Development

This repository includes pre-commit hooks that automatically run code quality checks (`ruff` and `mypy`) before each commit. To install them:

```bash
./scripts/install-hooks.sh
```

If any check fails, the commit will be blocked. You can bypass the hook with `git commit --no-verify` (not recommended).

---

## 📖 Citation

If you find this code, dataset, or model useful for your research, please cite our paper:

```bibtex
@misc{koçmarlı2026holopaswinrobustinlineholographic,
      title={HoloPASWIN: Robust Inline Holographic Reconstruction via Physics-Aware Swin Transformers}, 
      author={Gökhan Koçmarlı and G. Bora Esmer},
      year={2026},
      eprint={2603.04926},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2603.04926}, 
}
```
