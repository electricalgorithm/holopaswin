# Experiment 9: Large-Scale Training

**Date:** 2026-01-14

## Dataset Configuration

### Training Data
- **Path:** `../hologen/dataset-224`
- **Total Samples:** 25,000
- **Resolution:** 224×224 (native, no resize)
- **Format:** Parquet files (300 total)
- **Size:** 22 GB

### Noise Configurations (3,125 samples each)
1. No noise
2. Speckle noise
3. Shot noise
4. Speckle + Shot
5. Read noise
6. Dark current
7. Speckle + Shot + Read
8. Speckle + Shot + Read + Dark

### Test Data
- **Path:** `../hologen/test-dataset-224`
- **Samples:** 496
- **Seed:** 999999 (different from training)

## Physics Parameters
- **Wavelength:** 532 nm
- **Pixel Size:** 4.65 μm
- **Propagation Distance:** 20 mm
- **Bit Depth:** 12-bit

## Model Configuration
- **Architecture:** HoloPASWIN (Swin Transformer U-Net)
- **Input/Output:** 224×224
- **Channels:** 2 (Real, Imag)

## Training Configuration
- **Batch Size:** 32
- **Learning Rate:** 1e-4
- **Optimizer:** AdamW
- **Epochs:** 5 (with early stopping)
- **Split:** 80% train / 20% validation

## Loss Function
- **Amplitude L1:** 0.4
- **Phase L1:** 0.2
- **Frequency Loss:** 0.2
- **Complex L1:** 0.2
- **Physics Consistency:** λ=0.1

## Expected Improvements Over Exp 8
- Better generalization (3× more data)
- Improved noise robustness (diverse configurations)
- More stable convergence
