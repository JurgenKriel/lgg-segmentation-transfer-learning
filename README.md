# LGG Segmentation via Transfer Learning from GBM

Transfer learning approach for Low-Grade Glioma (LGG) histopathology segmentation using a pre-trained Glioblastoma (GBM) model. This repository documents the iterative development of techniques to detect rare tissue classes (Leading Edge and Infiltrating Tumor) in H&E-stained LGG slides.

## Overview

This project fine-tunes a pre-trained 8-class GBM segmentation model (GBM_WSSM) to classify 4 LGG tissue types:
- **Background** (class 0)
- **CT** - Cellular Tumor (class 1)
- **LE** - Leading Edge (class 2)
- **IT** - Infiltrating Tumor (class 3)

### Key Challenge

Severe class imbalance in the LGG dataset:
- Background: 45.6%
- CT: 37.1%
- **LE: 11.6%** (rare)
- **IT: 5.7%** (very rare)

Initial transfer learning attempts achieved 96% overall accuracy but **0% recall** on rare classes (LE/IT).

## Repository Contents

### Training Scripts

- **`gbm_fine_tune_progressive.py`** - LE-focused training with progressive layer unfreezing (3× LE oversampling)
- **`gbm_fine_tune_extreme_oversample.py`** - Extreme oversampling approach (IT: 20×, LE: 5×) with progressive unfreezing
- **`gbm_fine_tune_fully_unfrozen.py`** - Fully unfrozen training with extreme oversampling

### Data Preparation

- **`generate_crop_masks_with_tissue_detection.py`** - Creates tissue-aware masks for cropped region images using HSV thresholding

### Analysis & Visualization

- **`analyze_final_model.py`** - Comprehensive model evaluation with confusion matrices, probability distributions, and per-class metrics
- **`visualize_final.py`** - Generate prediction visualizations overlaid on validation images

### SLURM Scripts

- **`train_progressive.sh`** - Submit LE-focused progressive training job
- **`train_extreme_oversample.sh`** - Submit extreme oversampling training job
- **`train_fully_unfrozen.sh`** - Submit fully unfrozen training job
- **`run_viz_after_job.sh`** - Auto-run visualization after training completion

### Documentation

- **`LE_FOCUSED_APPROACH.md`** - Documents the breakthrough from 0% to 22% LE recall
- **`EXTREME_OVERSAMPLING_APPROACH.md`** - Details the extreme oversampling strategy for IT detection

## Methodology Evolution

### Phase 1: LE-Focused Training (Baseline → 22% LE Recall)

**Problem**: Despite 96% overall accuracy, LE/IT detection was 0%

**Solution**:
1. **3× LE oversampling** - LE-containing images appear 3 times per epoch
2. **Tissue-aware masks** - HSV-based tissue detection eliminates background mislabeling
3. **Progressive unfreezing** - 4 phases (head → 10 → 25 → 50 layers)
4. **Stain normalization** - Macenko method bridges GBM-LGG domain gap

**Results**:
- LE recall: **0% → 22%**
- LE max probability: **0.26 → 0.79**
- IT recall: Still **0%** (max prob 0.34)

### Phase 2: Extreme Oversampling (Targeting IT)

**Problem**: IT max probability never exceeded 0.34 - fundamental learning failure

**Solution**:
1. **20× IT oversampling** - IT images appear 20 times per epoch
2. **5× LE oversampling** - Increased from 3× to push further
3. **Parallel experiments**:
   - Progressive unfreezing (same as Phase 1)
   - Fully unfrozen (all 9.4M parameters trainable)

**Rationale**: Different rare classes need different strategies. IT needed far more aggressive intervention than LE.

## Key Technical Components

### Architecture

```python
# Preserved 8-class GBM encoder
original_model = build_GBM_WSSM_Model(nb_classes=8)
original_model.load_weights('GBM_WSSM.h5')

# Add 8→4 class mapping layer
feature_output = original_model.layers[-3].output
x = Conv2D(16, (1, 1), activation='relu')(feature_output)
x = Conv2D(4, (1, 1), name='lgg_4class_conv')(x)
x = Reshape((-1, 4))(x)
x = Activation('softmax')(x)
```

**Why preserve 8 classes?**: GBM features (necrosis, vascular patterns) provide valuable information for detecting LGG region boundaries.

### Data Generator with Differential Oversampling

```python
class LEFocusedDataGenerator:
    def __init__(self, it_oversample_rate=20.0, le_oversample_rate=5.0):
        # Categorize images by rarest class present
        for idx, mask in enumerate(masks):
            if has_it(mask):
                self.it_images.append(idx)
            elif has_le(mask):
                self.le_images.append(idx)
            else:
                self.other_images.append(idx)
        
        # Oversample rare classes
        it_repeated = self.it_images * int(it_oversample_rate)
        le_repeated = self.le_images * int(le_oversample_rate)
        self.indices = self.other_images + le_repeated + it_repeated
```

### Tissue Detection for Cropped Regions

```python
def detect_tissue_mask(image_array):
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    
    # Tissue has color (high saturation) and isn't too bright
    tissue_mask = (saturation > 15) & (value < 240)
    
    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    tissue_mask = cv2.morphologyEx(tissue_mask.astype(np.uint8), 
                                    cv2.MORPH_CLOSE, kernel)
    tissue_mask = cv2.morphologyEx(tissue_mask, 
                                    cv2.MORPH_OPEN, kernel)
    
    return tissue_mask.astype(bool)
```

## Usage

### Prerequisites

```bash
module load CUDA/12.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib/python3.11/site-packages/nvidia/cudnn/lib
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/stornext/System/data/software/rhel/9/base/nvidia/CUDA/12.1
```

### 1. Generate Tissue-Aware Masks

```bash
python generate_crop_masks_with_tissue_detection.py
```

### 2. Train Models

**LE-Focused (Progressive)**:
```bash
sbatch train_progressive.sh
```

**Extreme Oversampling (Progressive)**:
```bash
sbatch train_extreme_oversample.sh
```

**Fully Unfrozen**:
```bash
sbatch train_fully_unfrozen.sh
```

### 3. Analyze Results

```bash
python analyze_final_model.py
```

### 4. Visualize Predictions

```bash
sbatch run_viz_after_job.sh
```

## Results Summary

### LE-Focused Training

| Metric | Baseline | Phase 4 Result | Change |
|--------|----------|----------------|--------|
| LE Recall | 0% | 22% | ↑ 22pp |
| LE Max Prob | 0.26 | 0.79 | +204% |
| IT Recall | 0% | 0% | No change |

### Extreme Oversampling (In Progress)

Training with IT: 20× oversampling, LE: 5× oversampling  
Expected: IT recall >10%, LE recall >40%

## Key Insights

1. **High overall accuracy can mask rare class failure** - 96% accuracy with 0% rare class recall
2. **Probability analysis is critical** - Revealed IT never exceeded 0.5 probability
3. **Class imbalance requires aggressive intervention** - 3× insufficient, needed 20× for IT
4. **Mask quality matters more than quantity** - Tissue-aware masks dramatically improved training
5. **Different rare classes need different strategies** - LE responded to 3×, IT needs 20×
6. **Preserve pre-trained features** - 8-class GBM encoder provides valuable features for LGG

## Dependencies

- TensorFlow 2.x with Keras
- OpenCV (cv2)
- NumPy
- Matplotlib
- scikit-learn
- staintools (for Macenko normalization)

## Citation

If you use this code, please cite:

```
[Add your citation information here]
```

## License

[Add license information]

## Contact

Jurgen Kriel - https://github.com/JurgenKriel

---

**Last Updated**: December 2024
