# Extreme Oversampling Approach for IT/LE Detection

## Problem Statement

### Initial Results Analysis (Progressive Unfreezing - Phase 4)

After implementing progressive unfreezing with tissue-aware masks, stain normalization, and 8-class GBM preservation, we achieved:

**Overall Accuracy**: 64.25%

**Per-Class Performance**:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Background | 0.7783 | 0.7901 | 0.7841 | 57.9M pixels |
| CT | 0.5963 | 0.6924 | 0.6408 | 47.1M pixels |
| **LE** | **0.2364** | **0.2167** | **0.2261** | **14.7M pixels** |
| **IT** | **0.0545** | **0.0000** | **0.0000** | **7.2M pixels** |

### Critical Issues Identified

#### 1. IT Detection Failure
**Probability Analysis:**
- IT mean probability: **0.0986** (extremely low)
- IT max probability: **0.3411** (barely above threshold)
- **0% of IT pixels have probability > 0.5**
- Only **3 pixels** out of 7.2 million correctly predicted as IT

**Confusion Pattern:**
- 61% of IT pixels predicted as CT
- 26.5% of IT pixels predicted as Background
- 12.6% of IT pixels predicted as LE

**Root Cause**: Model fundamentally not learning IT features despite 50 layers unfrozen

#### 2. LE Detection - Modest Success
**Probability Analysis:**
- LE mean probability: 0.3269
- LE max probability: 0.7860
- 75.6% of LE pixels have probability > 0.3
- LE recall: **22%** (improvement from 0%)

**Confusion Pattern:**
- 69.4% of LE pixels predicted as CT (still high)
- 21.7% correctly predicted as LE
- 8.9% predicted as Background

**Root Cause**: Model learning LE features but they're overwhelmed by CT

#### 3. Class Imbalance in Dataset

**Ground Truth Distribution:**
```
Background: 57.9M pixels (45.62%)
CT:         47.1M pixels (37.12%)
LE:         14.7M pixels (11.55%)
IT:          7.2M pixels ( 5.71%)
```

**Prediction Distribution:**
```
Background: 58.8M pixels (46.31%)
CT:         54.7M pixels (43.10%)
LE:         13.4M pixels (10.59%)
IT:             55 pixels ( 0.00%) ← CRITICAL FAILURE
```

The model is essentially ignoring IT class entirely and strongly biased toward CT for both LE and IT regions.

## Solution: Extreme Oversampling Strategy

### Rationale

Since IT probabilities are fundamentally too low (max 0.34), this is **not** a thresholding problem - it's a learning problem. The model needs to see IT examples far more frequently during training.

### Approach Overview

**Goal**: Force the model to learn IT features by dramatically increasing their representation in training

**Method**: Differential oversampling based on class rarity and performance

### Implementation Details

#### 1. Enhanced Data Generator

Modified `LEFocusedDataGenerator` to track three image types:
- **IT images**: Images containing any IT pixels (class 3)
- **LE images**: Images containing LE pixels but no IT (class 2)
- **Other images**: Background/CT only images

**Priority hierarchy**: IT > LE > Others (prioritize rarest, hardest-to-learn classes)

#### 2. Oversampling Rates

```python
IT_OVERSAMPLE_RATE = 20.0  # IT images appear 20× per epoch
LE_OVERSAMPLE_RATE = 5.0   # LE images appear 5× per epoch  
```

**Impact on Training**:
- If dataset has ~15 IT images → 300 appearances per epoch
- If dataset has ~30 LE images → 150 appearances per epoch
- Total samples per epoch: ~894 (vs ~140 without oversampling)

**Rationale for rates**:
- IT needs 20× because recall is 0% (complete failure)
- LE needs 5× because recall is 22% (partial success)
- These rates balance the training to give rare classes equal learning opportunity

#### 3. Architecture

**Base Model**: GBM_WSSM 8-class encoder
- Preserved all 8 GBM classes (necrosis, vascular, perinecrotic features)
- Frozen pre-trained weights initially
- Added learnable 8→4 class mapping layer

**Progressive Unfreezing Strategy**:
```
Phase 1: Head only (0 layers unfrozen)    - 30 epochs, LR=1e-3
Phase 2: Last 10 layers                    - 30 epochs, LR=1e-4
Phase 3: Last 25 layers                    - 30 epochs, LR=1e-5
Phase 4: Last 50 layers                    - 30 epochs, LR=1e-5
```

Total trainable params at Phase 4: ~9.4M (up from 4,180 in Phase 1)

#### 4. Additional Techniques

**Stain Normalization** (Macenko method):
- Normalizes H&E color consistency between GBM and LGG datasets
- Converts RGB → Optical Density → Normalize stain concentrations
- Applied during data loading

**Tissue-Aware Masks**:
- For cropped region images (e.g., `*_LE1_preprocessed.png`)
- Uses HSV thresholding to detect tissue vs background
- Only tissue pixels labeled with region class (CT/LE/IT)
- Background pixels correctly labeled as class 0

**Focal Loss**:
```python
gamma = 2.0   # Focus on hard examples
alpha = 0.25  # Class balance weight
```

**Data Augmentation**:
- Random hue/saturation/brightness/contrast (H&E variation)
- Horizontal/vertical flips
- 90° rotations

## Expected Outcomes

### Success Metrics

**Minimum Success**:
- IT recall > 10% (currently 0%)
- IT max probability > 0.5 (currently 0.34)
- IT predictions > 1,000 pixels (currently 3)

**Good Success**:
- IT recall > 30%
- IT mean probability > 0.3
- LE recall > 40% (currently 22%)

**Excellent Success**:
- IT recall > 50%
- IT F1-score > 0.40
- LE recall > 60%

### Why This Should Work

1. **Frequency = Learning**: Deep learning models learn what they see most
   - Current: IT images seen 1× per epoch → model ignores them
   - New: IT images seen 20× per epoch → model forced to learn them

2. **Balanced Effective Distribution**:
   - Without oversampling: BG(46%) > CT(37%) >> LE(12%) >> IT(6%)
   - With oversampling: IT ≈ LE ≈ CT ≈ BG (more balanced)

3. **Progressive Adaptation**:
   - Phase 1: Quick mapping layer learns class distributions
   - Phases 2-4: Deeper layers adapt GBM features to LGG tissue patterns

4. **Preserved GBM Knowledge**:
   - Necrosis/vascular features from GBM help identify LE/IT boundaries
   - Small learning rates (1e-4, 1e-5) prevent catastrophic forgetting

## Parallel Experiment: Fully Unfrozen

Running simultaneously with extreme oversampling for comparison:

**Setup**:
- ALL layers trainable (9.4M parameters)
- LR = 1e-5 (very small)
- Same extreme oversampling (IT: 20×, LE: 5×)
- 100 epochs max

**Rationale**: 50 layers unfrozen may not be enough - try maximum flexibility

**Risk**: May lose GBM knowledge, but IT detection so poor it's worth the risk

## Files and Commands

### Training Scripts
```bash
# Extreme oversampling + progressive unfreezing
python gbm_fine_tune_extreme_oversample.py
sbatch train_extreme_oversample.sh  # Job 24553570

# Extreme oversampling + fully unfrozen
python gbm_fine_tune_fully_unfrozen.py
sbatch train_fully_unfrozen.sh      # Job 24553575
```

### Analysis
```bash
# Analyze model performance
python analyze_final_model.py

# Visualize predictions
python visualize_final.py
```

### Model Locations
- **Previous best**: `~/gbm_progressive_phase_4:_last_50_layers.keras`
- **Extreme oversample**: `~/gbm_progressive_phase_4:_last_50_layers.keras` (overwritten)
- **Fully unfrozen**: `~/gbm_fully_unfrozen.keras`

## Key Insights from This Iteration

1. **Transfer learning has limits**: GBM→LGG may be too different for IT detection
2. **Probability analysis is critical**: High accuracy (64%) masked 0% IT recall
3. **Class imbalance must be addressed aggressively**: 3× oversampling was insufficient
4. **Tissue-aware masking matters**: Background pixels were skewing training
5. **LE improvement (0%→22%) shows method works**: Need to apply same to IT

## Next Steps After Training

1. **Run analysis** on both models:
   ```bash
   python analyze_final_model.py
   ```

2. **Compare IT recall**:
   - Previous: 0.00%
   - Extreme oversample: ???
   - Fully unfrozen: ???

3. **If IT recall still < 10%**: Consider Option 2 (train from scratch on LGG only)

4. **If IT recall > 30%**: Fine-tune with custom class weights or thresholds

## Conclusion

This extreme oversampling approach represents a targeted solution to the specific failure mode identified in analysis: **the model is not learning IT features at all**. By ensuring IT examples appear 20× more frequently, we give the model no choice but to learn these patterns. The parallel fully-unfrozen experiment provides a backup strategy if 50 layers of unfreezing still isn't enough adaptation.

---

**Date**: December 4, 2025  
**Experiment**: Extreme IT/LE Oversampling + Progressive/Full Unfreezing  
**Previous Baseline**: IT recall 0%, LE recall 22%  
**Target**: IT recall >10%, LE recall >40%
