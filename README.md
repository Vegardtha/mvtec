# MVTec Anomaly Detection Project

A comprehensive implementation and evaluation framework for industrial anomaly detection methods on the MVTec AD dataset. This project includes implementations of multiple state-of-the-art anomaly detection algorithms and provides standardized evaluation tools.

##  Overview

This project implements and compares various anomaly detection methods for industrial quality control, specifically designed for the MVTec Anomaly Detection dataset. The framework includes:

- **Multiple Algorithm Implementations**: Autoencoder, PatchCore, and DRAEM
- **Comprehensive Evaluation Tools**: Standardized metrics including AUROC, PRO scores
- **Modular Design**: Easy to extend with new methods
- **Production-Ready Code**: Well-documented and tested implementations

## Project Structure

```
mvtec_anomaly_detection_project/
├── README.md                          # This file
├── requirement.txt                    # Python dependencies
├── notebooks/                         # Jupyter notebooks for different methods
│   ├── Autoencoder/                   # Autoencoder implementation
│   │   ├── Autoencoder_training.ipynb  # Training notebook
│   │   ├── Autoencoder_testing.ipynb   # Testing and evaluation
│   │   ├── autoencoder.py              # Autoencoder model definition
│   │   └── checkpoints/                # Saved model weights
│   ├── Anomalib/                      # Anomalib framework integration
│   └── patchCore/                     # PatchCore implementation
├── evaluate/                          # Evaluation scripts and tools
│   ├── evaluate_experiment.py         # Single experiment evaluation
│   ├── evaluate_multiple_experiments.py # Batch evaluation
│   ├── pro_curve_util.py              # PRO curve computation
│   ├── roc_curve_util.py              # ROC curve computation
│   ├── generic_util.py                # Utility functions
│   └── patchCore_Eval/                # PatchCore-specific evaluation
├── mvtec_anomaly_detection/           # MVTec AD dataset
│   ├── bottle/                        # Individual object categories
│   ├── cable/
│   ├── capsule/
│   └── ...                           # 15 categories total
├── output/                            # Generated outputs
│   ├── anomaly_maps/                  # Generated anomaly maps
│   └── metrics/                       # Evaluation results
├── patchcore-inspection/              # Official PatchCore implementation
├── Draem_implement.ipynb              # DRAEM implementation notebook
└── Patchcore_implement.ipynb         # PatchCore implementation notebook
```

##  Getting Started

### Prerequisites

- Python 3.8+


### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mvtec_anomaly_detection_project
   ```

2. **Install dependencies**
   ```bash
   # Install core dependencies
pip install -r requirements.txt

# For GPU acceleration (optional)
pip install faiss-gpu

# For experiment tracking (optional)
pip install wandb
   ```

3. **Download MVTec AD Dataset**
   - Download from [MVTec AD official website](https://www.mvtec.com/company/research/datasets/mvtec-ad)
   - Extract to `mvtec_anomaly_detection/` directory

### Quick Start

1. **Train an Autoencoder model**
   ```bash
   jupyter notebook notebooks/Autoencoder/Autoencoder_training.ipynb
   ```

2. **Test and evaluate**
   ```bash
   jupyter notebook notebooks/Autoencoder/Autoencoder_testing.ipynb
   ```


##  Implemented Methods

### 1. Autoencoder-Based Detection
- **Location**: `notebooks/Autoencoder/`
- **Architecture**: Enhanced 5-layer convolutional autoencoder with skip connections
- **Features**:
  - 128-dimensional latent space with adaptive global average pooling
  - ImageNet normalization for optimal feature extraction
  - Comprehensive AUROC metrics (Instance, Full Pixel, Anomaly Pixel)
  - Per-pixel MSE anomaly scoring

### 2. PatchCore
- **Location**: `patchcore-inspection/` and `notebooks/patchCore/`
- **Description**: Memory-efficient patch-based anomaly detection
- **Features**:
  - Pre-trained CNN feature extraction
  - Coreset subsampling for efficiency
  - Nearest neighbor-based anomaly scoring
  - State-of-the-art performance (99.6% Instance AUROC)

### 3. DRAEM (Discriminatively Trained Reconstruction Embedding)
- **Location**: `Draem_implement.ipynb`
- **Description**: Advanced reconstruction-based method with discriminative training

## 📊 Evaluation Metrics

This project implements comprehensive evaluation metrics following industry standards:

### Image-Level Metrics
- **Instance AUROC**: Image-level anomaly detection performance
- **Classification ROC**: Binary classification performance

### Pixel-Level Metrics
- **Full Pixel AUROC**: Pixel-level performance across all images
- **Anomaly Pixel AUROC**: Pixel-level performance on anomalous images only
- **PRO (Per-Region Overlap)**: Localization-aware performance metric

### Usage Example
```python
# Compute all AUROC metrics
instance_auroc = compute_imagewise_retrieval_metrics(scores, labels)["auroc"]
full_pixel_auroc = compute_pixelwise_retrieval_metrics(anomaly_maps, gt_masks)["auroc"]
anomaly_pixel_auroc = compute_pixelwise_retrieval_metrics(anomaly_maps_filtered, gt_masks_filtered)["auroc"]
```

## 🔧 Configuration

### Model Parameters

**Autoencoder Configuration:**
```python
IMG_HEIGHT = 224
IMG_WIDTH = 224
LATENT_DIM = 128
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
```

**PatchCore Configuration:**
```bash
--pretrain_embed_dimension 1024
--target_embed_dimension 384
--anomaly_scorer_num_nn 1
--patchsize 3
-p 0.01  # Coreset percentage
```

### Evaluation Settings
```bash
--pro_integration_limit 0.3  # PRO curve integration limit
--resize 366 --imagesize 320  # Image preprocessing
```


### Custom Model Implementation

1. **Create model class**
   ```python
   class CustomAnomalyDetector(tf.keras.Model):
       def __init__(self):
           # Implementation
           pass
   ```

2. **Generate anomaly maps**
   ```python
   # Follow the structure in notebooks/Autoencoder/Autoencoder_testing.ipynb
   # Save as TIFF files in output/anomaly_maps/
   ```

3. **Evaluate using standard tools**
   ```bash
   python evaluate/evaluate_experiment.py --dataset_base_dir ... --anomaly_maps_dir ...
   ```

### Batch Processing

For processing multiple experiments:
```bash
python evaluate/evaluate_multiple_experiments.py \
  --experiment_configs experiment_configs.json
```

### Custom Evaluation

```python
from evaluate.pro_curve_util import compute_pro
from evaluate.roc_curve_util import compute_classification_roc

# Custom evaluation pipeline
au_pro, au_roc, pro_curve, roc_curve = calculate_au_pro_au_roc(
    gt_filenames, prediction_filenames, integration_limit=0.3
)
```

##  File Formats

### Anomaly Maps
- **Format**: TIFF (.tiff)
- **Type**: Float32
- **Structure**: `<method>/<category>/test/<defect_type>/<image_id>.tiff`
- **Values**: Real-valued anomaly scores (higher = more anomalous)

### Ground Truth
- **Format**: PNG (.png)
- **Type**: Grayscale binary masks
- **Values**: 0 (normal), 255 (anomalous)

### Results
- **Format**: JSON
- **Content**: Comprehensive metrics per category and overall means

##  Troubleshooting


1. **Memory Issues**
   - Reduce batch size
   - Use smaller image resolution
   - Enable mixed precision training

2. **CUDA Out of Memory**
   ```python
   # Enable memory growth
   gpus = tf.config.experimental.list_physical_devices('GPU')
   tf.config.experimental.set_memory_growth(gpus[0], True)
   ```

3. **Evaluation Script Errors**
   - Ensure anomaly maps match ground truth dimensions
   - Check file naming conventions
   - Verify TIFF format compatibility

### Debug Mode
```bash
export CUDA_VISIBLE_DEVICES=0
python -m pdb evaluate/evaluate_experiment.py --args...
```



##  License

This project incorporates code from multiple sources:
- PatchCore implementation: See `patchcore-inspection/LICENSE`
- MVTec evaluation scripts: See `evaluate/LICENSE.txt`
- Original implementations: MIT License

##  References

1. **PatchCore**: Roth et al. "Towards Total Recall in Industrial Anomaly Detection" (2021)
2. **MVTec AD**: Bergmann et al. "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection" (2019)
3. **DRAEM**: Zavrtanik et al. "Reconstruction by inpainting for visual anomaly detection" (2021)

