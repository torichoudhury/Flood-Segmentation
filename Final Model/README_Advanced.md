# Advanced Flood Segmentation Models

This directory contains advanced implementations of flood segmentation models that utilize ALL the sophisticated features declared in the original `model.py` file.

## 📋 Overview

The original `train_optimized.py` uses a simple DeepLabV3+ model, while the advanced implementations in this directory leverage all the sophisticated components like CBAM attention, ASPP, dual-branch decoders, and transformer encoders.

## 🏗️ Architecture Comparison

| Feature                  | Original Model | Advanced Model | Lightweight Model |
| ------------------------ | -------------- | -------------- | ----------------- |
| **Backbone**             | ResNet-34      | ResNet-50      | ResNet-34         |
| **Parameters**           | 22.4M          | 37.6M          | 25.6M             |
| **Trainable Parameters** | 22.4M          | 29.1M          | 4.3M              |
| **CBAM Attention**       | ❌             | ✅             | ✅                |
| **ASPP Multi-scale**     | ❌             | ✅             | ✅                |
| **Dual Decoder**         | ❌             | ✅             | ❌                |
| **Self-Attention**       | ❌             | ✅             | ❌                |
| **Advanced Loss**        | Basic          | Enhanced       | Enhanced          |
| **Inference Time**       | 32ms           | 100ms          | 26ms              |

## 🚀 Key Features

### Advanced Model Features:

1. **ResNet-50 Backbone**: Robust feature extraction with deep residual connections
2. **CBAM Attention**: Channel and spatial attention mechanisms
3. **ASPP Module**: Multi-scale processing with dilated convolutions
4. **Dual-Branch Decoder**: CNN branch for local features + Transformer branch for global context
5. **Self-Attention**: Global context modeling in transformer branch
6. **Advanced Loss Function**: 5-component loss (Dice + Focal + Boundary + Consistency + Threshold-optimized)
7. **Differential Learning Rates**: Lower LR for pretrained backbone
8. **Enhanced Metrics**: Includes MCC, Balanced Accuracy, Specificity

### Lightweight Model Features:

- **Faster Inference**: 26ms vs 100ms for advanced model
- **Lower Memory**: Smaller ResNet-34 backbone
- **CBAM + ASPP**: Still includes attention and multi-scale processing
- **Good Performance**: Balanced speed vs accuracy

## 📁 Files Description

- **`advanced_model.py`**: Complete implementation of all advanced architectures
- **`train_advanced.py`**: Training script for advanced models with enhanced features
- **`model_demo.py`**: Demonstration script showing all features and comparisons
- **`train_optimized.py`**: Original training script (uses simple DeepLabV3+)
- **`model.py`**: Original model definitions (components used in advanced_model.py)

## 🔧 Usage

### 1. Train Advanced Model (All Features)

```bash
python train_advanced.py --model_type advanced --epochs 50 --batch_size 4 --use_aspp --use_dual_decoder
```

### 2. Train Lightweight Model (Faster)

```bash
python train_advanced.py --model_type lightweight --epochs 40 --batch_size 8
```

### 3. Compare Models

```bash
python model_demo.py --compare
```

### 4. See All Features Demo

```bash
python model_demo.py --all
```

## 🎯 Training Configurations

### Advanced Model Training:

- **Model**: AdvancedFloodSegmentationModel with all features
- **Loss**: 5-component enhanced loss function
- **Optimizer**: AdamW with differential learning rates
- **Scheduler**: CosineAnnealingWarmRestarts
- **Batch Size**: 4 (due to larger model)
- **Learning Rate**: 1e-5 (backbone: 1e-6)

### Lightweight Model Training:

- **Model**: LightweightAdvancedFloodModel
- **Loss**: Same enhanced loss function
- **Optimizer**: AdamW
- **Scheduler**: CosineAnnealingWarmRestarts
- **Batch Size**: 8 (smaller model allows larger batches)
- **Learning Rate**: 1e-5

## 📊 Expected Performance

Based on the architecture improvements, the advanced model should provide:

1. **Better Feature Extraction**: ResNet-50 with deep residual connections vs ResNet-34
2. **Improved Attention**: CBAM focuses on important regions
3. **Multi-scale Understanding**: ASPP captures different flood scales
4. **Enhanced Context**: Dual-branch decoder combines local + global features
5. **Better Boundaries**: Advanced loss emphasizes flood boundaries
6. **Threshold Optimization**: Loss optimized for 0.4 threshold

## 🔄 Model Selection Guide

### Use Advanced Model When:

- Maximum accuracy is needed
- GPU memory is sufficient (>8GB)
- Training time is not critical
- Complex flood patterns expected

### Use Lightweight Model When:

- Faster inference needed
- Limited GPU memory (<6GB)
- Good balance of speed vs accuracy required
- Deployment on edge devices

### Use Original Model When:

- Baseline comparison needed
- Quick prototyping
- Minimal computational resources

## 📈 Training Tips

1. **Start with Lightweight**: Test your setup first
2. **Use Mixed Precision**: Enabled by default for memory efficiency
3. **Monitor GPU Memory**: Reduce batch size if out of memory
4. **Resume Training**: Use `--resume` flag to continue interrupted training
5. **Experiment with Components**: Disable ASPP or dual decoder to find best config

## 🔍 Feature Analysis

The advanced model provides feature map extraction:

```python
model = AdvancedFloodSegmentationModel()
feature_maps = model.get_feature_maps(sar_input, optical_input)
# Returns: {'raw_features': tensor, 'processed_features': tensor}
```

## 📝 Output Files

Training produces:

- **`best_model_advanced_advanced.pth`**: Best advanced model weights
- **`best_model_lightweight_advanced.pth`**: Best lightweight model weights
- **`checkpoint_*_advanced.pth`**: Latest checkpoint with full state
- **`training_history_*_advanced.json`**: Complete training metrics history

## 🎊 Summary

The advanced implementation finally utilizes ALL the sophisticated components from the original `model.py`:

- ✅ CBAM attention mechanism
- ✅ ASPP multi-scale processing
- ✅ Dual-branch decoder architecture
- ✅ Self-attention transformers
- ✅ Advanced ResNet feature extraction
- ✅ Enhanced loss functions
- ✅ Proper weight initialization

Use `train_advanced.py` instead of `train_optimized.py` to leverage these advanced features!
