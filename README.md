# ResNet-18 FCN for Image Segmentation

This repository implements Fully Convolutional Networks (FCN) for semantic segmentation using a pretrained ResNet-18 backbone. It includes implementations both with and without skip connections to evaluate their impact on segmentation performance.

## Architecture

### Base Architecture
- Backbone: ResNet-18 pretrained on ImageNet
- Converted to FCN by removing final fully connected layers
- Upsampling layers to restore original image resolution
- Output produces per-pixel segmentation maps

### Two Variants
1. **FCN without Skip Connections**
   - Direct upsampling from final convolutional features
   - Single stream architecture

2. **FCN with Skip Connections**
   - Combines features from multiple ResNet layers
   - Skip connections from earlier layers for fine-grained details
   - Element-wise addition of features before upsampling

## Requirements

```
torch
torchvision
numpy
PIL
matplotlib
sklearn.metrics
tqdm
```

## Dataset Preparation

Structure your data as follows:
```
data/
    train/
        images/
        masks/
    val/
        images/
        masks/
    test/
        images/
        masks/
```

## Usage

### Training

```python
# Train FCN without skip connections
python train.py --model fcn_basic --epochs 100 --batch_size 16

# Train FCN with skip connections
python train.py --model fcn_skip --epochs 100 --batch_size 16
```

### Evaluation

```python
python evaluate.py --model_path checkpoints/best_model.pth --test_dir data/test
```

## Model Performance

### Quantitative Results

| Model | Pixel Accuracy | Mean IOU |
|-------|---------------|----------|
| FCN without Skip | X% | Y% |
| FCN with Skip | X% | Y% |

### Key Findings
- Impact of skip connections on fine detail preservation
- Trade-offs between computational complexity and accuracy
- Analysis of feature map resolution effects

## Implementation Details

### Model Conversion
```python
class FCN(nn.Module):
    def __init__(self, n_classes, with_skip=False):
        super(FCN, self).__init__()
        # Load pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=True)
        # Remove final FC layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        # Add upsampling layers
        self.upsampling = self._make_upsampling_layers()
```

### Training Configuration
- Optimizer: Adam
- Learning Rate: 0.001 with decay
- Loss Function: Cross-Entropy
- Batch Size: 16
- Epochs: 100

## Project Structure

```
.
├── models/
│   ├── fcn_basic.py
│   └── fcn_skip.py
├── utils/
│   ├── dataset.py
│   ├── metrics.py
│   └── transforms.py
├── train.py
├── evaluate.py
└── config.py
```

## Visualization

The repository includes tools for visualizing:
- Segmentation predictions
- Feature maps at different stages
- Training progress metrics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

```
@article{long2015fully,
  title={Fully convolutional networks for semantic segmentation},
  author={Long, Jonathan and Shelhamer, Evan and Darrell, Trevor},
  journal={CVPR},
  year={2015}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ResNet architecture by He et al.
- FCN paper by Long et al.
- PyTorch team for the framework and pretrained models
