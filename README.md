# 🌊 Underwater Image Enhancement Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Advanced deep learning model for underwater image enhancement combining Inception modules, Squeeze-and-Excitation layers, and depthwise separable convolutions.**

---
### Detection Results

<table>
  <tr>
    <td><img src="Enhanced Images/4.png.jpg" width="250"/></td>
  </tr>
  <tr>
    <td align="center">Tobacco Plants</td>
  </tr>
</table>

## 📋 Overview

This project implements a novel underwater image enhancement model that improves image quality by addressing color distortion, low contrast, and poor visibility issues common in underwater imaging. The architecture combines:

- **Inception Modules** for multi-scale feature extraction
- **Squeeze-and-Excitation (SE) Layers** for channel attention
- **Depthwise Separable Convolutions** for efficient processing
- **Combined Loss Function** (MSE + SSIM + Perceptual VGG Loss)

### Key Features

✅ **High Performance**: 24.95 dB PSNR on UIEB, 29.60 dB on EUVP  
✅ **Multi-GPU Training**: Supports distributed training across multiple GPUs  
✅ **Comprehensive Metrics**: MSE, PSNR, SSIM, UIQM, UCIQE evaluation  
✅ **Dual Dataset Support**: Trained and tested on UIEB and EUVP datasets  
✅ **Efficient Architecture**: Optimized for both quality and speed

---

## 🏗️ Model Architecture

### High-Level Overview

```
Input (RGB Image)
    ↓
Split into R, G, B channels
    ↓
[Inception + SE] for each channel → Concatenate (768 channels)
    ↓
DSConv Blocks (768→256→128→16)
    ↓
Additional DSConv with Skip Connections (16→32→64→32)
    ↓
Concatenate [16+32+64+32] → Select first 80 channels
    ↓
ConvBlock4 with SE Layer (80→32)
    ↓
1x1 Conv + Sigmoid (32→3)
    ↓
Enhanced Output (RGB Image)
```

### Components

1. **Inception Module (Inc)**: Multi-scale feature extraction with parallel convolutions (1x1, 3x3, 5x5, pooling)
2. **SE Layer**: Channel-wise attention mechanism for adaptive feature recalibration
3. **DSConv Block**: Depthwise separable convolutions for efficient processing
4. **ConvBlock4**: Enhanced convolution block with SE integration

---

## 📊 Performance Metrics

### UIEB Dataset Results

| Metric | Value |
|--------|-------|
| **MSE** | 0.0059 |
| **PSNR** | 24.9510 dB |
| **SSIM** | 0.8924 |
| **UIQM** | 3.1885 |
| **UCIQE** | 0.5891 |

### EUVP Dataset Results

| Metric | Value |
|--------|-------|
| **MSE** | 0.0066 |
| **PSNR** | 29.5985 dB |
| **SSIM** | 0.8603 |
| **UIQM** | 3.0313 |
| **UCIQE** | 0.5134 |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM recommended
- Multiple GPUs (optional, for distributed training)

### Setup

```bash
# Clone the repository
git clone https://github.com/Qaiser007khan/Underwater-Image-Enhancement.git
cd Underwater-Image-Enhancement

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Training the Model

#### Single GPU Training

```bash
python training_on_multi_GPUs.py
```

#### Multi-GPU Training

Modify the `devices` list in the script:

```python
devices = [0, 1]  # Use GPU 0 and GPU 1
```

Then run:

```bash
python training_on_multi_GPUs.py
```

### Configuration

Edit hyperparameters in `training_on_multi_GPUs.py`:

```python
learning_rate = 0.0002  # ADAM optimizer learning rate
batch_size = 8          # Batch size per GPU
num_epochs = 200        # Total training epochs
```

### Inference

```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = CustomModel()
model.load_state_dict(torch.load('UIEModel_final.pth'))
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load and enhance image
image = Image.open('underwater_image.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    enhanced = model(input_tensor)

# Save result
save_image(enhanced, 'enhanced_output.jpg')
```

---

## 📁 Project Structure

```
Underwater-Image-Enhancement/
├── codes/
│   ├── training_on_multi_GPUs.py    # Main training script
│   ├── evaluation.py                 # Evaluation metrics
│   └── uiqm.py                       # UIQM calculation
│
├── models/
│   ├── UIEModel_UIEB.pth            # Trained on UIEB dataset
│   └── UIEModel_EUVP.pth            # Trained on EUVP dataset
│
├── enhanced_images/
│   ├── UIEB_outputs/                # Enhanced images (UIEB)
│   └── EUVP_outputs/                # Enhanced images (EUVP)
│
├── results/
│   ├── training_metrics.png         # Loss, PSNR, SSIM, MSE curves
│   ├── sample_results/              # Visual comparisons
│   └── metrics_table.csv            # Detailed metrics
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📊 Datasets

### UIEB (Underwater Image Enhancement Benchmark)

- **Training**: 800 image pairs
- **Testing**: 90 image pairs
- **Resolution**: 256×256 (scaled)
- **Source**: [UIEB Dataset](https://li-chongyi.github.io/proj_benchmark.html)

### EUVP (Enhancing Underwater Visual Perception)

- **Training**: 9,108 image pairs (underwater_dark, underwater_imagenet, underwater_scenes)
- **Testing**: 2,277 image pairs
- **Resolution**: 256×256 (scaled)
- **Source**: [EUVP Dataset](https://irvlab.cs.umn.edu/resources/euvp-dataset)

**📥 Dataset Access**: Datasets are not included in this repository due to size constraints. Please contact the author to request access or download from the original sources.

---

## 🔬 Loss Function

The model uses a combined loss function:

**L_TOTAL = L_MSE + L_SSIM + L_VGG**

Where:
- **L_MSE**: Mean Squared Error for pixel-wise accuracy
- **L_SSIM**: Structural Similarity Index for perceptual quality
- **L_VGG**: Perceptual loss using pre-trained VGG19 features

---

## 🎓 Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | ADAM |
| **Learning Rate** | 0.0002 |
| **Batch Size** | 8 |
| **Epochs** | 200 |
| **Image Size** | 256×256 |
| **GPU** | NVIDIA Tesla V100 (or equivalent) |

---

## 📈 Results Visualization

### Training Metrics

The model automatically generates training curves showing:
- Loss vs. Epoch
- PSNR vs. Epoch
- SSIM vs. Epoch
- MSE vs. Epoch

![Training Metrics](results/training_metrics.png)

### Sample Enhancements

| Raw Image | Enhanced Image |
|-----------|----------------|
| ![Raw](results/sample_results/raw_1.jpg) | ![Enhanced](results/sample_results/enhanced_1.jpg) |
| ![Raw](results/sample_results/raw_2.jpg) | ![Enhanced](results/sample_results/enhanced_2.jpg) |

---

## 🔮 Future Work

- [ ] Real-time video enhancement
- [ ] Mobile deployment optimization
- [ ] Attention mechanism improvements
- [ ] Integration with underwater robotics systems
- [ ] Few-shot learning for limited data scenarios
- [ ] 3D underwater scene reconstruction

---

## 👨‍💻 Author

**Qaiser Khan**
- AI Developer, CENTAIC-NASTP
- MS Mechatronics (AI & Robotics), NUST
- 📧 qkhan.mts21ceme@student.nust.edu.pk
- 🔗 [LinkedIn](https://www.linkedin.com/in/engr-qaiser-khan-520252112) | [GitHub](https://github.com/Qaiser007khan)

---

## 🙏 Acknowledgments

This project builds upon research from:
- **DICAM**: [Inception module reference](https://github.com/hfarhaditolie/DICAM)
- **LiteEnhanceNet**: [SE Layer and lightweight architecture](https://github.com/zhangsong1213/LiteEnhanceNet)
- **UIEB**: Underwater Image Enhancement Benchmark dataset
- **EUVP**: Enhancing Underwater Visual Perception dataset

Special thanks to:
- CENTAIC-NASTP for computational resources
- NUST for research support
- Open-source deep learning community

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

### For Technical Questions:
- 📧 Email: qkhan.mts21ceme@student.nust.edu.pk
- 💬 [Create an Issue](https://github.com/Qaiser007khan/Underwater-Image-Enhancement/issues)

### For Dataset Access:
- 📧 Email: qaiserkhan.centaic@gmail.com
- 📝 Please specify which dataset you need (UIEB/EUVP)

### For Collaboration:
- 💼 LinkedIn: [Qaiser Khan](https://www.linkedin.com/in/engr-qaiser-khan-520252112)
- 📱 WhatsApp: +92-318-9000211

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{khan2024underwater,
  author = {Khan, Qaiser},
  title = {Underwater Image Enhancement Using Deep Learning},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/Qaiser007khan/Underwater-Image-Enhancement}
}
```

---

## 📊 References

1. Li, C., et al. "An Underwater Image Enhancement Benchmark Dataset and Beyond." IEEE TIP, 2019.
2. Islam, M. J., et al. "Fast Underwater Image Enhancement for Improved Visual Perception." IEEE RAL, 2020.
3. Hu, J., et al. "Squeeze-and-Excitation Networks." CVPR, 2018.
4. Szegedy, C., et al. "Going Deeper with Convolutions." CVPR, 2015.

---

<div align="center">

### 🌟 Star this repository if you find it useful!

### 🤝 Contributions and feedback are welcome!

![GitHub stars](https://img.shields.io/github/stars/Qaiser007khan/Underwater-Image-Enhancement?style=social)
![GitHub forks](https://img.shields.io/github/forks/Qaiser007khan/Underwater-Image-Enhancement?style=social)

**Made with ❤️ for underwater imaging research**

</div>
