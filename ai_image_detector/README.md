# AI Image Detector 🔍

A comprehensive multi-layer AI image detection system that analyzes images to determine if they are real or AI-generated. Uses a combination of deep learning, frequency analysis, noise pattern analysis, metadata inspection, and pixel statistics.

## 🌟 Features

### Multi-Layer Analysis System

**Layer 1 - Low Level (Signal)**
- 📊 **Fourier Spectrum Analysis**: Detects frequency domain artifacts
- 🔊 **Noise Pattern Analysis**: Identifies synthetic noise vs real camera noise
- 📈 **Pixel Statistics**: Analyzes color distribution and intensity patterns

**Layer 2 - Mid Level (Structural)**
- 🖼️ **Edge Coherence**: Examines edge consistency
- 🎨 **Texture Continuity**: Detects texture artifacts
- ⚖️ **Symmetry Analysis**: Identifies unnatural symmetry

**Layer 3 - High Level (Semantic)**
- 🤖 **Deep Learning Model**: EfficientNet-based classifier
- 🏷️ **Model Attribution**: Identifies specific AI generators (Stable Diffusion, Midjourney, DALL-E)
- 📝 **Metadata Analysis**: Examines EXIF data for authenticity indicators

### Supported AI Generators
- ✅ Stable Diffusion
- ✅ Midjourney
- ✅ DALL-E
- ✅ Unknown/Generic AI

## 📋 Requirements

- Python 3.8+
- PyTorch
- Flask
- OpenCV
- NumPy, SciPy
- Modern web browser

## 🚀 Installation

### 1. Clone/Download the project

```bash
cd ai_image_detector
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Installation may take 5-10 minutes depending on your internet connection.

### 3. Download Training Dataset

The system requires a dataset for training. We recommend the CIFAKE dataset:

#### Option A: Automatic Download (Recommended)

```bash
cd datasets
python download_datasets.py
```

Follow the prompts to:
1. Setup Kaggle API credentials
2. Download the dataset automatically
3. Organize images into the correct folders

#### Option B: Manual Download

1. Visit [CIFAKE Dataset on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
2. Download and extract the dataset
3. Organize images into this structure:

```
datasets/
├── train/
│   ├── real/              (place real images here)
│   ├── stable_diffusion/  (place AI-generated images here)
│   ├── midjourney/        (optional)
│   ├── dalle/             (optional)
│   └── unknown/           (optional)
└── test/
    └── (same structure)
```

**Minimum Requirements**:
- At least 100-200 images per class for basic training
- Recommended: 1000+ images per class for better accuracy

## 🎯 Training the Model

Once you have organized your dataset:

```bash
cd backend
python train_model.py
```

**Training Process**:
- Uses EfficientNet-B0 (lightweight, CPU-friendly)
- Trains for 15 epochs by default
- Creates checkpoints: `models/best_model.pth` and `models/final_model.pth`
- Training time: 10-30 minutes on CPU (depending on dataset size)

**Training Configuration** (can be modified in `train_model.py`):
```python
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.001
```

## 🖥️ Running the Application

### Start Backend Server

```bash
cd backend
python app.py
```

The server will start at `http://localhost:5000`

### Open Frontend

1. Open `frontend/index.html` in your web browser
2. Or use a simple HTTP server:

```bash
cd frontend
python -m http.server 8000
```

Then visit `http://localhost:8000`

## 📖 Usage

1. **Upload Image**: Drag & drop or click to select an image (JPG, PNG, JPEG, SVG)
2. **Analyze**: Click "Analyze Image" button
3. **View Results**: Get comprehensive analysis including:
   - Final classification (Real vs AI-Generated)
   - Confidence score (0-100%)
   - Model prediction with probabilities
   - Individual component scores
   - Detailed technical metrics
   - Image metadata

## 🔬 How It Works

### Analysis Pipeline

```
Image Upload
    ↓
Frequency Analysis (FFT, spectrum analysis)
    ↓
Noise Pattern Analysis (residual extraction)
    ↓
Metadata Inspection (EXIF data)
    ↓
Pixel Statistics (color distribution, contrast)
    ↓
Deep Learning Model (EfficientNet classifier)
    ↓
Score Combination (weighted average)
    ↓
Final Result (Real/AI with confidence)
```

### Scoring System

The final score is a weighted combination:
- **Deep Learning Model**: 40%
- **Frequency Analysis**: 15%
- **Noise Analysis**: 15%
- **Pixel Statistics**: 15%
- **Metadata Analysis**: 15%

**Classification Thresholds**:
- Score ≥ 70: Real Image (High Confidence)
- Score 50-69: Uncertain (Medium Confidence)
- Score < 50: AI-Generated (High Confidence)

## 📁 Project Structure

```
ai_image_detector/
├── backend/
│   ├── app.py                    # Flask API server
│   ├── train_model.py            # Model training script
│   ├── frequency_analyzer.py     # Frequency analysis module
│   ├── noise_analyzer.py         # Noise pattern analysis
│   ├── metadata_analyzer.py      # EXIF metadata analysis
│   └── pixel_analyzer.py         # Pixel statistics analysis
├── frontend/
│   └── index.html                # Web interface
├── models/
│   ├── best_model.pth           # Best model checkpoint
│   └── final_model.pth          # Final trained model
├── datasets/
│   ├── train/                   # Training data
│   ├── test/                    # Testing data
│   └── download_datasets.py     # Dataset downloader
├── static/
│   └── uploads/                 # Temporary upload folder
└── requirements.txt             # Python dependencies
```

## 🎨 API Endpoints

### `POST /api/analyze`
Analyze an uploaded image

**Request**: `multipart/form-data` with `image` field

**Response**:
```json
{
  "success": true,
  "image_info": {
    "filename": "image.jpg",
    "width": 1920,
    "height": 1080,
    ...
  },
  "analysis": {
    "final_result": {
      "final_score": 85.5,
      "classification": "Real Image",
      "confidence_level": "High"
    },
    "model_prediction": {
      "class": "Real",
      "confidence": 92.3,
      "probabilities": {...}
    },
    ...
  }
}
```

### `GET /api/health`
Check server health and model status

### `GET /api/model-info`
Get information about the loaded model

## 🔧 Configuration

### Adjusting Analysis Weights

Edit `backend/app.py`, function `combine_confidence_scores()`:

```python
final_score = (
    0.40 * model_score +      # Deep learning
    0.15 * freq_score +       # Frequency
    0.15 * noise_score +      # Noise
    0.15 * pixel_score +      # Pixels
    0.15 * metadata_score     # Metadata
)
```

### Model Architecture

To use a different EfficientNet variant, edit `train_model.py`:

```python
# Options: efficientnet-b0 to efficientnet-b7
self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
```

**Note**: Higher variants (b1-b7) are more accurate but slower.

## ⚠️ Limitations

1. **Training Data Required**: Model needs to be trained on labeled data
2. **CPU Performance**: Training and inference are slower on CPU vs GPU
3. **Dataset Diversity**: Accuracy depends on diversity of training data
4. **Metadata Stripping**: Many platforms remove EXIF data, limiting metadata analysis
5. **Evolving AI**: New AI generators may not be detected accurately

## 🐛 Troubleshooting

### "Model not found" error
- Run `python train_model.py` to train the model first
- Ensure `models/best_model.pth` exists

### "No images found" during training
- Verify dataset structure matches the format above
- Check file permissions
- Ensure image files are valid

### CORS errors in browser
- Ensure backend is running on port 5000
- Check that frontend is accessing `http://localhost:5000`

### Low accuracy results
- Train with more diverse data
- Increase epochs in `train_model.py`
- Use more training images (1000+ per class)

## 📊 Performance Tips

1. **For CPU Training**:
   - Use smaller batch size (8-16)
   - Consider EfficientNet-B0 (smallest variant)
   - Be patient - training takes time

2. **For Better Accuracy**:
   - Collect more diverse training data
   - Balance classes (equal images per class)
   - Use data augmentation (already included)
   - Train for more epochs

3. **For Faster Inference**:
   - Use GPU if available
   - Reduce image resolution
   - Consider model quantization

## 🤝 Contributing

This is an educational project. Feel free to:
- Improve analysis algorithms
- Add new AI generator detectors
- Enhance the UI
- Optimize performance

## 📚 References

- **EfficientNet**: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- **CIFAKE Dataset**: [Bird & Lotfi, 2023](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- **Frequency Analysis**: Traditional signal processing techniques
- **Noise Analysis**: Camera forensics methodologies

## 📝 License

This project is for educational purposes. Datasets may have their own licenses.

## 🆘 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the code comments
3. Verify all dependencies are installed
4. Ensure Python 3.8+ is being used

## ✨ Future Enhancements

- [ ] Support for video analysis
- [ ] Batch processing
- [ ] API key authentication
- [ ] More AI generator types
- [ ] Real-time webcam analysis
- [ ] Export analysis reports
- [ ] GPU acceleration optimization
- [ ] Docker containerization

---

**Made with 🔬 for AI research and education**
