# Multimodal Digit Classification

A comprehensive machine learning project developed for a Kaggle competition demonstrating advanced multimodal fusion techniques for digit classification using both image and audio data. This project showcases end-to-end data science capabilities including data preprocessing, model architecture design, hyperparameter optimization, and comprehensive evaluation in a competitive environment.

**Perfect for demonstrating:** Multimodal ML, Neural Networks, Data Visualization, Model Evaluation, Software Engineering

## Project Highlights

- **Kaggle Competition**: Real competitive environment with performance metrics
- **Advanced ML Technique**: Multimodal fusion combining image and audio data
- **Performance Focus**: Designed to beat baseline F1 scores
- **Professional Architecture**: Separate encoders with concatenation-based fusion
- **Systematic Optimization**: Grid search hyperparameter tuning
- **Comprehensive Evaluation**: t-SNE visualization + K-means clustering analysis
- **Production-Ready Code**: Modular design with proper documentation
- **Scientific Rigor**: Homogeneity scores, confusion matrices, training curves

## Project Overview

This project addresses the challenge of multimodal digit classification by:
- Creating separate encoder models for image and audio data
- Implementing fusion methods to combine modalities
- Performing comprehensive evaluation with t-SNE visualization and K-means clustering
- Generating predictions on test data

## Project Structure

```
multimodal-digit-classifier/
│
├── data/                    # Data directory (place your .npy and .csv files here)
│   ├── x_train_wr.npy      # Training image data
│   ├── x_train_sp.npy      # Training audio data
│   ├── x_test_wr.npy       # Test image data
│   ├── x_test_sp.npy       # Test audio data
│   └── y_train.csv         # Training labels
│
├── notebooks/
│   └── Jnana_Preeti_Parlapalli_hw5.ipynb  # Main analysis notebook
│
├── src/
│   ├── preprocess.py       # Data loading and preprocessing
│   ├── model.py           # Model architecture definitions
│   ├── train.py           # Training and hyperparameter tuning
│   └── evaluate.py        # Evaluation and visualization
│
├── results/
│   ├── image_tsne_visualization.png    # t-SNE plot of image embeddings
│   ├── audio_tsne_visualization.png    # t-SNE plot of audio embeddings
│   ├── image_confusion_matrix.png      # Confusion matrix for image clustering
│   ├── audio_confusion_matrix.png      # Confusion matrix for audio clustering
│   ├── main.tex                        # LaTeX source for project report
│   └── neurips_2022.sty               # LaTeX style file
│
├── preds/
│   └── final_preds.csv     # Final predictions for submission
│
└── README.md              # This file
```

## Architecture Overview

### Model Architecture
The project implements a **dual-encoder architecture** with concatenation-based fusion for multimodal digit classification:

```
Image Input (28×28) → Image Encoder → Image Embedding (64-dim)
                                    ↓
Audio Input (507-dim) → Audio Encoder → Audio Embedding (64-dim)
                                    ↓
                            Concatenate → Combined Features (128-dim)
                                    ↓
                            Dense Layers → Classification (10 classes)
```


## Requirements

### Data Files
**Note: Due to data privacy and confidentiality requirements, the actual dataset files are not included in this repository.**

To run this project, you'll need to obtain the following data files and place them in the `data/` directory:
- `x_train_wr.npy`: Training image data (shape: [N, 28*28])
- `x_train_sp.npy`: Training audio data (shape: [N, 507])
- `x_test_wr.npy`: Test image data (shape: [M, 28*28])
- `x_test_sp.npy`: Test audio data (shape: [M, 507])
- `y_train.csv`: Training labels with 'label' column

### Data Description
- **Images**: 28×28 grayscale handwritten digits
- **Audio**: 507-dimensional audio features corresponding to each digit
- **Labels**: Digit classes 0-9

### Usage
Once you have the data files, place them in the `data/` directory and run:

```python
from src.preprocess import preprocess_data
data = preprocess_data(data_path='data/')
```

### Python Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```

## Usage

### Quick Start
1. Place your data files in the `data/` directory
2. Run the main notebook:
   ```bash
   jupyter notebook notebooks/multimodal_fusion.ipynb
   ```

### Using Individual Modules

#### Data Preprocessing
```python
from src.preprocess import preprocess_data

# Load and preprocess data
data = preprocess_data(data_path='data/')
```

#### Model Creation
```python
from src.model import create_models

# Create all models
image_encoder, audio_encoder, combined_encoder, multimodal_model = create_models()
```

#### Training
```python
from src.train import train_multimodal_model

# Train with hyperparameter tuning
best_model, best_params, history = train_multimodal_model(data)
```

#### Evaluation
```python
from src.evaluate import evaluate_embeddings, generate_predictions

# Evaluate embeddings
results = evaluate_embeddings(image_embeddings, audio_embeddings, labels)

# Generate predictions
predictions = generate_predictions(model, image_test, audio_test)
```

## Model Architecture

### Image Encoder
- Input: 28×28 grayscale images
- Architecture: Flatten → Dense(64, ReLU)
- Output: 64-dimensional embedding

### Audio Encoder
- Input: 507-dimensional audio features
- Architecture: Dense(64, ReLU)
- Output: 64-dimensional embedding

### Multimodal Fusion
- Method: Concatenation of image and audio embeddings
- Classification: Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)

## Key Features

### Hyperparameter Optimization
- Grid search over learning rates, batch sizes, and optimizers
- Automatic selection of best performing configuration

### Comprehensive Evaluation
- t-SNE visualization of embeddings
- K-means clustering analysis
- Homogeneity score calculation
- Confusion matrix visualization

### Modular Design
- Clean separation of concerns
- Reusable components
- Easy to extend and modify

## Results

The model achieves good performance by combining visual and auditory information:
- Image embeddings show better clustering performance
- Audio features provide complementary information
- Multimodal fusion improves overall classification accuracy

## Analysis Insights

### Clustering Results
- **Image embeddings**: Higher homogeneity score indicates better discriminative features
- **Audio embeddings**: Lower homogeneity but still provides useful information
- **Combined approach**: Leverages strengths of both modalities

### Visualization
- t-SNE plots show clear separation of digit classes
- Confusion matrices reveal model strengths and weaknesses
- Training curves demonstrate stable convergence

## File Descriptions

### Source Code
- `preprocess.py`: Handles data loading, reshaping, and train/validation split
- `model.py`: Defines encoder architectures and multimodal fusion
- `train.py`: Implements training pipeline with hyperparameter search
- `evaluate.py`: Provides evaluation metrics and visualization functions

### Output Files
- `*_visualization.png`: t-SNE plots and confusion matrices
- `final_preds.csv`: Submission-ready predictions
- `main.tex`: Comprehensive technical report

**Note**: Trained model files (`.h5`) and embedding files (`.npy`) are not included due to file size and privacy considerations, but the complete code to generate them is provided.

## Future Improvements

1. **Advanced Fusion Methods**:
   - Attention mechanisms
   - Cross-modal attention
   - Transformer-based fusion

2. **Enhanced Encoders**:
   - CNN for image processing
   - RNN/LSTM for audio sequences
   - Pre-trained feature extractors

3. **Ensemble Methods**:
   - Multiple model combinations
   - Different fusion strategies
   - Voting mechanisms


## License

This project is part of academic coursework for SP24 TAMU-CSCE-633-600.
