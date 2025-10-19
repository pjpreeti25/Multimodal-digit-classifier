# Source Code Directory

This directory contains the modular Python implementation of the multimodal digit classification system. The code is organized into logical modules for better maintainability and reusability.

## Module Overview

### `__init__.py`
Package initialization file that makes `src` a Python package and exports all public functions.

### `preprocess.py`
Data loading and preprocessing module.

**Key Functions:**
- `load_data()`: Load raw data from files
- `reshape_data()`: Reshape data for model input
- `split_data()`: Split data into train/validation sets
- `preprocess_data()`: Complete preprocessing pipeline

**Usage:**
```python
from src.preprocess import preprocess_data
data = preprocess_data()
```

### `model.py`
Neural network architecture definitions.

**Key Functions:**
- `create_image_encoder()`: Create image encoder model
- `create_audio_encoder()`: Create audio encoder model
- `create_combined_model()`: Create multimodal fusion model
- `create_models()`: Create all models at once
- `compile_model()`: Compile model with optimizer

**Usage:**
```python
from src.model import create_models, compile_model
image_encoder, audio_encoder, multimodal_model = create_models()
compiled_model = compile_model(multimodal_model, optimizer='sgd', learning_rate=0.001)
```

### `train.py`
Training and hyperparameter optimization module.

**Key Functions:**
- `hyperparameter_search()`: Grid search for optimal hyperparameters
- `train_best_model()`: Train model with best hyperparameters
- `train_multimodal_model()`: Complete training pipeline
- `generate_embeddings()`: Generate embeddings from trained encoders

**Usage:**
```python
from src.train import train_multimodal_model, generate_embeddings
best_model, best_params, history = train_multimodal_model(data)
image_embeddings, audio_embeddings = generate_embeddings(image_encoder, audio_encoder, image_data, audio_data)
```

### `evaluate.py`
Evaluation and visualization module.

**Key Functions:**
- `visualize_embeddings_tsne()`: Create t-SNE visualizations
- `perform_clustering_analysis()`: K-means clustering analysis
- `plot_confusion_matrix()`: Plot confusion matrices
- `evaluate_embeddings()`: Complete embedding evaluation
- `generate_predictions()`: Generate test predictions
- `evaluate_model_performance()`: Evaluate model on validation data

**Usage:**
```python
from src.evaluate import evaluate_embeddings, generate_predictions
results = evaluate_embeddings(image_embeddings, audio_embeddings, labels)
predictions = generate_predictions(model, image_test, audio_test)
```

## Architecture Overview

### Model Architecture
```
Input: [Image (28×28), Audio (507)]
    ↓
Image Encoder: Flatten → Dense(64, ReLU)
Audio Encoder: Dense(64, ReLU)
    ↓
Concatenation: [64 + 64 = 128]
    ↓
Classification: Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)
    ↓
Output: Digit probabilities (0-9)
```

### Data Flow
1. **Preprocessing**: Load → Reshape → Split
2. **Training**: Hyperparameter search → Best model training
3. **Evaluation**: Embedding analysis → Visualization
4. **Prediction**: Test data → Final predictions

## Key Features

### Modular Design
- **Separation of Concerns**: Each module has a specific responsibility
- **Reusable Components**: Functions can be imported and used independently
- **Easy Maintenance**: Changes to one module don't affect others

### Professional Standards
- **Comprehensive Documentation**: All functions have detailed docstrings
- **Type Hints**: Clear parameter and return type specifications
- **Error Handling**: Proper exception handling and validation
- **Consistent Naming**: Clear, descriptive function and variable names

### Advanced ML Techniques
- **Multimodal Fusion**: Combines image and audio data effectively
- **Hyperparameter Optimization**: Systematic grid search approach
- **Embedding Analysis**: t-SNE visualization and clustering
- **Performance Evaluation**: Comprehensive metrics and visualizations

## Dependencies

The modules require the following Python packages:
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `tensorflow/keras`: Deep learning framework
- `scikit-learn`: Machine learning utilities
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical visualization

## Usage Examples

### Complete Pipeline
```python
# Import modules
from src.preprocess import preprocess_data
from src.model import create_models
from src.train import train_multimodal_model
from src.evaluate import evaluate_embeddings, generate_predictions

# Load and preprocess data
data = preprocess_data()

# Create models
image_encoder, audio_encoder, multimodal_model = create_models()

# Train model
best_model, best_params, history = train_multimodal_model(data)

# Evaluate embeddings
results = evaluate_embeddings(image_embeddings, audio_embeddings, data['labels_train'])

# Generate predictions
predictions = generate_predictions(best_model, data['image_test'], data['audio_test'])
```

### Individual Module Usage
```python
# Use specific functions
from src.preprocess import load_data, reshape_data
from src.model import create_image_encoder
from src.evaluate import visualize_embeddings_tsne

# Load data
x_train_wr, x_train_sp, x_test_wr, x_test_sp, y_train = load_data()

# Create specific encoder
image_encoder = create_image_encoder(embedding_dim=128)

# Visualize embeddings
tsne_result = visualize_embeddings_tsne(embeddings, labels, "My Visualization")
```

## Performance Results

- **Kaggle Competition Score**: 0.969 (excellent performance!)
- **Best Hyperparameters**: Learning rate = 0.001, Batch size = 32, Optimizer = SGD
- **Image Embedding Homogeneity**: 0.296
- **Audio Embedding Homogeneity**: 0.146

## Extensibility

The modular design makes it easy to extend the system:

### Adding New Encoders
```python
def create_cnn_encoder(input_shape=(28, 28, 1)):
    # Add CNN layers
    pass
```

### Adding New Fusion Methods
```python
def attention_fusion(image_features, audio_features):
    # Implement attention mechanism
    pass
```

### Adding New Evaluation Metrics
```python
def calculate_f1_score(y_true, y_pred):
    # Add F1 score calculation
    pass
```

## Best Practices

1. **Import Specific Functions**: Use `from src.module import function` for better performance
2. **Handle Data Paths**: Ensure data files are in the correct directory
3. **Save Intermediate Results**: Use the save_path parameters to store outputs
4. **Monitor Training**: Check validation accuracy during training
5. **Visualize Results**: Always examine t-SNE plots and confusion matrices

## Troubleshooting

### Common Issues
- **Import Errors**: Ensure you're in the correct directory
- **Data Loading**: Check that data files exist and are accessible
- **Memory Issues**: Reduce batch size or use data generators
- **Convergence**: Adjust learning rate or add regularization

### Debugging Tips
- Use `print()` statements to check data shapes
- Visualize intermediate results
- Check model summaries for architecture verification
- Monitor training curves for overfitting
