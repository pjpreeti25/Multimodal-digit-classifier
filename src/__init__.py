"""
Multimodal Digit Classification Package

This package provides modules for multimodal digit classification using
image and audio data fusion (matches original notebook approach).
"""

__version__ = "1.0.0"
__author__ = "Jnana Preeti Parlapalli"

from .preprocess import preprocess_data, load_data, reshape_data, split_data
from .model import create_models, create_image_encoder, create_audio_encoder, create_combined_model, compile_model
from .train import train_multimodal_model, hyperparameter_search, generate_embeddings
from .evaluate import evaluate_embeddings, generate_predictions, evaluate_model_performance

__all__ = [
    'preprocess_data',
    'load_data', 
    'reshape_data',
    'split_data',
    'create_models',
    'create_image_encoder',
    'create_audio_encoder',
    'create_combined_model',
    'compile_model',
    'train_multimodal_model',
    'hyperparameter_search',
    'generate_embeddings',
    'evaluate_embeddings',
    'generate_predictions',
    'evaluate_model_performance'
]
