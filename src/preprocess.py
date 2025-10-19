"""
Data preprocessing module for multimodal digit classification.

This module handles loading and preprocessing of image and audio data
for the multimodal digit classifier.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    """
    Load training and test data for multimodal digit classification.
    This matches the original notebook's data loading approach.
    
    Returns:
        tuple: (x_train_wr, x_train_sp, x_test_wr, x_test_sp, y_train)
            - x_train_wr: Training image data
            - x_train_sp: Training audio data  
            - x_test_wr: Test image data
            - x_test_sp: Test audio data
            - y_train: Training labels
    """
    # Load numpy arrays for image and audio data (original approach)
    x_train_wr = np.load('x_train_wr.npy')
    x_train_sp = np.load('x_train_sp.npy')
    x_test_wr = np.load('x_test_wr.npy')
    x_test_sp = np.load('x_test_sp.npy')
    
    # Load training labels
    y_train = pd.read_csv('y_train.csv')
    
    return x_train_wr, x_train_sp, x_test_wr, x_test_sp, y_train


def reshape_data(x_train_wr, x_train_sp, x_test_wr, x_test_sp):
    """
    Reshape data for model input (matches original notebook).
    
    Args:
        x_train_wr: Training image data
        x_train_sp: Training audio data
        x_test_wr: Test image data  
        x_test_sp: Test audio data
        
    Returns:
        tuple: Reshaped data arrays
    """
    # Reshape image data to (samples, 28, 28) - original approach
    image_test = x_test_wr.reshape(-1, 28, 28)
    x_train_wr_reshaped = x_train_wr.reshape(-1, 28, 28)
    
    # Reshape audio data to (samples, 507) - original approach
    audio_test = x_test_sp.reshape(-1, 507)
    x_train_sp_reshaped = x_train_sp.reshape(-1, 507)
    
    return x_train_wr_reshaped, x_train_sp_reshaped, image_test, audio_test


def split_data(image_train, audio_train, y_train, test_size=0.2, random_state=42):
    """
    Split training data into training and validation sets (original approach).
    
    Args:
        image_train: Training image data
        audio_train: Training audio data
        y_train: Training labels
        test_size (float): Proportion of data for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: Split data arrays
    """
    image_train_split, image_val, audio_train_split, audio_val, labels_train, labels_val = train_test_split(
        image_train, audio_train, y_train, 
        test_size=test_size, 
        random_state=random_state
    )
    
    return image_train_split, image_val, audio_train_split, audio_val, labels_train, labels_val


def preprocess_data():
    """
    Complete data preprocessing pipeline (matches original notebook).
    
    Returns:
        tuple: Preprocessed data ready for training
    """
    # Load raw data
    x_train_wr, x_train_sp, x_test_wr, x_test_sp, y_train = load_data()
    
    # Reshape data
    image_train, audio_train, image_test, audio_test = reshape_data(
        x_train_wr, x_train_sp, x_test_wr, x_test_sp
    )
    
    # Split training data
    image_train_split, image_val, audio_train_split, audio_val, labels_train, labels_val = split_data(
        image_train, audio_train, y_train
    )
    
    return {
        'image_train': image_train_split,
        'image_val': image_val,
        'image_test': image_test,
        'audio_train': audio_train_split,
        'audio_val': audio_val,
        'audio_test': audio_test,
        'labels_train': labels_train,
        'labels_val': labels_val
    }


if __name__ == "__main__":
    # Example usage
    data = preprocess_data()
    print("Data preprocessing completed successfully!")
    print(f"Training images shape: {data['image_train'].shape}")
    print(f"Training audio shape: {data['audio_train'].shape}")
    print(f"Test images shape: {data['image_test'].shape}")
    print(f"Test audio shape: {data['audio_test'].shape}")
