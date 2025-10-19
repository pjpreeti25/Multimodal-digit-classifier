"""
Training module for multimodal digit classification.

This module handles model training, hyperparameter tuning, and model saving.
"""

import keras
import numpy as np
from keras.callbacks import EarlyStopping
from .model import create_models, compile_model


def hyperparameter_search(model, image_train, audio_train, labels_train, 
                         image_val, audio_val, labels_val,
                         learning_rates=[0.01, 0.001], 
                         batch_sizes=[32, 64], 
                         optimizers=['Adam', 'SGD'],
                         epochs=10):
    """
    Perform grid search for optimal hyperparameters.
    
    Args:
        model: Model to train
        image_train: Training image data
        audio_train: Training audio data
        labels_train: Training labels
        image_val: Validation image data
        audio_val: Validation audio data
        labels_val: Validation labels
        learning_rates (list): List of learning rates to try
        batch_sizes (list): List of batch sizes to try
        optimizers (list): List of optimizers to try
        epochs (int): Number of epochs for each training run
        
    Returns:
        dict: Best hyperparameters found
    """
    best_val_accuracy = 0
    best_hyperparameters = {}
    
    for lr in learning_rates:
        for batch_size_value in batch_sizes:
            for optimizer_name in optimizers:
                print(f"Learning rate: {lr}, Batch size: {batch_size_value}, Optimizer: {optimizer_name}")
                
                # Compile model with current hyperparameters
                compiled_model = compile_model(model, optimizer=optimizer_name, learning_rate=lr)
                
                # Train model
                history = compiled_model.fit(
                    [image_train, audio_train],
                    labels_train['label'],
                    epochs=epochs,
                    batch_size=batch_size_value,
                    validation_data=([image_val, audio_val], labels_val['label']),
                    verbose=0
                )
                
                # Get validation accuracy
                val_accuracy = history.history['val_accuracy'][-1]
                
                # Update best hyperparameters if current is better
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_hyperparameters = {
                        'learning_rate': lr,
                        'batch_size': batch_size_value,
                        'optimizer': optimizer_name
                    }
    
    print("Best Hyperparameters:")
    print(best_hyperparameters)
    
    return best_hyperparameters


def train_best_model(model, image_train, audio_train, labels_train,
                    image_val, audio_val, labels_val,
                    best_hyperparameters, epochs=10):
    """
    Train the model with the best hyperparameters found.
    
    Args:
        model: Model to train
        image_train: Training image data
        audio_train: Training audio data
        labels_train: Training labels
        image_val: Validation image data
        audio_val: Validation audio data
        labels_val: Validation labels
        best_hyperparameters (dict): Best hyperparameters from grid search
        epochs (int): Number of epochs to train
        
    Returns:
        keras.Model: Trained model
        keras.callbacks.History: Training history
    """
    # Compile model with best hyperparameters
    best_model = compile_model(
        model, 
        optimizer=best_hyperparameters['optimizer'],
        learning_rate=best_hyperparameters['learning_rate']
    )
    
    # Train the model
    history = best_model.fit(
        [image_train, audio_train],
        labels_train['label'],
        epochs=epochs,
        batch_size=best_hyperparameters['batch_size'],
        validation_data=([image_val, audio_val], labels_val['label'])
    )
    
    return best_model, history


def train_multimodal_model(data, epochs=10, save_path='../results/'):
    """
    Complete training pipeline for multimodal model.
    
    Args:
        data (dict): Preprocessed data from preprocess module
        epochs (int): Number of epochs for final training
        save_path (str): Path to save trained models
        
    Returns:
        tuple: (best_model, best_hyperparameters, training_history)
    """
    # Create models
    image_encoder, audio_encoder, combined_encoder, multimodal_model = create_models()
    
    # Perform hyperparameter search
    print("Starting hyperparameter search...")
    best_hyperparameters = hyperparameter_search(
        multimodal_model,
        data['image_train'], data['audio_train'], data['labels_train'],
        data['image_val'], data['audio_val'], data['labels_val'],
        epochs=epochs
    )
    
    # Train final model with best hyperparameters
    print("Training final model with best hyperparameters...")
    best_model, history = train_best_model(
        multimodal_model,
        data['image_train'], data['audio_train'], data['labels_train'],
        data['image_val'], data['audio_val'], data['labels_val'],
        best_hyperparameters,
        epochs=epochs
    )
    
    # Save models
    best_model.save(f'{save_path}best_multimodal_model.h5')
    image_encoder.save(f'{save_path}image_encoder.h5')
    audio_encoder.save(f'{save_path}audio_encoder.h5')
    
    print(f"Models saved to {save_path}")
    
    return best_model, best_hyperparameters, history


def generate_embeddings(image_encoder, audio_encoder, image_data, audio_data, save_path='../results/'):
    """
    Generate and save embeddings from trained encoders.
    
    Args:
        image_encoder: Trained image encoder
        audio_encoder: Trained audio encoder
        image_data: Image data to encode
        audio_data: Audio data to encode
        save_path (str): Path to save embeddings
        
    Returns:
        tuple: (image_embeddings, audio_embeddings)
    """
    # Generate embeddings
    image_embeddings = image_encoder.predict(image_data)
    audio_embeddings = audio_encoder.predict(audio_data)
    
    # Save embeddings
    np.save(f'{save_path}image_embeddings.npy', image_embeddings)
    np.save(f'{save_path}audio_embeddings.npy', audio_embeddings)
    
    print(f"Embeddings saved to {save_path}")
    
    return image_embeddings, audio_embeddings


if __name__ == "__main__":
    # Example usage
    from preprocess import preprocess_data
    
    # Load and preprocess data
    data = preprocess_data()
    
    # Train model
    best_model, best_hyperparameters, history = train_multimodal_model(data)
    
    print("Training completed successfully!")
    print(f"Final validation accuracy: {max(history.history['val_accuracy'])}")
