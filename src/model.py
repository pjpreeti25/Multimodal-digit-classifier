"""
Model architecture module for multimodal digit classification.

This module defines the encoder models for image and audio data,
as well as the combined multimodal model (matches original notebook).
"""

import keras
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
from tensorflow.keras.models import Model


def create_image_encoder(input_shape=(28, 28), embedding_dim=64):
    """
    Create image encoder model (matches original notebook).
    
    Args:
        input_shape (tuple): Shape of input images
        embedding_dim (int): Dimension of output embedding
        
    Returns:
        keras.Model: Image encoder model
    """
    image_input = Input(shape=input_shape)
    image_flatten = Flatten()(image_input)
    image_encoder_output = Dense(embedding_dim, activation='relu')(image_flatten)
    image_encoder = Model(image_input, image_encoder_output)
    
    return image_encoder


def create_audio_encoder(input_shape=(507,), embedding_dim=64):
    """
    Create audio encoder model (matches original notebook).
    
    Args:
        input_shape (tuple): Shape of input audio data
        embedding_dim (int): Dimension of output embedding
        
    Returns:
        keras.Model: Audio encoder model
    """
    audio_input = Input(shape=input_shape)
    audio_encoder_output = Dense(embedding_dim, activation='relu')(audio_input)
    audio_encoder = Model(audio_input, audio_encoder_output)
    
    return audio_encoder


def create_combined_model(image_encoder, audio_encoder, num_classes=10):
    """
    Create complete multimodal classification model (matches original notebook).
    
    Args:
        image_encoder: Image encoder model
        audio_encoder: Audio encoder model
        num_classes (int): Number of output classes
        
    Returns:
        keras.Model: Complete multimodal model
    """
    # Get the input layers from individual encoders
    image_input = image_encoder.input
    audio_input = audio_encoder.input
    
    # Get the output layers from individual encoders
    image_encoder_output = image_encoder.output
    audio_encoder_output = audio_encoder.output
    
    # Concatenate the outputs
    combined_encoder_output = Concatenate()([image_encoder_output, audio_encoder_output])
    
    # Add classification layers (matches original notebook)
    x = Dense(128, activation='relu')(combined_encoder_output)
    x = Dense(64, activation='relu')(x)
    classification_output = Dense(num_classes, activation='softmax')(x)
    
    # Create the complete model
    combined_model = Model([image_input, audio_input], classification_output)
    
    return combined_model


def create_models(embedding_dim=64, num_classes=10):
    """
    Create all models needed for multimodal classification (matches original notebook).
    
    Args:
        embedding_dim (int): Dimension of encoder embeddings
        num_classes (int): Number of output classes
        
    Returns:
        tuple: (image_encoder, audio_encoder, multimodal_model)
    """
    # Create individual encoders
    image_encoder = create_image_encoder(embedding_dim=embedding_dim)
    audio_encoder = create_audio_encoder(embedding_dim=embedding_dim)
    
    # Create complete multimodal model
    multimodal_model = create_combined_model(image_encoder, audio_encoder, num_classes)
    
    return image_encoder, audio_encoder, multimodal_model


def compile_model(model, optimizer='adam', learning_rate=0.001, loss='sparse_categorical_crossentropy'):
    """
    Compile the model with specified optimizer and loss function (matches original notebook).
    
    Args:
        model: Model to compile
        optimizer (str): Optimizer name ('adam', 'sgd', 'rmsprop')
        learning_rate (float): Learning rate for optimizer
        loss (str): Loss function
        
    Returns:
        keras.Model: Compiled model
    """
    if optimizer.lower() == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer.lower() == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    
    return model


if __name__ == "__main__":
    # Example usage
    image_encoder, audio_encoder, multimodal_model = create_models()
    
    print("Models created successfully!")
    print(f"Image encoder summary:")
    image_encoder.summary()
    print(f"\nAudio encoder summary:")
    audio_encoder.summary()
    print(f"\nMultimodal model summary:")
    multimodal_model.summary()
