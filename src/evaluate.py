"""
Evaluation module for multimodal digit classification.

This module handles model evaluation, visualization, clustering analysis,
and prediction generation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, homogeneity_score
import json


def visualize_embeddings_tsne(embeddings, labels, title, save_path=None):
    """
    Create t-SNE visualization of embeddings.
    
    Args:
        embeddings: Embedding vectors
        labels: True labels for coloring
        title (str): Title for the plot
        save_path (str): Path to save the plot
        
    Returns:
        numpy.ndarray: t-SNE transformed embeddings
    """
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embeddings_tsne[:, 0], 
        y=embeddings_tsne[:, 1], 
        hue=labels, 
        palette='tab10'
    )
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Labels')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return embeddings_tsne


def perform_clustering_analysis(embeddings, labels, n_clusters=10, random_state=42):
    """
    Perform K-means clustering analysis on embeddings.
    
    Args:
        embeddings: Embedding vectors
        labels: True labels
        n_clusters (int): Number of clusters
        random_state (int): Random seed
        
    Returns:
        tuple: (cluster_labels, homogeneity_score)
    """
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Calculate homogeneity score
    homogeneity = homogeneity_score(labels, cluster_labels)
    
    return cluster_labels, homogeneity


def plot_confusion_matrix(true_labels, predicted_labels, title, save_path=None):
    """
    Plot confusion matrix for clustering results.
    
    Args:
        true_labels: True labels
        predicted_labels: Predicted cluster labels
        title (str): Title for the plot
        save_path (str): Path to save the plot
    """
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    # Create visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', cbar=False)
    plt.title(title)
    plt.xlabel('Cluster')
    plt.ylabel('True Digit')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def evaluate_embeddings(image_embeddings, audio_embeddings, labels_train, save_path='../results/'):
    """
    Complete evaluation pipeline for embeddings.
    
    Args:
        image_embeddings: Image embeddings
        audio_embeddings: Audio embeddings
        labels_train: Training labels
        save_path (str): Path to save results
        
    Returns:
        dict: Evaluation results
    """
    results = {}
    
    # t-SNE visualization for image embeddings
    print("Creating t-SNE visualization for image embeddings...")
    image_tsne = visualize_embeddings_tsne(
        image_embeddings, 
        labels_train['label'], 
        't-SNE Visualization of Image Embeddings',
        f'{save_path}image_tsne_visualization.png'
    )
    
    # t-SNE visualization for audio embeddings
    print("Creating t-SNE visualization for audio embeddings...")
    audio_tsne = visualize_embeddings_tsne(
        audio_embeddings, 
        labels_train['label'], 
        't-SNE Visualization of Audio Embeddings',
        f'{save_path}audio_tsne_visualization.png'
    )
    
    # Clustering analysis for image embeddings
    print("Performing clustering analysis for image embeddings...")
    img_clusters, img_homogeneity = perform_clustering_analysis(
        image_embeddings, labels_train['label']
    )
    
    # Clustering analysis for audio embeddings
    print("Performing clustering analysis for audio embeddings...")
    audio_clusters, audio_homogeneity = perform_clustering_analysis(
        audio_embeddings, labels_train['label']
    )
    
    # Plot confusion matrices
    print("Creating confusion matrices...")
    plot_confusion_matrix(
        labels_train['label'], img_clusters,
        'Image Clustering Confusion Matrix',
        f'{save_path}image_confusion_matrix.png'
    )
    
    plot_confusion_matrix(
        labels_train['label'], audio_clusters,
        'Audio Clustering Confusion Matrix',
        f'{save_path}audio_confusion_matrix.png'
    )
    
    # Store results
    results = {
        'image_homogeneity': img_homogeneity,
        'audio_homogeneity': audio_homogeneity,
        'image_clusters': img_clusters.tolist(),
        'audio_clusters': audio_clusters.tolist()
    }
    
    # Save results to JSON
    with open(f'{save_path}evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {save_path}")
    print(f"Image clustering homogeneity score: {img_homogeneity:.4f}")
    print(f"Audio clustering homogeneity score: {audio_homogeneity:.4f}")
    
    return results


def generate_predictions(model, image_test, audio_test, save_path='../preds/'):
    """
    Generate predictions on test data and save to CSV.
    
    Args:
        model: Trained model
        image_test: Test image data
        audio_test: Test audio data
        save_path (str): Path to save predictions
        
    Returns:
        pandas.DataFrame: Predictions dataframe
    """
    # Generate predictions
    print("Generating predictions on test data...")
    predicted_probs = model.predict([image_test, audio_test])
    predicted_labels = predicted_probs.argmax(axis=1)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'row_id': range(len(predicted_labels)),
        'Digit_Probability': predicted_labels
    })
    
    # Save to CSV
    submission_df.to_csv(f'{save_path}final_preds.csv', index=False)
    
    print(f"Predictions saved to {save_path}final_preds.csv")
    
    return submission_df


def evaluate_model_performance(model, image_val, audio_val, labels_val):
    """
    Evaluate model performance on validation data.
    
    Args:
        model: Trained model
        image_val: Validation image data
        audio_val: Validation audio data
        labels_val: Validation labels
        
    Returns:
        dict: Performance metrics
    """
    # Evaluate model
    loss, accuracy = model.evaluate([image_val, audio_val], labels_val['label'], verbose=0)
    
    # Generate predictions for detailed analysis
    predictions = model.predict([image_val, audio_val])
    predicted_labels = predictions.argmax(axis=1)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels_val['label'], predicted_labels)
    
    metrics = {
        'validation_loss': float(loss),
        'validation_accuracy': float(accuracy),
        'confusion_matrix': conf_matrix.tolist()
    }
    
    return metrics


if __name__ == "__main__":
    # Example usage
    import numpy as np
    from preprocess import preprocess_data
    from train import train_multimodal_model, generate_embeddings
    
    # Load and preprocess data
    data = preprocess_data()
    
    # Train model
    best_model, best_hyperparameters, history = train_multimodal_model(data)
    
    # Generate embeddings
    image_encoder, audio_encoder, _, _ = create_models()
    image_embeddings, audio_embeddings = generate_embeddings(
        image_encoder, audio_encoder, 
        data['image_train'], data['audio_train']
    )
    
    # Evaluate embeddings
    results = evaluate_embeddings(image_embeddings, audio_embeddings, data['labels_train'])
    
    # Generate predictions
    predictions = generate_predictions(best_model, data['image_test'], data['audio_test'])
    
    print("Evaluation completed successfully!")
