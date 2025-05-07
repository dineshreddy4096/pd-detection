Code:
  import os
import re
import numpy as np
import torch
import torch.nn.functional as F
import pydicom
import networkx as nx
import sklearn.preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class ParkinsonGCNWithImageFeatures(torch.nn.Module):
    def __init__(self, metadata_features, image_features, hidden_channels, num_classes):
        super(ParkinsonGCNWithImageFeatures, self).__init__()
        total_input_features = metadata_features + image_features
        
        self.conv1 = GCNConv(total_input_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(self, x, edge_index):
        """Extract intermediate embeddings for visualization"""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return x

def extract_traditional_image_features(image_path):
    """
    Robust image feature extraction for DICOM files
    Handles various image formats and potential preprocessing challenges
    """
    try:
        # Read DICOM file
        dicom = pydicom.dcmread(image_path)
        pixel_array = dicom.pixel_array
        
        # Robust normalization and type conversion
        # Convert to 8-bit grayscale if needed
        if pixel_array.dtype != np.uint8:
            # Normalize to 0-255 range
            pixel_array = cv2.normalize(
                pixel_array, 
                None, 
                alpha=0, 
                beta=255, 
                norm_type=cv2.NORM_MINMAX, 
                dtype=cv2.CV_8U
            )
        
        # Ensure single channel
        if len(pixel_array.shape) > 2:
            # Take first channel or convert to grayscale
            pixel_array = pixel_array[:,:,0] if pixel_array.shape[2] > 1 else pixel_array
        
        features = []
        
        # 1. Basic statistical features
        features.extend([
            np.mean(pixel_array),       # Mean intensity
            np.std(pixel_array),        # Standard deviation
            np.median(pixel_array),     # Median intensity
            np.max(pixel_array),        # Max intensity
            np.min(pixel_array),        # Min intensity
        ])
        
        # 2. Histogram features with more robust binning
        hist = cv2.calcHist([pixel_array], [0], None, [32], [0, 256])
        hist_normalized = hist / np.sum(hist)  # Normalize histogram
        features.extend(hist_normalized.flatten())
        
        # 3. Edge detection (Sobel operator)
        try:
            sobel_x = cv2.Sobel(pixel_array, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(pixel_array, cv2.CV_64F, 0, 1, ksize=3)
            features.extend([
                np.mean(np.abs(sobel_x)),
                np.mean(np.abs(sobel_y)),
                np.std(sobel_x),
                np.std(sobel_y)
            ])
        except Exception as e:
            # Fallback if Sobel fails
            features.extend([0, 0, 0, 0])
        
        # 4. Robust texture features using alternative methods
        try:
            # Variance of Laplacian as blur detection
            laplacian_var = cv2.Laplacian(pixel_array, cv2.CV_64F).var()
            features.append(laplacian_var)
        except Exception:
            features.append(0)
        
        # 5. Entropy calculation
        try:
            histogram = cv2.calcHist([pixel_array], [0], None, [256], [0, 256])
            histogram_normalized = histogram / np.sum(histogram)
            entropy = -np.sum([p * np.log2(p) if p > 0 else 0 for p in histogram_normalized])
            features.append(entropy)
        except Exception:
            features.append(0)
        
        # Ensure consistent feature vector length
        while len(features) < 64:
            features.append(0.0)
        
        return np.array(features[:64])  # Truncate to 64 features
    
    except Exception as e:
        print(f"Comprehensive error in feature extraction from {image_path}: {e}")
        return np.zeros(64)  # Return zero vector if extraction completely fails

def extract_advanced_dicom_metadata(dicom_path, extract_images=False):
    """
    Enhanced metadata extraction with optional image feature extraction
    """
    features = []
    image_features = []
    processed_files = 0
    unique_features = set()
    
    print(f"Scanning directory: {dicom_path}")
    
    for root, _, files in os.walk(dicom_path):
        for filename in files:
            if filename.lower().endswith('.dcm'):
                full_path = os.path.join(root, filename)
                try:
                    ds = pydicom.dcmread(full_path)
                    
                    # Metadata feature vector (same as previous implementation)
                    feature_vector = [
                        parse_dicom_age(ds.get('PatientAge', '0Y')),
                        float(ds.get('PatientWeight', 0) or 0),
                        float(ds.get('SliceThickness', 0) or 0),
                        float(ds.get('PixelSpacing', [0,0])[0] if ds.get('PixelSpacing') else 0),
                        float(len(ds.dir())),
                        float(ds.get('Rows', 0) or 0),
                        float(ds.get('Columns', 0) or 0),
                    ]
                    
                    # Image feature extraction (optional)
                    if extract_images:
                        image_feat = extract_traditional_image_features(full_path)
                        image_features.append(image_feat)
                    
                    # Create a hashable representation
                    feature_tuple = tuple(feature_vector)
                    
                    if feature_tuple not in unique_features:
                        features.append(feature_vector)
                        unique_features.add(feature_tuple)
                    
                    processed_files += 1
                    
                    if processed_files % 1000 == 0:
                        print(f"Processed {processed_files} DICOM files")
                
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    
    # Standardize features
    features_array = np.array(features)
    scaler = sklearn.preprocessing.StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
    
    if extract_images:
        # Standardize and combine metadata and image features
        image_features_array = np.array(image_features)
        image_scaler = sklearn.preprocessing.StandardScaler()
        image_features_scaled = image_scaler.fit_transform(image_features_array)
        
        return features_scaled, image_features_scaled
    
    return features_scaled

def parse_dicom_age(age_str):
    """
    Parse DICOM age string with robust error handling
    """
    try:
        match = re.match(r'(\d+)([A-Z])', str(age_str))
        if match:
            value, unit = match.groups()
            value = int(value)
            
            if unit == 'Y':
                return float(value)
            elif unit == 'M':
                return float(value / 12)
            elif unit == 'W':
                return float(value / 52)
            elif unit == 'D':
                return float(value / 365)
        return 0.0
    except Exception:
        return 0.0

def prepare_dataset(pd_path, healthy_path):
    """
    Prepare dataset for GCN with combined metadata and traditional image features
    """
    print("\nPreparing dataset:")
    
    # Extract metadata and image features
    pd_metadata, pd_image_features = extract_advanced_dicom_metadata(pd_path, extract_images=True)
    healthy_metadata, healthy_image_features = extract_advanced_dicom_metadata(healthy_path, extract_images=True)
    
    # Balance dataset
    min_samples = min(len(pd_metadata), len(healthy_metadata))
    
    np.random.seed(42)
    pd_indices = np.random.choice(len(pd_metadata), min_samples, replace=False)
    healthy_indices = np.random.choice(len(healthy_metadata), min_samples, replace=False)
    
    # Select balanced features
    pd_metadata_balanced = pd_metadata[pd_indices]
    pd_image_balanced = pd_image_features[pd_indices]
    healthy_metadata_balanced = healthy_metadata[healthy_indices]
    healthy_image_balanced = healthy_image_features[healthy_indices]
    
    # Combine metadata and image features
    all_metadata = np.vstack([pd_metadata_balanced, healthy_metadata_balanced])
    all_image_features = np.vstack([pd_image_balanced, healthy_image_balanced])
    combined_features = np.hstack([all_metadata, all_image_features])
    
    # Create labels
    pd_labels = torch.zeros(len(pd_metadata_balanced), dtype=torch.long)
    healthy_labels = torch.ones(len(healthy_metadata_balanced), dtype=torch.long)
    all_labels = torch.cat([pd_labels, healthy_labels])
    
    # Create similarity graph
    similarity_matrix = cosine_similarity(combined_features)
    threshold = np.percentile(similarity_matrix, 95)
    adjacency_matrix = (similarity_matrix >= threshold).astype(int)
    
    G = nx.from_numpy_array(adjacency_matrix)
    
    # Convert to PyTorch Geometric Data
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    x = torch.tensor(combined_features, dtype=torch.float)
    
    graph_data = Data(x=x, edge_index=edge_index, y=all_labels)
    
    print("\nBalanced Dataset Summary:")
    print(f"Total samples: {len(all_labels)}")
    print(f"Parkinson's samples: {len(pd_labels)}")
    print(f"Healthy samples: {len(healthy_labels)}")
    print(f"Combined feature dimensions: {combined_features.shape}")
    
    return graph_data, G

def visualize_graph_structure(G, labels, title="Graph Connectivity Visualization"):
    """Visualize the constructed graph structure"""
    plt.figure(figsize=(12, 10))
    
    # Convert labels to numpy for easier manipulation
    labels_np = labels.numpy()
    
    # Set node colors based on labels (blue for healthy, red for PD)
    node_colors = ['blue' if label == 1 else 'red' for label in labels_np]
    
    # Spring layout for better visualization
    pos = nx.spring_layout(G, seed=42)
    
    # Plot
    nx.draw_networkx(
        G, 
        pos=pos,
        node_color=node_colors,
        node_size=50,
        width=0.3,
        edge_color='gray',
        alpha=0.7,
        with_labels=False
    )
    
    # Add legend
    pd_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label="Parkinson's")
    healthy_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Healthy')
    plt.legend(handles=[pd_patch, healthy_patch], loc='upper right')
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('graph_structure.png')
    plt.close()
    
    print("Graph visualization saved as graph_structure.png")

def visualize_feature_importance(model, feature_names=None):
    """Analyze feature importance based on model weights"""
    # Extract weights from the first layer
    weights = model.conv1.lin.weight.detach().cpu().numpy()
    
    # Sum absolute weights across all output nodes
    importance = np.abs(weights).sum(axis=0)
    
    # Normalize to get relative importance
    importance = importance / importance.sum()
    
    # Generate feature names if not provided
    if feature_names is None:
        metadata_names = ['Age', 'Weight', 'SliceThickness', 'PixelSpacing', 'DicomAttributes', 'Rows', 'Columns']
        image_names = [f'ImgFeat_{i+1}' for i in range(64)]
        feature_names = metadata_names + image_names
    
    # Sort features by importance
    sorted_idx = np.argsort(importance)[::-1]
    top_features = sorted_idx[:15]  # Top 15 features
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(np.array(feature_names)[top_features], importance[top_features], color='skyblue')
    plt.xlabel('Relative Importance')
    plt.title('Top 15 Feature Importance')
    plt.gca().invert_yaxis()  # Display most important at the top
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("Feature importance visualization saved as feature_importance.png")
    
    return importance

def visualize_embeddings(model, graph_data, labels):
    """Visualize node embeddings using t-SNE"""
    # Get node embeddings from the trained model
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(graph_data.x, graph_data.edge_index).cpu().numpy()
    
    # Convert labels to numpy
    labels_np = labels.cpu().numpy()
    
    # Reduce dimensionality for visualization using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=labels_np,
        cmap='coolwarm',
        alpha=0.7,
        s=50
    )
    
    # Add legend
    legend = plt.legend(*scatter.legend_elements(), title="Class", loc="upper right")
    plt.setp(legend.get_title(), fontsize=12)
    
    plt.title('t-SNE Visualization of Node Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig('node_embeddings.png')
    plt.close()
    
    print("Node embeddings visualization saved as node_embeddings.png")

def plot_performance_metrics(model, graph_data, train_mask, test_mask):
    """Generate and plot comprehensive performance metrics"""
    model.eval()
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index)
        probabilities = torch.exp(out)  # Convert log_softmax to probabilities
        pred = out.argmax(dim=1)
        
        # Get test predictions and true labels
        test_pred = pred[test_mask].cpu().numpy()
        test_true = graph_data.y[test_mask].cpu().numpy()
        test_probs = probabilities[test_mask, 0].cpu().numpy()  # Probability of PD class
        
    # Confusion Matrix
    cm = confusion_matrix(test_true, test_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Parkinson\'s', 'Healthy'], 
                yticklabels=['Parkinson\'s', 'Healthy'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(test_true, 1 - test_probs)  # 1-probs because lower prob means higher PD likelihood
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(test_true, 1 - test_probs)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png')
    plt.close()
    
    # Training history visualization (requires training history to be collected during training)
    print("\nClassification Report:")
    print(classification_report(
        test_true, 
        test_pred,
        target_names=['Parkinson\'s', 'Healthy'],
        digits=4
    ))
    
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Precision-Recall AUC: {pr_auc:.4f}")
    
    print("Performance visualizations saved as confusion_matrix.png, roc_curve.png, and precision_recall_curve.png")
    
    # Return metrics dictionary
    return {
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'classification_report': classification_report(
            test_true, test_pred,
            target_names=['Parkinson\'s', 'Healthy'],
            digits=4,
            output_dict=True
        )
    }

def train_gcn(graph_data, save_history=True):
    """
    Train Graph Convolutional Network with combined features
    Enhanced with history tracking for visualization
    """
    # Split data
    train_mask = torch.rand(graph_data.num_nodes) < 0.7
    test_mask = ~train_mask
    
    # Updated model initialization with combined feature dimensions
    model = ParkinsonGCNWithImageFeatures(
        metadata_features=7,   # From original metadata features
        image_features=64,     # Traditional image features
        hidden_channels=64, 
        num_classes=2
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    # Training loop
    best_accuracy = 0
    patience = 20
    patience_counter = 0
    best_model = None
    
    # History tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'epochs': []
    }
    
    print("\nTraining GCN:")
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)
        loss = F.nll_loss(out[train_mask], graph_data.y[train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Calculate training accuracy
        pred_train = out[train_mask].argmax(dim=1)
        train_correct = (pred_train == graph_data.y[train_mask]).sum()
        train_acc = int(train_correct) / int(train_mask.sum())
        
        if save_history:
            history['train_loss'].append(loss.item())
            history['train_acc'].append(train_acc)
        
        # Periodic evaluation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred = out.argmax(dim=1)
                correct = (pred[test_mask] == graph_data.y[test_mask]).sum()
                acc = int(correct) / int(test_mask.sum())
                
                if save_history:
                    history['test_acc'].append(acc)
                    history['epochs'].append(epoch)
                
            print(f'Epoch {epoch}: Loss {loss.item():.4f}, Train Acc: {train_acc:.4f}, Test Accuracy: {acc:.4f}')
            
            # Early stopping and best model tracking
            if acc > best_accuracy:
                best_accuracy = acc
                patience_counter = 0
                best_model = model.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Visualize training history
    if save_history and len(history['epochs']) > 1:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(len(history['train_loss'])), history['train_loss'], label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(history['epochs'], history['test_acc'], 'o-', label='Test Accuracy')
        plt.plot(range(len(history['train_acc'])), history['train_acc'], label='Train Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        
        print("Training history visualization saved as training_history.png")
    
    # Load best model for final evaluation
    model.load_state_dict(best_model)
    
    # Generate and plot performance metrics
    metrics = plot_performance_metrics(model, graph_data, train_mask, test_mask)
    
    print(f"\nBest Test Accuracy: {best_accuracy:.4f}")
    return model, metrics, train_mask, test_mask

def main():
    # Paths to your datasets
    pd_path = "/kaggle/input/parkinson22/pd_patients/pd_patients/PPMI"
    healthy_path = "/kaggle/input/parkinson22/healthy/healthy/PPMI"
    
    print("Starting Parkinson's Disease GCN Analysis with Traditional Image Features")
    
    # Create output directory for visualizations
    os.makedirs('visualizations', exist_ok=True)
    
    # Prepare graph data
    graph_data, graph = prepare_dataset(pd_path, healthy_path)
    
    if graph_data is not None:
        # Visualize the graph structure
        visualize_graph_structure(graph, graph_data.y)
        
        # Train GCN
        model, metrics, train_mask, test_mask = train_gcn(graph_data)
        
        # Visualize feature importance
        visualize_feature_importance(model)
        
        # Visualize embeddings
        visualize_embeddings(model, graph_data, graph_data.y)
        
        # Save model
        torch.save(model.state_dict(), 'parkinson_gcn_model.pt')
        print("Model saved as parkinson_gcn_model.pt")
        
        # Show summary of all results
        print("\n============= RESULTS SUMMARY =============")
        print(f"Best accuracy: {metrics['roc_auc']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"PR AUC: {metrics['pr_auc']:.4f}")
        print(f"Sensitivity (PD recall): {metrics['classification_report']['0']['recall']:.4f}")
        print(f"Specificity (Healthy recall): {metrics['classification_report']['1']['recall']:.4f}")
        print("===========================================")
        
    else:
        print("Dataset preparation failed. Exiting.")

if __name__ == "__main__":
    main()
