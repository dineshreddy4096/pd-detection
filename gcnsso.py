!pip install torch_geometric
!pip install scikit-image

import os
import re
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import pydicom
import networkx as nx
import sklearn.preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis, entropy

class ParkinsonGCN(torch.nn.Module):
    def _init_(self, num_features, hidden_channels, num_classes):
        super(ParkinsonGCN, self)._init_()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class SalpSwarmOptimizer:
    def _init_(self, num_salps, num_dimensions, lower_bounds, upper_bounds):
        self.num_salps = num_salps
        self.num_dimensions = num_dimensions
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.positions = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(num_salps, num_dimensions))
        self.best_position = None
        self.best_fitness = float('-inf')
    
    def fitness_function(self, model, graph_data, device):
        num_nodes = graph_data.num_nodes
        perm = torch.randperm(num_nodes)
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_idx = perm[:train_size]
        val_idx = perm[train_size:train_size + val_size]
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        
        model = model.to(device)
        graph_data = graph_data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.current_lr)
        
        for _ in range(10):
            model.train()
            optimizer.zero_grad()
            out = model(graph_data.x, graph_data.edge_index)
            loss = F.nll_loss(out[train_mask], graph_data.y[train_mask])
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            out = model(graph_data.x, graph_data.edge_index)
            pred = out.argmax(dim=1)
            correct = (pred[val_mask] == graph_data.y[val_mask]).sum()
            accuracy = int(correct) / int(val_mask.sum())
        
        return accuracy
    
    def optimize(self, model_class, graph_data, max_iterations=20, device='cpu'):
        print("Starting SSO Optimization...")
        for iteration in range(max_iterations):
            c1 = 2 * np.exp(-((4 * iteration / max_iterations) ** 2))
            
            for i in range(self.num_salps):
                if i == 0:
                    for j in range(self.num_dimensions):
                        c2 = random.random()
                        c3 = random.random()
                        if c3 < 0.5:
                            self.positions[i, j] = np.clip(
                                self.positions[i, j] + c1 * ((self.upper_bounds[j] - self.lower_bounds[j]) * c2 + self.lower_bounds[j]),
                                self.lower_bounds[j],
                                self.upper_bounds[j]
                            )
                        else:
                            self.positions[i, j] = np.clip(
                                self.positions[i, j] - c1 * ((self.upper_bounds[j] - self.lower_bounds[j]) * c2 + self.lower_bounds[j]),
                                self.lower_bounds[j],
                                self.upper_bounds[j]
                            )
                else:
                    self.positions[i, :] = (self.positions[i, :] + self.positions[i-1, :]) / 2
                
                self.current_lr = self.positions[i, 0]
                hidden_channels = int(self.positions[i, 1])
                
                model = model_class(
                    num_features=graph_data.x.shape[1], 
                    hidden_channels=hidden_channels, 
                    num_classes=2
                )
                
                try:
                    fitness = self.fitness_function(model, graph_data, device)
                    if fitness > self.best_fitness:
                        self.best_fitness = fitness
                        self.best_position = self.positions[i].copy()
                    print(f"Iteration {iteration+1}/{max_iterations}, Salp {i+1}: LR={self.current_lr:.4f}, "
                          f"Hidden Channels={hidden_channels}, Accuracy={fitness:.4f}")
                except RuntimeError as e:
                    print(f"Error in fitness evaluation: {e}")
                    fitness = float('-inf')
        
        return self.best_position, self.best_fitness

def parse_dicom_age(age_str):
    try:
        match = re.match(r'(\d+)([A-Z])', str(age_str))
        if match:
            value, unit = match.groups()
            value = int(value)
            if unit == 'Y': return float(value)
            elif unit == 'M': return float(value / 12)
            elif unit == 'W': return float(value / 52)
            elif unit == 'D': return float(value / 365)
        return 0.0
    except Exception:
        return 0.0

def extract_advanced_dicom_metadata(dicom_path):
    features = []
    processed_files = 0
    unique_features = set()
    
    print(f"Scanning directory: {dicom_path}")
    feature_names = [
        'PatientAge', 'PatientWeight', 'SliceThickness', 'PixelSpacingX', 'DicomAttributeCount', 
        'ImageRows', 'ImageColumns', 'PixelMean', 'PixelStd', 'PixelSkew',
        'GLCM_Contrast', 'GLCM_Dissimilarity', 'GLCM_Homogeneity', 'GLCM_Energy',
        'PixelEntropy', 'PixelKurtosis', 'PixelEnergy'
    ]
    
    for root, _, files in os.walk(dicom_path):
        for filename in files:
            if filename.lower().endswith('.dcm'):
                full_path = os.path.join(root, filename)
                try:
                    ds = pydicom.dcmread(full_path)
                    metadata_features = [
                        parse_dicom_age(ds.get('PatientAge', '0Y')),
                        float(ds.get('PatientWeight', 0) or 0),
                        float(ds.get('SliceThickness', 0) or 0),
                        float(ds.get('PixelSpacing', [0,0])[0] if ds.get('PixelSpacing') else 0),
                        float(len(ds.dir())),
                        float(ds.get('Rows', 0) or 0),
                        float(ds.get('Columns', 0) or 0),
                    ]
                    if hasattr(ds, 'pixel_array'):
                        pixel_data = ds.pixel_array.astype(float)
                        if pixel_data.ndim > 2:
                            pixel_data = pixel_data[0] if pixel_data.shape[0] > 1 else pixel_data.mean(axis=0)
                        pixel_range = pixel_data.max() - pixel_data.min()
                        if pixel_range > 0:
                            pixel_data_normalized = (pixel_data - pixel_data.min()) / pixel_range * 255
                        else:
                            pixel_data_normalized = np.zeros_like(pixel_data)
                        pixel_data_normalized = pixel_data_normalized.astype(np.uint8)
                        
                        # Statistical features
                        pixel_mean = np.mean(pixel_data)
                        pixel_std = np.std(pixel_data) if np.std(pixel_data) > 0 else 0.0
                        pixel_skew = skew(pixel_data.flatten()) if not np.isnan(skew(pixel_data.flatten())) else 0.0
                        
                        # GLCM features
                        glcm = graycomatrix(pixel_data_normalized, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
                        glcm_contrast = graycoprops(glcm, 'contrast')[0, 0]
                        glcm_dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
                        glcm_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                        glcm_energy = graycoprops(glcm, 'energy')[0, 0]
                        
                        # Additional image features
                        pixel_entropy = entropy(pixel_data.flatten()) if np.any(pixel_data.flatten()) else 0.0
                        pixel_kurtosis = kurtosis(pixel_data.flatten()) if not np.isnan(kurtosis(pixel_data.flatten())) else 0.0
                        pixel_energy = np.sum(pixel_data ** 2) / pixel_data.size  # Normalized energy
                        
                        image_features = [
                            pixel_mean, pixel_std, pixel_skew,
                            glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy,
                            pixel_entropy, pixel_kurtosis, pixel_energy
                        ]
                    else:
                        image_features = [0.0] * 10  # Match length of image features
                    
                    feature_vector = metadata_features + image_features
                    feature_vector = [0.0 if np.isnan(x) else x for x in feature_vector]
                    feature_tuple = tuple(feature_vector)
                    if feature_tuple not in unique_features:
                        features.append(feature_vector)
                        unique_features.add(feature_tuple)
                    processed_files += 1
                    if processed_files % 1000 == 0:
                        print(f"Processed {processed_files} DICOM files")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    
    print(f"\nTotal unique files processed: {len(features)}")
    features_array = np.array(features)
    print(f"Feature shape: {features_array.shape}")
    if np.any(np.isnan(features_array)):
        print("Warning: NaN values found in features_array. Replacing with 0.0")
        features_array = np.nan_to_num(features_array, nan=0.0)
    scaler = sklearn.preprocessing.StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
    if np.any(np.isnan(features_scaled)):
        print("Warning: NaN values found after scaling. Replacing with 0.0")
        features_scaled = np.nan_to_num(features_scaled, nan=0.0)
    return features_scaled

def create_graph_from_features(features):
    if len(features) == 0:
        print("No features to create graph!")
        return None
    
    if np.any(np.isnan(features)):
        print("Warning: NaN values found in features. Replacing with 0.0")
        features = np.nan_to_num(features, nan=0.0)
    
    similarity_matrix = cosine_similarity(features)
    threshold = np.percentile(similarity_matrix, 99)
    adjacency_matrix = (similarity_matrix >= threshold).astype(int)
    
    G = nx.from_numpy_array(adjacency_matrix)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    
    print(f"Graph - Nodes: {G.number_of_nodes()}")
    print(f"Graph - Edges: {G.number_of_edges()}")
    
    return Data(x=x, edge_index=edge_index)

def prepare_dataset(pd_path, healthy_path):
    print("\nPreparing dataset:")
    print("Extracting Parkinson's disease features...")
    pd_features = extract_advanced_dicom_metadata(pd_path)
    print("\nExtracting healthy control features...")
    healthy_features = extract_advanced_dicom_metadata(healthy_path)
    
    min_samples = min(len(pd_features), len(healthy_features))
    pd_indices = np.random.choice(len(pd_features), min_samples, replace=False)
    healthy_indices = np.random.choice(len(healthy_features), min_samples, replace=False)
    
    pd_features_balanced = pd_features[pd_indices]
    healthy_features_balanced = healthy_features[healthy_indices]
    
    all_features = np.vstack([pd_features_balanced, healthy_features_balanced])
    print(f"Combined feature shape before thresholding: {all_features.shape}")
    
    selector = VarianceThreshold(threshold=1e-6)
    all_features_selected = selector.fit_transform(all_features)
    print(f"Features after variance thresholding: {all_features_selected.shape}")
    
    pd_labels = torch.zeros(len(pd_features_balanced), dtype=torch.long)
    healthy_labels = torch.ones(len(healthy_features_balanced), dtype=torch.long)
    all_labels = torch.cat([pd_labels, healthy_labels])
    
    print("\nBalanced Dataset Summary:")
    print(f"Total samples: {len(all_labels)}")
    print(f"Parkinson's samples: {len(pd_labels)}")
    print(f"Healthy samples: {len(healthy_labels)}")
    
    graph_data = create_graph_from_features(all_features_selected)
    if graph_data is not None:
        graph_data.y = all_labels
    return graph_data

def optimized_train_gcn(graph_data, best_params, device='cpu'):
    best_lr, best_hidden_channels = best_params
    model = ParkinsonGCN(
        num_features=graph_data.x.shape[1], 
        hidden_channels=int(best_hidden_channels), 
        num_classes=2
    ).to(device)
    graph_data = graph_data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    
    num_nodes = graph_data.num_nodes
    perm = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    best_val_accuracy = 0
    patience = 20
    patience_counter = 0
    best_model = None
    
    train_losses, train_accs, val_accs, test_accs = [], [], [], []
    train_aucs, val_aucs, test_aucs = [], [], []
    epochs = []
    
    print("\nTraining Optimized GCN:")
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)
        loss = F.nll_loss(out[train_mask], graph_data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            probs = torch.exp(out)[:, 1]
            
            train_acc = (pred[train_mask] == graph_data.y[train_mask]).sum().item() / train_mask.sum().item()
            val_acc = (pred[val_mask] == graph_data.y[val_mask]).sum().item() / val_mask.sum().item()
            test_acc = (pred[test_mask] == graph_data.y[test_mask]).sum().item() / test_mask.sum().item()
            
            train_fpr, train_tpr, _ = roc_curve(graph_data.y[train_mask].cpu(), probs[train_mask].cpu())
            val_fpr, val_tpr, _ = roc_curve(graph_data.y[val_mask].cpu(), probs[val_mask].cpu())
            test_fpr, test_tpr, _ = roc_curve(graph_data.y[test_mask].cpu(), probs[test_mask].cpu())
            
            train_auc = auc(train_fpr, train_tpr)
            val_auc = auc(val_fpr, val_tpr)
            test_auc = auc(test_fpr, test_tpr)
        
        epochs.append(epoch)
        train_losses.append(loss.item())
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        test_aucs.append(test_auc)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss {loss.item():.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, '
                  f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}')
        
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index)
        final_pred = out.argmax(dim=1)
        final_probs = torch.exp(out)
        
        test_predictions = final_pred[test_mask].cpu().numpy()
        test_labels = graph_data.y[test_mask].cpu().numpy()
        test_probs = final_probs[test_mask].cpu().numpy()[:, 1]
        
        final_train_acc = (final_pred[train_mask] == graph_data.y[train_mask]).sum().item() / train_mask.sum().item()
        final_val_acc = (final_pred[val_mask] == graph_data.y[val_mask]).sum().item() / val_mask.sum().item()
        final_test_acc = (final_pred[test_mask] == graph_data.y[test_mask]).sum().item() / test_mask.sum().item()
        
        final_train_fpr, final_train_tpr, _ = roc_curve(graph_data.y[train_mask].cpu(), final_probs[train_mask, 1].cpu())
        final_val_fpr, final_val_tpr, _ = roc_curve(graph_data.y[val_mask].cpu(), final_probs[val_mask, 1].cpu())
        final_test_fpr, final_test_tpr, _ = roc_curve(graph_data.y[test_mask].cpu(), final_probs[test_mask, 1].cpu())
        
        final_train_auc = auc(final_train_fpr, final_train_tpr)
        final_val_auc = auc(final_val_fpr, final_val_tpr)
        final_test_auc = auc(final_test_fpr, final_test_tpr)
        
        print("\nFinal Results:")
        print(f"Final Train Accuracy: {final_train_acc:.4f}, Final Train AUC: {final_train_auc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}, Final Val AUC: {final_val_auc:.4f}")
        print(f"Final Test Accuracy: {final_test_acc:.4f}, Final Test AUC: {final_test_auc:.4f}")
        print("\nFinal Test Set Classification Report:")
        print(classification_report(test_labels, test_predictions, target_names=['Parkinson\'s', 'Healthy'], digits=4))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, label='Training Accuracy', marker='o', linestyle='-', color='b')
    plt.plot(epochs, val_accs, label='Validation Accuracy', marker='o', linestyle='-', color='g')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xticks([i for i in range(0, len(epochs), 10)])  # Adjust x-ticks for readability
    for i, (t_acc, v_acc) in enumerate(zip(train_accs[::10], val_accs[::10])):
        plt.text(epochs[i*10], t_acc + 0.01, f'{t_acc:.4f}', ha='center', color='b', fontsize=8)
        plt.text(epochs[i*10], v_acc - 0.03, f'{v_acc:.4f}', ha='center', color='g', fontsize=8)
    plt.tight_layout()
    plt.show()
    
    return model

def main():
    pd_path = "/kaggle/input/parkinson22/pd_patients/pd_patients/PPMI"
    healthy_path = "/kaggle/input/parkinson22/healthy/healthy/PPMI"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Starting Parkinson's Disease GCN Analysis with Salp Swarm Optimization")
    print("=" * 50)
    
    graph_data = prepare_dataset(pd_path, healthy_path)
    
    if graph_data is not None:
        print(f"Graph feature dimensionality: {graph_data.x.shape[1]}")
        sso = SalpSwarmOptimizer(num_salps=5, num_dimensions=2, lower_bounds=[0.001, 8], upper_bounds=[0.1, 64])
        best_params, best_fitness = sso.optimize(model_class=ParkinsonGCN, graph_data=graph_data, device=device)
        
        print("\nBest Hyperparameters:")
        print(f"Learning Rate: {best_params[0]:.4f}")
        print(f"Hidden Channels: {int(best_params[1])}")
        print(f"Best Accuracy: {best_fitness:.4f}")
        
        optimized_model = optimized_train_gcn(graph_data, best_params, device=device)
    else:
        print("Dataset preparation failed. Exiting.")

if _name_ == "_main_":
    main()
