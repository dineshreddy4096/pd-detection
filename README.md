# pd-detection
Deep learning models (VGG19, GCN) with Salp Swarm Optimization for Parkinson’s Disease detection using MRI scans


Parkinson's Disease Detection
This repository implements a machine learning pipeline for detecting Parkinson's disease (PD) using DICOM imaging data from the PPMI dataset. The pipeline employs deep learning (VGG19-based models) and graph-based methods (Graph Convolutional Networks) with Salp Swarm Optimization (SSO) for hyperparameter tuning. This README ensures reproducibility and transparency per the specified format.
Reproducibility
Algorithms and Code Used
The pipeline includes four main scripts for PD detection, leveraging both convolutional neural networks (CNNs) and graph convolutional networks (GCNs):

vgg19.py (VGG19-based Model):

Purpose: Classifies DICOM images as PD or healthy using a pre-trained VGG19 model.
Steps:
Loads and preprocesses DICOM images (normalize to 0-255, resize to 128x128, convert to 3-channel RGB).
Splits data into train (60%), validation (20%), and test (20%) sets.
Builds a VGG19 model with frozen base layers, adding GlobalAveragePooling2D, Dense (512 units), Dropout (0.5), and softmax output for binary classification.
Trains with data augmentation (rotation, zoom, flip) for 25 epochs, using Adam optimizer (lr=0.0001), categorical crossentropy loss, and callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint).
Evaluates on test set (accuracy, AUC, classification report, confusion matrix).
Visualizes training history, confusion matrix, and ROC curve.


Outputs: parkinson_vgg19_final_model.h5, best_parkinson_model.keras, plots (training_history.png, auc_history.png, confusion_matrix.png, roc_curve.png).


vgg19Sso.py (VGG19 with Salp Swarm Optimization):

Purpose: Optimizes VGG19 hyperparameters using SSO for improved performance.
Steps:
Loads DICOM images in batches, preprocesses similarly to vgg19.py.
Splits data into train (70%), validation (15%), and test (15%) sets with stratification.
Uses SSO to optimize learning rate (1e-5 to 1e-3), dense units (128 to 512), and dropout rate (0.3 to 0.6) over 3 iterations with 5 salps.
Trains a VGG19 model with best hyperparameters, using data augmentation (rotation, zoom, translation) and early stopping (patience=5).
Evaluates on test set (accuracy, AUC, classification report, confusion matrix).
Visualizes training curves, confusion matrix, ROC curve, and classification metrics.


Outputs: pd_detection_model.h5, plots (training_curves.png, confusion_matrix.png, roc_curve.png, classification_metrics.png).


gcnsso.py (GCN with Salp Swarm Optimization):

Purpose: Uses a GCN to classify PD vs. healthy based on graph-structured data with SSO for hyperparameter tuning.
Steps:
Extracts advanced metadata (e.g., PatientAge, SliceThickness) and image features (e.g., GLCM, pixel statistics) from DICOM files.
Creates a graph using cosine similarity (threshold at 99th percentile) between feature vectors.
Balances PD and healthy samples, applies variance thresholding to features.
Uses SSO to optimize learning rate (0.001 to 0.1) and hidden channels (8 to 64) over 20 iterations.
Trains a GCN with two convolutional layers for 100 epochs with early stopping (patience=20).
Evaluates on test set (accuracy, AUC, classification report).
Visualizes training accuracy.


Outputs: Accuracy and AUC plots (displayed, not saved).


gcn.py (GCN with Image Features):

Purpose: Enhances GCN with traditional image features for improved classification.
Steps:
Extracts metadata (7 features) and traditional image features (64 features, e.g., histogram, Sobel edges, entropy) from DICOM files.
Creates a graph using cosine similarity (threshold at 95th percentile).
Balances dataset, combines metadata and image features.
Trains a GCN with two convolutional layers (metadata_features=7, image_features=64, hidden_channels=64) for 200 epochs with early stopping (patience=20).
Evaluates on test set (accuracy, AUC, classification report, confusion matrix).
Visualizes graph structure, feature importance, node embeddings (t-SNE), training history, confusion matrix, ROC curve, and precision-recall curve.


Outputs: parkinson_gcn_model.pt, plots (graph_structure.png, feature_importance.png, node_embeddings.png, training_history.png, confusion_matrix.png, roc_curve.png, precision_recall_curve.png).



README File: Implementation Steps
To reproduce the results:

Clone the Repository:git clone https://github.com/dineshreddy4096/pd-detection.git
cd pd-detection


Install Dependencies:Use Python 3.8+ and install required packages:pip install -r requirements.txt

Example requirements.txt:numpy==1.23.5
pandas==1.5.3
tensorflow==2.12.0
torch==2.0.1
torch-geometric==2.3.1
scikit-learn==1.2.2
pydicom==2.4.3
matplotlib==3.7.1
seaborn==0.12.2
scikit-image==0.21.0
networkx==3.1
opencv-python==4.8.0


Download Dataset:Access the dataset at https://drive.google.com/drive/folders/16fD8c2chMV63POxbTZGUuZQd5zbULvsq?usp=sharing. Download and organize DICOM files into data/pd_patients/PPMI and data/healthy/PPMI directories. Ensure folder structure matches /kaggle/input/parkinson22/ as used in scripts.
Update File Paths:Modify paths in scripts to point to local dataset directories:
Replace /kaggle/input/parkinson22/pd_patients/pd_patients/PPMI with ./data/pd_patients/PPMI.
Replace /kaggle/input/parkinson22/healthy/healthy/PPMI with ./data/healthy/PPMI.


Run the Pipeline:
VGG19 Model:python vgg19.py

Outputs: Models (parkinson_vgg19_final_model.h5, best_parkinson_model.keras), plots in current directory.
VGG19 with SSO:python vgg19Sso.py

Outputs: Model (pd_detection_model.h5), plots in plots/ directory.
GCN with SSO:python gcnsso.py

Outputs: Accuracy and AUC plots (displayed).
GCN with Image Features:python gcn.py

Outputs: Model (parkinson_gcn_model.pt), plots in visualizations/ directory.


Verify Outputs:
Models: Check for .h5, .keras, and .pt files.
Plots: Check directories (plots/, visualizations/) or current directory for PNG files.
Metrics: Review console output for accuracy, AUC, and classification reports.



Materials & Methods
Computing Infrastructure

Operating System: Tested on Ubuntu 20.04 (Kaggle environment) and compatible with Windows 10.
Hardware: Local machine with 16GB RAM, Intel i7 CPU, optional GPU (NVIDIA CUDA-compatible for TensorFlow/PyTorch).



3rd Party Dataset

Dataset: Parkinson's disease DICOM imaging data.
URL: https://drive.google.com/drive/folders/16fD8c2chMV63POxbTZGUuZQd5zbULvsq?usp=sharing.
Details: Contains PPMI dataset with DICOM files for PD patients and healthy controls, including clinical and imaging data. Access may require Google Drive permissions.

Evaluation Method
Models are evaluated on test sets using:

Primary Metrics: Accuracy and Area Under the ROC Curve (AUC).
Secondary Metrics: Precision, recall, F1-score (via classification report), confusion matrix.
Process:
VGG19 and VGG19-SSO: Evaluate on test images (15-20% of data) for binary classification (PD vs. healthy).
GCN and GCN-SSO: Evaluate on test nodes (20-30% of graph) for binary classification.
Metrics are computed using sklearn.metrics and visualized (confusion matrix, ROC curve, precision-recall curve).



Conclusions
Limitations

Dataset: The dataset may have selection bias, limiting generalizability to diverse PD populations.
Preprocessing: DICOM image preprocessing (normalization, resizing) may lose fine-grained details critical for diagnosis.
Models:
VGG19 models rely on pre-trained weights, which may not fully adapt to medical imaging.
GCN models depend on graph construction quality (e.g., similarity threshold), which can affect performance.


Compute: GPU is recommended for faster training; local CPU runs may be slow for large datasets.
Evaluation: Focus on binary classification may miss nuanced PD progression stages.

Additional Notes

Ensure GPU drivers and CUDA are installed for TensorFlow/PyTorch if using GPU.
Check visualizations/ and plots/ directories for output plots.
Report issues on the GitHub repository: https://github.com/dineshreddy4096/pd-detection.

License
MIT License. See LICENSE for details.
