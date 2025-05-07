Sso code:
import os
import numpy as np
import pydicom
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Set up paths
print("Setting up paths...")
pd_path = '/kaggle/input/parkinson22/pd_patients/pd_patients/PPMI'
healthy_path = '/kaggle/input/parkinson22/healthy/healthy/PPMI'

# Load DICOM images in batches
def load_dicom_images(folder_path, label):
    images, labels = [], []
    for subject in os.listdir(folder_path):
        subject_path = os.path.join(folder_path, subject)
        if not os.path.isdir(subject_path):
            continue
        for file in os.listdir(subject_path):
            file_path = os.path.join(subject_path, file)
            try:
                dicom = pydicom.dcmread(file_path)
                image_array = dicom.pixel_array.astype(float)
                if np.min(image_array) == np.max(image_array):
                    continue
                image_array = 255 * (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
                image_array = tf.image.resize(image_array[..., np.newaxis], [128, 128]).numpy()
                image_array = np.repeat(image_array, 3, axis=-1)
                image_array = preprocess_input(image_array)
                if image_array.shape == (128, 128, 3):
                    images.append(image_array)
                    labels.append(label)
                if len(images) >= 1000:
                    yield np.stack(images), np.array(labels)
                    images, labels = [], []
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    if images:
        yield np.stack(images), np.array(labels)

# Load images
print("Loading images in batches...")
pd_images, pd_labels = next(load_dicom_images(pd_path, label=1))
healthy_images, healthy_labels = next(load_dicom_images(healthy_path, label=0))

X = np.concatenate((pd_images, healthy_images), axis=0)
y = np.concatenate((pd_labels, healthy_labels), axis=0)

# Train-test-validation split with proper stratification
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Create a small evaluation subset for quick SSO fitness evaluation
eval_size = min(100, len(X_val))
X_eval, y_eval = X_val[:eval_size], y_val[:eval_size]
y_eval_cat = to_categorical(y_eval, num_classes=2)

# Define model
def create_model(learning_rate, dense_units, dropout_rate):
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(int(dense_units), activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# Salp Swarm Optimization (SSO) Implementation
def sso_optimization(n_salps=5, max_iter=3):
    # Define search space bounds
    bounds = {
        'learning_rate': [1e-5, 1e-3],
        'dense_units': [128, 512],
        'dropout_rate': [0.3, 0.6]
    }

    # Initialize salps' positions
    salps = np.zeros((n_salps, 3))  # 3 parameters: lr, units, dropout
    for i in range(n_salps):
        salps[i, 0] = np.random.uniform(bounds['learning_rate'][0], bounds['learning_rate'][1])
        salps[i, 1] = np.random.randint(bounds['dense_units'][0], bounds['dense_units'][1])
        salps[i, 2] = np.random.uniform(bounds['dropout_rate'][0], bounds['dropout_rate'][1])

    # Initialize best solution (food source)
    best_pos = np.zeros(3)
    best_score = -np.inf

    # SSO main loop
    print("Running Salp Swarm Optimizer...")
    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}/{max_iter}")
        for i in range(n_salps):
            # Evaluate fitness
            learning_rate, dense_units, dropout_rate = salps[i]
            model = create_model(learning_rate, int(dense_units), dropout_rate)
            
            # Quick evaluation instead of full training for fitness calculation
            val_loss, val_acc, val_auc = model.evaluate(X_eval, y_eval_cat, verbose=0)
            fitness_score = val_acc * 0.7 + val_auc * 0.3  # Combined metric
            
            print(f"  Salp {i+1}: LR={learning_rate:.6f}, Units={int(dense_units)}, Dropout={dropout_rate:.2f}, Score={fitness_score:.4f}")
            
            # Update best solution
            if fitness_score > best_score:
                best_score = fitness_score
                best_pos = salps[i].copy()
            
            # Clean up to avoid memory issues
            del model
            gc.collect()
            tf.keras.backend.clear_session()

        # Update salp positions
        c1 = 2 * np.exp(-((4 * iteration / max_iter) ** 2))  # Exploration factor
        for i in range(n_salps):
            if i == 0:  # Leader salp
                for j in range(3):
                    r1, r2 = np.random.rand(2)
                    salps[i, j] = best_pos[j] + c1 * ((bounds[list(bounds.keys())[j]][1] - bounds[list(bounds.keys())[j]][0]) * r1 - r2)
            else:  # Follower salps
                for j in range(3):
                    salps[i, j] = (salps[i, j] + salps[i-1, j]) / 2

            # Clamp to bounds
            salps[i, 0] = np.clip(salps[i, 0], bounds['learning_rate'][0], bounds['learning_rate'][1])
            salps[i, 1] = np.clip(salps[i, 1], bounds['dense_units'][0], bounds['dense_units'][1])
            salps[i, 2] = np.clip(salps[i, 2], bounds['dropout_rate'][0], bounds['dropout_rate'][1])

    return best_pos, best_score

# Run SSO
best_params, best_score = sso_optimization(n_salps=5, max_iter=3)
best_learning_rate, best_dense_units, best_dropout_rate = best_params
best_dense_units = int(best_dense_units)  # Ensure dense_units is an integer
print(f"Best parameters: LR={best_learning_rate:.6f}, Units={best_dense_units}, Dropout={best_dropout_rate:.2f}, Score={best_score:.4f}")

# Train final model with best parameters
print("Training final model with best hyperparameters...")
tf.keras.backend.clear_session()  # Clear session before final training
best_model = create_model(best_learning_rate, best_dense_units, best_dropout_rate)

# Use a longer patience for the final model training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Prepare categorical labels for training
y_train_cat = to_categorical(y_train, num_classes=2)
y_val_cat = to_categorical(y_val, num_classes=2)

# Add data augmentation for training
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])

# Custom training with data augmentation
batch_size = 32
epochs = 15
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
train_dataset = train_dataset.batch(batch_size)

# Training function with data augmentation
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # Data augmentation is applied only during training
        augmented_images = data_augmentation(images, training=True)
        predictions = best_model(augmented_images, training=True)
        loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
    gradients = tape.gradient(loss, best_model.trainable_variables)
    best_model.optimizer.apply_gradients(zip(gradients, best_model.trainable_variables))
    return loss

# Training loop with early stopping
best_val_loss = float('inf')
patience_counter = 0
history = {'loss': [], 'val_loss': [], 'val_accuracy': [], 'val_auc': []}

for epoch in range(epochs):
    # Training
    epoch_loss_avg = tf.keras.metrics.Mean()
    for images, labels in train_dataset:
        loss = train_step(images, labels)
        epoch_loss_avg.update_state(loss)
    
    # Validation
    val_loss, val_accuracy, val_auc = best_model.evaluate(X_val, y_val_cat, verbose=0)
    
    # Update history
    history['loss'].append(epoch_loss_avg.result().numpy())
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_accuracy)
    history['val_auc'].append(val_auc)
    
    print(f"Epoch {epoch+1}/{epochs}: loss={epoch_loss_avg.result().numpy():.4f}, val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}, val_auc={val_auc:.4f}")
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best weights
        best_weights = best_model.get_weights()
    else:
        patience_counter += 1
        if patience_counter >= early_stopping.patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

# Restore best weights
best_model.set_weights(best_weights)

# Convert history to proper format for visualization
history_obj = type('obj', (object,), {'history': history})

# Evaluate on test set
y_test_cat = to_categorical(y_test, num_classes=2)
test_loss, test_accuracy, test_auc = best_model.evaluate(X_test, y_test_cat, verbose=1)
print(f"Test accuracy: {test_accuracy:.4f}, Test AUC: {test_auc:.4f}")

y_pred = best_model.predict(X_test)
y_pred_binary = np.argmax(y_pred, axis=1)
y_test_binary = y_test

# Create directory for plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

def plot_visualizations(history, y_test_binary, y_pred, y_pred_binary):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Over Epochs')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Over Epochs')
    plt.tight_layout()
    plt.savefig('plots/training_curves.png')
    plt.close()

    cm = confusion_matrix(y_test_binary, y_pred_binary)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'PD'], yticklabels=['Healthy', 'PD'])
    plt.title('Confusion Matrix (Test Data)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()

    fpr, tpr, _ = roc_curve(y_test_binary, y_pred[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test Data)')
    plt.legend(loc="lower right")
    plt.savefig('plots/roc_curve.png')
    plt.close()

    report = classification_report(y_test_binary, y_pred_binary, target_names=['Healthy', 'PD'], output_dict=True)
    metrics = ['precision', 'recall', 'f1-score']
    healthy_scores = [report['Healthy'][m] for m in metrics]
    pd_scores = [report['PD'][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, healthy_scores, width, label='Healthy', color='skyblue')
    plt.bar(x + width/2, pd_scores, width, label='PD', color='salmon')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1-Score (Test Data)')
    plt.xticks(x, metrics)
    plt.legend()
    plt.savefig('plots/classification_metrics.png')
    plt.close()

    print("\nClassification Report (Test Data):")
    print(classification_report(y_test_binary, y_pred_binary, target_names=['Healthy', 'PD']))

# Generate visualizations
plot_visualizations(history_obj, y_test_binary, y_pred, y_pred_binary)

# Save the model
best_model.save('pd_detection_model.h5')
print("Model saved as pd_detection_model.h5")
