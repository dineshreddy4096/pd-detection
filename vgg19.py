    # Step 1: Import necessary libraries
    import os
    import numpy as np
    import pydicom
    import tensorflow as tf
    from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    import seaborn as sns
    import gc
    import pandas as pd
    from tqdm import tqdm
    
    # Step 2: Set up paths and define helper functions
    print("Setting up paths and helper functions...")
    pd_path = '/kaggle/input/parkinson22/pd_patients/pd_patients/PPMI'
    healthy_path = '/kaggle/input/parkinson22/healthy/healthy/PPMI'
    
    # Function to load and preprocess DICOM image
    def load_and_preprocess_dicom(dicom_path):
        try:
            dicom = pydicom.dcmread(dicom_path)
            image_array = dicom.pixel_array.astype(float)
    
            # Skip images with no variation
            if np.min(image_array) == np.max(image_array):
                return None
    
            # Normalize to 0-255
            image_array = 255 * (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    
            # Handle different channel configurations
            if len(image_array.shape) > 2:
                image_array = image_array[..., 0]  # Take first channel if multi-channel
            
            # Resize and add channel dimension
            image_array = tf.image.resize(image_array[..., np.newaxis], [128, 128]).numpy()
    
            # Convert to 3-channel for VGG (grayscale to RGB)
            image_array = np.repeat(image_array, 3, axis=-1)
            
            # Preprocess for VGG19
            image_array = preprocess_input(image_array)
    
            return image_array
        except Exception as e:
            print(f"Error processing {dicom_path}: {str(e)}")
            return None
    
    # Collect all files with their labels
    print("Collecting dataset files...")
    def collect_files(pd_path, healthy_path):
        pd_files = []
        healthy_files = []
        
        # Collect PD files
        for subj in tqdm(os.listdir(pd_path), desc="Collecting PD files"):
            subj_dir = os.path.join(pd_path, subj)
            if os.path.isdir(subj_dir):
                for f in os.listdir(subj_dir):
                    if f.lower().endswith('.dcm'):
                        pd_files.append((os.path.join(subj_dir, f), 1))
        
        # Collect healthy files
        for subj in tqdm(os.listdir(healthy_path), desc="Collecting healthy files"):
            subj_dir = os.path.join(healthy_path, subj)
            if os.path.isdir(subj_dir):
                for f in os.listdir(subj_dir):
                    if f.lower().endswith('.dcm'):
                        healthy_files.append((os.path.join(subj_dir, f), 0))
        
        print(f"Found {len(pd_files)} Parkinson's files and {len(healthy_files)} healthy control files")
        return pd_files, healthy_files
    
    pd_files, healthy_files = collect_files(pd_path, healthy_path)
    all_files = pd_files + healthy_files
    np.random.shuffle(all_files)
    
    # Split into train, validation, and test sets
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)
    
    print(f"Training set: {len(train_files)} files")
    print(f"Validation set: {len(val_files)} files")
    print(f"Test set: {len(test_files)} files")
    
    # Generator function for dataset batches
    def dataset_generator(file_list, batch_size=32, augment=False):
        datagen = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest'
        ) if augment else None
        
        while True:
            np.random.shuffle(file_list)
            for start in range(0, len(file_list), batch_size):
                X_batch = []
                y_batch = []
                end = min(start + batch_size, len(file_list))
    
                for file_path, label in file_list[start:end]:
                    image = load_and_preprocess_dicom(file_path)
                    if image is not None:
                        X_batch.append(image)
                        y_batch.append(label)
    
                if X_batch:
                    X_batch_array = np.array(X_batch)
                    y_batch_array = to_categorical(y_batch, num_classes=2)
                    
                    if augment and datagen:
                        # Only yield the first batch from the generator to match expected behavior
                        for augmented_batch in datagen.flow(X_batch_array, y_batch_array, batch_size=len(X_batch_array)):
                            yield augmented_batch
                            break
                    else:
                        yield X_batch_array, y_batch_array
    
    # Step 3: Build and compile the model
    print("Building the VGG19-based model...")
    def build_model(input_shape=(128, 128, 3)):
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
    
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        output = Dense(2, activation='softmax')(x)
    
        model = Model(inputs=base_model.input, outputs=output)
    
        model.compile(
            optimizer=Adam(learning_rate=0.0001), 
            loss='categorical_crossentropy', 
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    model = build_model()
    model.summary()
    
    # Step 4: Train the model
    print("Starting training...")
    batch_size = 32
    train_gen = dataset_generator(train_files, batch_size=batch_size, augment=True)
    val_gen = dataset_generator(val_files, batch_size=batch_size, augment=False)
    
    # Calculate steps
    steps_per_epoch = len(train_files) // batch_size
    validation_steps = max(1, len(val_files) // batch_size)
    
    # Callbacks for training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint('best_parkinson_model.keras', monitor='val_auc', mode='max', save_best_only=True, verbose=1)
    ]
    
    
    # Train the model
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=25,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Clean up memory
    gc.collect()
    
    # Step 5: Evaluate the model on test set
    print("Evaluating the model on test set...")
    
    # Function to evaluate on full test set
    def evaluate_on_test_set(model, test_files, batch_size=32):
        all_y_true = []
        all_y_pred = []
        all_X_test = []
        
        # Process test files in batches
        for i in range(0, len(test_files), batch_size):
            batch_files = test_files[i:i+batch_size]
            X_batch = []
            y_batch = []
            
            for file_path, label in batch_files:
                image = load_and_preprocess_dicom(file_path)
                if image is not None:
                    X_batch.append(image)
                    y_batch.append(label)
            
            if X_batch:
                X_batch = np.array(X_batch)
                y_batch = np.array(y_batch)
                
                y_pred = model.predict(X_batch)
                
                all_X_test.extend(X_batch)
                all_y_true.extend(y_batch)
                all_y_pred.extend(y_pred)
        
        all_X_test = np.array(all_X_test)
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        
        return all_X_test, all_y_true, all_y_pred
    
    # Evaluate on full test set
    X_test, y_true, y_pred_prob = evaluate_on_test_set(model, test_files)
    y_pred_class = np.argmax(y_pred_prob, axis=1)
    y_true_categorical = to_categorical(y_true, num_classes=2)
    
    # Calculate metrics
    test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_true_categorical, verbose=1)
    print(f"\nTest metrics:")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_class, target_names=['Healthy', 'Parkinson\'s']))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred_class)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Step 6: Visualize results
    print("Generating visualizations...")
    
    # Plot training history
    def plot_training_history(history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        
        # Plot AUC
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['auc'], label='Train AUC')
        plt.plot(history.history['val_auc'], label='Validation AUC')
        plt.title('Model AUC over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('auc_history.png')
        plt.close()
        
        # Plot difference between train and validation
        plt.figure(figsize=(20, 6))
        
        plt.subplot(1, 3, 1)
        acc_diff = [t - v for t, v in zip(history.history['accuracy'], history.history['val_accuracy'])]
        plt.plot(acc_diff)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Train-Val Accuracy Difference')
        plt.xlabel('Epoch')
        plt.ylabel('Difference')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.subplot(1, 3, 2)
        loss_diff = [t - v for t, v in zip(history.history['loss'], history.history['val_loss'])]
        plt.plot(loss_diff)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Train-Val Loss Difference')
        plt.xlabel('Epoch')
        plt.ylabel('Difference')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.subplot(1, 3, 3)
        auc_diff = [t - v for t, v in zip(history.history['auc'], history.history['val_auc'])]
        plt.plot(auc_diff)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Train-Val AUC Difference')
        plt.xlabel('Epoch')
        plt.ylabel('Difference')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig('train_val_differences.png')
        plt.close()
    
    # Plot confusion matrix
    def plot_confusion_matrix(cm):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Healthy', 'Parkinson\'s'],
                    yticklabels=['Healthy', 'Parkinson\'s'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
    
    # Plot ROC curve
    def plot_roc_curve(y_true, y_pred_prob):
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
        auc = roc_auc_score(y_true, y_pred_prob[:, 1])
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('roc_curve.png')
        plt.close()
    
    # Generate all the plots
    plot_training_history(history)
    plot_confusion_matrix(cm)
    plot_roc_curve(y_true, y_pred_prob)
    
    # Step 7: Save the model
    print("Saving the model...")
    model.save('parkinson_vgg19_final_model.h5')
    print("Model saved as 'parkinson_vgg19_final_model.h5'")
    
    # Display a summary of what was generated
    print("\nSummary of outputs:")
    print("1. training_history.png - Accuracy and loss curves")
    print("2. auc_history.png - AUC performance over time")
    print("3. train_val_differences.png - Differences between training and validation metrics")
    print("4. confusion_matrix.png - Confusion matrix visualization")
    print("5. roc_curve.png - ROC curve with AUC score")
    print("6. parkinson_vgg19_final_model.h5 - Final trained model")
    print("7. best_parkinson_model.h5 - Best model based on validation AUC")
    
    print("\nTraining and evaluation completed!")
