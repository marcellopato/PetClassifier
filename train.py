import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import pathlib
import io
import numpy as np
from PIL import Image

# Constants
IMG_SIZE = 224  # Changed from 160 to 224 to match MobileNetV2's requirements
BATCH_SIZE = 32
EPOCHS = 10
AUTOTUNE = tf.data.AUTOTUNE

def decode_img(file_path):
    try:
        # Read the image file
        img_raw = tf.io.read_file(file_path)
        
        # Check if the file is empty
        if tf.equal(tf.size(img_raw), 0):
            tf.print("Empty file:", file_path)
            return None
            
        # Try different decoding methods
        img = None
        
        # Method 1: Try decode_jpeg
        try:
            img = tf.io.decode_jpeg(img_raw, channels=3)
            img = tf.cast(img, tf.float32)
        except Exception as e:
            tf.print("JPEG decode failed:", str(e), "Path:", file_path)
            
        # Method 2: Try decode_png if JPEG failed
        if img is None:
            try:
                img = tf.io.decode_png(img_raw, channels=3)
                img = tf.cast(img, tf.float32)
            except Exception as e:
                tf.print("PNG decode failed:", str(e), "Path:", file_path)
                
        # Method 3: Try PIL if both TensorFlow methods failed
        if img is None:
            try:
                # Convert the raw bytes to a PIL Image
                img_bytes = img_raw.numpy()
                img_pil = Image.open(io.BytesIO(img_bytes))
                
                # Convert to RGB if necessary
                if img_pil.mode != 'RGB':
                    img_pil = img_pil.convert('RGB')
                    
                # Resize image
                img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
                
                # Convert to numpy array
                img_array = np.array(img_pil, dtype=np.float32)
                
                # Convert to tensor
                img = tf.convert_to_tensor(img_array)
                
                tf.print("Successfully decoded with PIL:", file_path)
                
            except Exception as e:
                tf.print("PIL decode failed:", str(e), "Path:", file_path)
                return None
                
        # If all methods failed, return None
        if img is None:
            tf.print("All decode methods failed for:", file_path)
            return None
            
        # Get image shape
        shape = tf.shape(img)
        
        # Check image dimensions
        if shape[0] < 32 or shape[1] < 32:
            tf.print("Image too small:", shape, "Path:", file_path)
            return None
            
        # Resize if needed
        if shape[0] != IMG_SIZE or shape[1] != IMG_SIZE:
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            
        # Ensure we're working with float32
        img = tf.cast(img, tf.float32)
            
        # Check for NaN or Inf values
        if tf.reduce_any(tf.math.is_nan(img)) or tf.reduce_any(tf.math.is_inf(img)):
            tf.print("Invalid values in image:", file_path)
            return None
            
        # Check pixel range
        min_val = tf.reduce_min(img)
        max_val = tf.reduce_max(img)
        if min_val < 0 or max_val > 255:
            tf.print("Invalid pixel range:", min_val, max_val, "Path:", file_path)
            return None
            
        # Normalize to [0,1]
        img = img / 255.0
        
        # Return the processed image
        return img
        
    except Exception as e:
        tf.print("Unexpected error in decode_img:", str(e), "Path:", file_path)
        return None

def process_path(file_path):
    # Get the label
    parts = tf.strings.split(file_path, os.path.sep)
    label = parts[-2]
    label = tf.cast(label == "dogs", tf.int32)
    
    # Get and process the image
    img = decode_img(file_path)
    
    if img is None:
        # Return a default image if decoding fails
        img = tf.zeros([IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32)
        
    return img, label

def prepare_dataset(data_dir, is_training=True):
    # Get dataset size first
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(f"\nFound {image_count} images in {data_dir}")
    
    # Create the dataset
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=is_training)
    
    # Set the dataset size
    dataset_size = image_count
    
    # Process the images in parallel with error handling
    labeled_ds = list_ds.map(
        process_path,
        num_parallel_calls=AUTOTUNE
    )
    
    # Filter out error cases and extract only image and label
    valid_count = tf.Variable(0, dtype=tf.int32)
    invalid_count = tf.Variable(0, dtype=tf.int32)
    
    def count_valid(x, y):
        valid_count.assign_add(1)
        return True
        
    labeled_ds = labeled_ds.map(
        lambda x, y: (x, y),
        num_parallel_calls=AUTOTUNE
    )
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Total images: {dataset_size}")
    print(f"Valid images: {valid_count.numpy()}")
    print(f"Invalid images: {invalid_count.numpy()}")
    
    if is_training:
        # Data augmentation only during training
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
        ])
        
        labeled_ds = labeled_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )
    
    # Set batching and prefetching
    labeled_ds = labeled_ds.cache()
    if is_training:
        labeled_ds = labeled_ds.shuffle(buffer_size=1000)
    labeled_ds = labeled_ds.batch(BATCH_SIZE)
    labeled_ds = labeled_ds.prefetch(buffer_size=AUTOTUNE)
    
    return labeled_ds, dataset_size

def create_model():
    # Create data augmentation layer
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])
    
    # Create the base model from the pre-trained model MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    # Create new model on top
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

def plot_training_results(history):
    # Print available metrics for debugging
    print("\nPlotting training results...")
    print("Available metrics:", history.history.keys())
    
    # Define metrics to plot with their display names
    metrics_info = [
        ('accuracy', 'Accuracy'),
        ('loss', 'Loss'),
        ('auc', 'AUC'),
        ('precision', 'Precision'),
        ('recall', 'Recall')
    ]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    for i, (metric, title) in enumerate(metrics_info, 1):
        ax = fig.add_subplot(2, 3, i)
        
        # Plot training
        if metric in history.history:
            train_values = history.history[metric]
            ax.plot(train_values, label=f'Training {title}')
            print(f"\nTraining {title}:")
            print(f"  Start: {train_values[0]:.4f}")
            print(f"  End: {train_values[-1]:.4f}")
            print(f"  Best: {max(train_values):.4f}")
        else:
            print(f"Warning: {metric} not found in history")
        
        # Plot validation
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            val_values = history.history[val_metric]
            ax.plot(val_values, label=f'Validation {title}')
            print(f"\nValidation {title}:")
            print(f"  Start: {val_values[0]:.4f}")
            print(f"  End: {val_values[-1]:.4f}")
            print(f"  Best: {max(val_values):.4f}")
        else:
            print(f"Warning: {val_metric} not found in history")
        
        ax.set_title(f'Training and Validation {title}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    try:
        plt.savefig('training_results.png')
        print("\nTraining plots saved successfully to 'training_results.png'")
    except Exception as e:
        print(f"\nError saving training plots: {str(e)}")
    
    plt.close()

def main():
    print("Setting up training...")
    
    # Create necessary directories
    os.makedirs('data/train/cats', exist_ok=True)
    os.makedirs('data/train/dogs', exist_ok=True)
    os.makedirs('data/test/cats', exist_ok=True)
    os.makedirs('data/test/dogs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Prepare datasets
    print("Loading training dataset...")
    train_ds, train_size = prepare_dataset('data/train', is_training=True)
    print("Loading validation dataset...")
    val_ds, val_size = prepare_dataset('data/test', is_training=False)
    
    # Calculate steps per epoch
    steps_per_epoch = train_size // BATCH_SIZE
    validation_steps = val_size // BATCH_SIZE
    
    print(f"\nDataset Statistics:")
    print(f"Training images: {train_size}")
    print(f"Validation images: {val_size}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # Create and compile model
    print("\nCreating model...")
    model = create_model()

    # Add callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/pet_classifier.keras',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=0.00001
        )
    ]

    # Train the model
    print("\nStarting training...")
    try:
        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1  # Ensure we see detailed progress
        )
        
        # Verify that we have training history
        if not history.history:
            print("Warning: Training history is empty!")
            return
            
        # Print training summary
        print("\nTraining Summary:")
        for metric in history.history.keys():
            values = history.history[metric]
            print(f"{metric}:")
            print(f"  Start: {values[0]:.4f}")
            print(f"  End: {values[-1]:.4f}")
            print(f"  Best: {max(values):.4f}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return

    # Save the final model
    try:
        print("\nSaving model...")
        model.save('models/pet_classifier.keras')
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

    # Plot results
    print("Generating training plots...")
    plot_training_results(history)
    
    # Print final metrics
    print("\nTraining completed!")
    if 'val_accuracy' in history.history:
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.2%}")
    if 'val_auc' in history.history:
        print(f"Final validation AUC: {history.history['val_auc'][-1]:.4f}")
    if 'val_precision' in history.history:
        print(f"Final validation precision: {history.history['val_precision'][-1]:.4f}")
    if 'val_recall' in history.history:
        print(f"Final validation recall: {history.history['val_recall'][-1]:.4f}")

if __name__ == '__main__':
    main()
