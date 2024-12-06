import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import pathlib

# Constants
IMG_SIZE = 224  # Changed from 160 to 224 to match MobileNetV2's requirements
BATCH_SIZE = 32
EPOCHS = 10
AUTOTUNE = tf.data.AUTOTUNE

def decode_img(img):
    try:
        # Convert the compressed string to a tensor
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        
        # Ensure we have a valid image
        if tf.rank(img) != 3 or tf.shape(img)[2] not in [1, 2, 3, 4]:
            return None
            
        # Convert grayscale to RGB if needed
        if tf.shape(img)[2] == 1:
            img = tf.image.grayscale_to_rgb(img)
        elif tf.shape(img)[2] == 2:
            # For 2-channel images, duplicate one channel to make it RGB
            img = tf.concat([img[..., :1], img[..., :1], img[..., :1]], axis=-1)
        elif tf.shape(img)[2] == 4:
            # For RGBA images, just take the RGB channels
            img = img[..., :3]
            
        # Resize the image to the desired size
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        return img
    except:
        return None

def process_path(file_path):
    try:
        # Convert file_path tensor to string for logging
        file_path_str = tf.strings.as_string(file_path)
        
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        
        # Skip if we got a None image (error case)
        if img is None:
            tf.print("Error processing image:", file_path_str)
            return (
                tf.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32)
            )
            
        # Normalize pixel values and ensure shape
        img = img / 255.0
        img = tf.ensure_shape(img, (IMG_SIZE, IMG_SIZE, 3))
        
        # Determine class based on path
        label = tf.strings.split(file_path, os.path.sep)[-2]
        label = tf.cast(tf.equal(label, "dogs"), tf.float32)
        
        # Verify the image has valid values
        if tf.reduce_any(tf.math.is_nan(img)) or tf.reduce_any(tf.math.is_inf(img)):
            tf.print("Invalid pixel values in image:", file_path_str)
            return (
                tf.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32)
            )
            
        return img, label
    except Exception as e:
        tf.print("Error processing image:", file_path_str, "Error:", str(e))
        return (
            tf.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.constant(0.0, dtype=tf.float32)
        )

def prepare_dataset(data_dir, is_training=True):
    data_dir = pathlib.Path(data_dir)
    
    # Get list of files
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=is_training)
    dataset_size = len(list(data_dir.glob('*/*')))
    
    # Process files and filter out errors
    labeled_ds = list_ds.map(
        process_path,
        num_parallel_calls=AUTOTUNE
    )
    
    # Filter out error cases (all zero images)
    labeled_ds = labeled_ds.filter(
        lambda x, y: tf.math.reduce_any(tf.math.not_equal(x, 0.0))
    )
    
    if is_training:
        # Add data augmentation for training
        labeled_ds = labeled_ds.map(
            lambda x, y: (tf.image.random_flip_left_right(x), y),
            num_parallel_calls=AUTOTUNE
        )
        labeled_ds = labeled_ds.map(
            lambda x, y: (tf.image.random_brightness(x, 0.2), y),
            num_parallel_calls=AUTOTUNE
        )
        # For training, repeat indefinitely
        labeled_ds = labeled_ds.repeat()
    else:
        # For validation, repeat only once
        labeled_ds = labeled_ds.repeat(1)
    
    # Cache, shuffle, batch, and prefetch
    labeled_ds = labeled_ds.cache()
    if is_training:
        labeled_ds = labeled_ds.shuffle(1000)
    labeled_ds = labeled_ds.batch(BATCH_SIZE)
    labeled_ds = labeled_ds.prefetch(buffer_size=AUTOTUNE)
    
    return labeled_ds, dataset_size

def create_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)  # No activation for logits
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=0.0),  # For logits
            tf.keras.metrics.AUC(name='auc', from_logits=True),
            tf.keras.metrics.Precision(name='precision', thresholds=0.0),  # For logits
            tf.keras.metrics.Recall(name='recall', thresholds=0.0)  # For logits
        ]
    )
    
    return model

def plot_training_results(history):
    # Print available metrics for debugging
    print("\nAvailable metrics:", history.history.keys())
    
    # Define metrics to plot with their display names
    metrics_info = [
        ('binary_accuracy', 'Accuracy'),
        ('loss', 'Loss'),
        ('auc', 'AUC'),
        ('precision', 'Precision'),
        ('recall', 'Recall')
    ]
    
    plt.figure(figsize=(15, 10))
    
    for i, (metric, title) in enumerate(metrics_info):
        plt.subplot(2, 3, i+1)
        
        # Plot training
        if metric in history.history:
            plt.plot(history.history[metric], label=f'Training {title}')
        else:
            print(f"Warning: {metric} not found in history")
        
        # Plot validation
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            plt.plot(history.history[val_metric], label=f'Validation {title}')
        else:
            print(f"Warning: {val_metric} not found in history")
        
        plt.title(f'Training and Validation {title}')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()
    
    plt.tight_layout()
    try:
        plt.savefig('training_results.png')
        print("Training plots saved successfully")
    except Exception as e:
        print(f"Error saving training plots: {str(e)}")
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
            monitor='val_binary_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_binary_accuracy',
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
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    # Save the final model
    print("\nSaving model...")
    model.save('models/pet_classifier.keras')

    # Plot results
    print("Generating training plots...")
    plot_training_results(history)
    
    # Print final metrics
    print("\nTraining completed!")
    if 'val_binary_accuracy' in history.history:
        print(f"Final validation accuracy: {history.history['val_binary_accuracy'][-1]:.2%}")
    if 'val_auc' in history.history:
        print(f"Final validation AUC: {history.history['val_auc'][-1]:.4f}")
    if 'val_precision' in history.history:
        print(f"Final validation precision: {history.history['val_precision'][-1]:.4f}")
    if 'val_recall' in history.history:
        print(f"Final validation recall: {history.history['val_recall'][-1]:.4f}")

if __name__ == '__main__':
    main()
