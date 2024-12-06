import os
import shutil
import random
from PIL import Image
import sys
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_organization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def is_valid_image(file_path):
    try:
        # Skip files that are not images
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            logging.warning(f"Skipping {file_path}: Not a supported image format")
            return False
            
        # Skip files that are too small
        if os.path.getsize(file_path) < 1024:  # Skip files smaller than 1KB
            logging.warning(f"Skipping {file_path}: File too small (<1KB)")
            return False
            
        with Image.open(file_path) as img:
            # Verify the image
            img.verify()
            
            # Try to load the image data
            with Image.open(file_path) as img2:
                img2.load()
                
                # Check if image has valid size
                if img2.size[0] < 10 or img2.size[1] < 10:
                    logging.warning(f"Skipping {file_path}: Image dimensions too small")
                    return False
                
                # Convert to RGB if needed
                if img2.mode != 'RGB':
                    img2 = img2.convert('RGB')
                
                # Check for invalid pixel values
                img_array = np.array(img2)
                if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
                    logging.warning(f"Skipping {file_path}: Contains NaN or infinite pixel values")
                    return False
                
                # Check if image is mostly empty or constant
                if np.std(img_array) < 1.0:
                    logging.warning(f"Skipping {file_path}: Image appears to be empty or constant")
                    return False
                
                return True
                
    except (IOError, SyntaxError) as e:
        logging.error(f"Error reading image {file_path}: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error with image {file_path}: {str(e)}")
        return False

def organize_dataset(source_dir, train_dir, test_dir, train_split=0.8):
    try:
        # Create directories if they don't exist
        for dir_path in [
            os.path.join(train_dir, 'cats'),
            os.path.join(train_dir, 'dogs'),
            os.path.join(test_dir, 'cats'),
            os.path.join(test_dir, 'dogs')
        ]:
            os.makedirs(dir_path, exist_ok=True)

        dataset_stats = {'total': 0, 'valid': 0, 'corrupted': 0}
        
        # Process cats and dogs
        for animal in ['Cat', 'Dog']:
            logging.info(f"\nProcessing {animal} images...")
            source_path = os.path.join(source_dir, 'PetImages', animal)
            
            if not os.path.exists(source_path):
                logging.error(f"Source directory not found: {source_path}")
                continue
            
            # Get all valid images
            valid_images = []
            animal_stats = {'total': 0, 'valid': 0, 'corrupted': 0}
            
            for img_name in os.listdir(source_path):
                animal_stats['total'] += 1
                img_path = os.path.join(source_path, img_name)
                
                if is_valid_image(img_path):
                    valid_images.append(img_name)
                    animal_stats['valid'] += 1
                else:
                    animal_stats['corrupted'] += 1
                
                # Log progress every 100 images
                if animal_stats['total'] % 100 == 0:
                    logging.info(f"Processed {animal_stats['total']} {animal} images...")
            
            # Update overall stats
            dataset_stats['total'] += animal_stats['total']
            dataset_stats['valid'] += animal_stats['valid']
            dataset_stats['corrupted'] += animal_stats['corrupted']
            
            logging.info(f"{animal} dataset statistics:")
            logging.info(f"- Total images: {animal_stats['total']}")
            logging.info(f"- Valid images: {animal_stats['valid']}")
            logging.info(f"- Corrupted images: {animal_stats['corrupted']}")
            
            # Shuffle and split images
            random.shuffle(valid_images)
            split_idx = int(len(valid_images) * train_split)
            train_images = valid_images[:split_idx]
            test_images = valid_images[split_idx:]
            
            # Copy images to respective directories
            target_name = animal.lower() + 's'
            
            logging.info(f"Copying {len(train_images)} images to training set...")
            for img_name in train_images:
                try:
                    src = os.path.join(source_path, img_name)
                    dst = os.path.join(train_dir, target_name, img_name)
                    shutil.copy2(src, dst)
                except Exception as e:
                    logging.error(f"Error copying {img_name} to training set: {str(e)}")
            
            logging.info(f"Copying {len(test_images)} images to test set...")
            for img_name in test_images:
                try:
                    src = os.path.join(source_path, img_name)
                    dst = os.path.join(test_dir, target_name, img_name)
                    shutil.copy2(src, dst)
                except Exception as e:
                    logging.error(f"Error copying {img_name} to test set: {str(e)}")
        
        # Log final statistics
        logging.info("\nFinal dataset statistics:")
        logging.info(f"Total images processed: {dataset_stats['total']}")
        logging.info(f"Valid images: {dataset_stats['valid']}")
        logging.info(f"Corrupted images: {dataset_stats['corrupted']}")
        logging.info(f"Valid image ratio: {dataset_stats['valid']/dataset_stats['total']*100:.2f}%")
        
    except Exception as e:
        logging.error(f"An error occurred during dataset organization: {str(e)}")
        raise

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, 'temp_download')
    train_dir = os.path.join(base_dir, 'data', 'train')
    test_dir = os.path.join(base_dir, 'data', 'test')
    
    print("Starting dataset organization...")
    organize_dataset(source_dir, train_dir, test_dir)
    print("\nDataset organization completed!")
