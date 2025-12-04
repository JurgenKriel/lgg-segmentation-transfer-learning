import os
import glob
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

def parse_region_from_filename(filename):
    """Extract region type from filename"""
    basename = os.path.basename(filename).replace('_preprocessed.png', '')
    match = re.search(r'_(CT|LE|IT)\d*$', basename)
    
    if match:
        region = match.group(1)
        if region == 'CT':
            return 1
        elif region == 'LE':
            return 2
        elif region == 'IT':
            return 3
    
    return None

def detect_tissue_mask(image_array):
    """
    Detect tissue vs background using color thresholding
    Background is typically white/light gray
    Tissue has color (H&E staining)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    
    # Tissue has saturation > threshold (not white/gray)
    # and brightness < threshold (not too bright)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    
    # Tissue criteria:
    # - Saturation > 15 (has some color)
    # - Value < 240 (not pure white)
    tissue_mask = (saturation > 15) & (value < 240)
    
    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    tissue_mask = cv2.morphologyEx(tissue_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
    
    return tissue_mask.astype(bool)

def generate_mask_for_crop(image_path, output_dir):
    """Generate mask with tissue detection"""
    
    class_label = parse_region_from_filename(image_path)
    
    if class_label is None:
        return False
    
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    # Detect tissue
    tissue_mask = detect_tissue_mask(img_array)
    
    # Create mask: 0 for background, class_label for tissue
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[tissue_mask] = class_label
    
    # Save mask
    mask_filename = os.path.basename(image_path).replace('_preprocessed.png', '_mask.png')
    mask_path = os.path.join(output_dir, mask_filename)
    
    Image.fromarray(mask).save(mask_path)
    
    return True

def main():
    images_dir = "/vast/projects/Histology_Glioma_ML/Annotated_HE_ML_Validation/all_images/preprocessed"
    
    print("="*80)
    print("GENERATING TISSUE-AWARE MASKS FOR CROPPED REGION IMAGES")
    print("="*80)
    
    # Find images without masks
    all_images = sorted(glob.glob(os.path.join(images_dir, "*_preprocessed.png")))
    mask_paths = [p.replace('_preprocessed.png', '_mask.png') for p in all_images]
    
    images_without_masks = []
    for img_path, mask_path in zip(all_images, mask_paths):
        if not os.path.exists(mask_path):
            images_without_masks.append(img_path)
    
    print(f"\nFound {len(images_without_masks)} images without masks")
    
    # Identify region crops
    region_crops = []
    for img_path in images_without_masks:
        class_label = parse_region_from_filename(img_path)
        if class_label is not None:
            region_crops.append((img_path, class_label))
    
    print(f"Identified {len(region_crops)} as labeled region crops")
    
    # Count by class
    class_names = {1: 'CT', 2: 'LE', 3: 'IT'}
    class_counts = {1: 0, 2: 0, 3: 0}
    for _, label in region_crops:
        class_counts[label] += 1
    
    print("\nBreakdown by class:")
    for label, name in class_names.items():
        count = class_counts[label]
        print(f"  {name}: {count} images")
    
    # Generate masks with tissue detection
    print(f"\nGenerating tissue-aware masks...")
    generated = 0
    
    for img_path, class_label in tqdm(region_crops):
        if generate_mask_for_crop(img_path, images_dir):
            generated += 1
    
    print(f"\n✓ Generated {generated} tissue-aware masks")
    print(f"✓ Background pixels set to class 0")
    print(f"✓ Tissue pixels set to appropriate region class")
    print(f"✓ Saved to: {images_dir}")
    
    # Show statistics
    print("\n" + "="*80)
    print("SAMPLE MASK STATISTICS")
    print("="*80)
    
    # Check a few masks
    sample_masks = [p.replace('_preprocessed.png', '_mask.png') for p, _ in region_crops[:5]]
    for mask_path in sample_masks:
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            total_pixels = mask.size
            tissue_pixels = (mask > 0).sum()
            tissue_pct = 100 * tissue_pixels / total_pixels
            
            print(f"\n{os.path.basename(mask_path)}:")
            print(f"  Total pixels: {total_pixels:,}")
            print(f"  Tissue pixels: {tissue_pixels:,} ({tissue_pct:.1f}%)")
            print(f"  Background pixels: {total_pixels - tissue_pixels:,} ({100-tissue_pct:.1f}%)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
