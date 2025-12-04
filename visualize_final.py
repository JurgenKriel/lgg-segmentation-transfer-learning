import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Class names and colors
class_names = ['Background', 'CT', 'LE', 'IT']
colors = ['black', 'red', 'lime', 'blue']
cmap = ListedColormap(colors)

def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        ce = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1 - y_pred, gamma)
        focal_loss_val = alpha * focal_weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal_loss_val, axis=-1))
    return loss_fn

def visualize_model(model_path, images_dir, output_dir, num_samples=20):
    """Create 6-panel visualizations"""
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, custom_objects={'loss_fn': focal_loss()}, compile=False)
    print("✓ Model loaded")
    
    # Get valid image-mask pairs
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*_preprocessed.png")))
    mask_paths = [p.replace('_preprocessed.png', '_mask.png') for p in image_paths]
    valid_pairs = [(img, mask) for img, mask in zip(image_paths, mask_paths) if os.path.exists(mask)]
    
    # Sample randomly
    if len(valid_pairs) > num_samples:
        indices = np.random.choice(len(valid_pairs), num_samples, replace=False)
        valid_pairs = [valid_pairs[i] for i in indices]
    
    print(f"Visualizing {len(valid_pairs)} images...")
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path, mask_path in valid_pairs:
        # Load image
        img = load_img(img_path, target_size=(1024, 1024))
        img_array = img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Load ground truth
        mask = load_img(mask_path, target_size=(1024, 1024), color_mode='grayscale')
        mask_array = img_to_array(mask).squeeze()
        
        # Predict
        pred_flat = model.predict(img_batch, verbose=0)  # (1, H*W, 4)
        pred_probs = pred_flat.reshape(1024, 1024, 4)
        pred_classes = np.argmax(pred_probs, axis=-1)
        
        # Create 6-panel figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Row 1: Image, Ground Truth, Prediction
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask_array, cmap=cmap, vmin=0, vmax=3)
        axes[0, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(pred_classes, cmap=cmap, vmin=0, vmax=3)
        axes[0, 2].set_title('Prediction', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Probability heatmaps for LE, IT, CT
        for idx, (class_idx, class_name) in enumerate([(2, 'LE'), (3, 'IT'), (1, 'CT')]):
            im = axes[1, idx].imshow(pred_probs[:, :, class_idx], cmap='hot', vmin=0, vmax=1)
            axes[1, idx].set_title(f'{class_name} Probability', fontsize=14, fontweight='bold')
            axes[1, idx].axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[1, idx], fraction=0.046, pad=0.04)
            cbar.set_label('Probability', fontsize=10)
        
        plt.suptitle(f'{os.path.basename(img_path)}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_name = os.path.basename(img_path).replace('_preprocessed.png', '_final_viz.png')
        output_path = os.path.join(output_dir, output_name)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_name}")
    
    print(f"\n✓ All visualizations saved to {output_dir}")

if __name__ == "__main__":
    model_path = os.path.expanduser("~/gbm_progressive_phase_4:_last_50_layers.keras")
    images_dir = "/vast/projects/Histology_Glioma_ML/Annotated_HE_ML_Validation/all_images/preprocessed"
    output_dir = os.path.expanduser("~/final_model_visualizations")
    
    visualize_model(model_path, images_dir, output_dir, num_samples=20)
