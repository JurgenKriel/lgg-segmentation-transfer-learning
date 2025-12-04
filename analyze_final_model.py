import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        ce = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1 - y_pred, gamma)
        focal_loss_val = alpha * focal_weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal_loss_val, axis=-1))
    return loss_fn

def main():
    model_path = os.path.expanduser("~/gbm_progressive_phase_4:_last_50_layers.keras")
    images_dir = "/vast/projects/Histology_Glioma_ML/Annotated_HE_ML_Validation/all_images/preprocessed"
    
    print("="*80)
    print("FINAL MODEL ANALYSIS (WITH TISSUE-AWARE MASKS)")
    print("="*80)
    
    print("\nLoading model...")
    model = load_model(model_path, custom_objects={'loss_fn': focal_loss()}, compile=False)
    print("✓ Model loaded")
    
    # Get valid pairs
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*_preprocessed.png")))
    mask_paths = [p.replace('_preprocessed.png', '_mask.png') for p in image_paths]
    valid_pairs = [(img, mask) for img, mask in zip(image_paths, mask_paths) if os.path.exists(mask)]
    
    print(f"\nAnalyzing {len(valid_pairs)} images...")
    
    all_true = []
    all_pred = []
    all_probs = []
    
    for img_path, mask_path in valid_pairs:
        img = load_img(img_path, target_size=(1024, 1024))
        img_array = img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        mask = load_img(mask_path, target_size=(1024, 1024), color_mode='grayscale')
        mask_array = img_to_array(mask).squeeze().flatten().astype(int)
        
        pred_flat = model.predict(img_batch, verbose=0)
        pred_probs = pred_flat[0]
        pred_classes = np.argmax(pred_probs, axis=-1)
        
        all_true.extend(mask_array)
        all_pred.extend(pred_classes)
        all_probs.append(pred_probs)
    
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_probs = np.vstack(all_probs)
    
    # Confusion matrix
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)
    cm = confusion_matrix(all_true, all_pred)
    print("           Pred: BG      CT      LE      IT")
    class_names = ['Background', 'CT', 'LE', 'IT']
    for i, name in enumerate(class_names):
        print(f"True {name:10s}: {cm[i,0]:7d} {cm[i,1]:7d} {cm[i,2]:7d} {cm[i,3]:7d}")
    
    # Per-class metrics
    print("\n" + "="*80)
    print("PER-CLASS METRICS")
    print("="*80)
    print(classification_report(all_true, all_pred, target_names=class_names, digits=4, zero_division=0))
    
    # Detailed probability analysis
    print("\n" + "="*80)
    print("PROBABILITY ANALYSIS")
    print("="*80)
    
    for class_idx, class_name in enumerate(class_names):
        true_class_mask = (all_true == class_idx)
        
        if true_class_mask.sum() > 0:
            true_probs = all_probs[true_class_mask, class_idx]
            
            print(f"\n{class_name} (n={true_class_mask.sum():,} pixels):")
            print(f"  Probability distribution:")
            print(f"    Mean: {true_probs.mean():.4f}")
            print(f"    Median: {np.median(true_probs):.4f}")
            print(f"    Max: {true_probs.max():.4f}")
            print(f"    % > 0.3: {100*(true_probs > 0.3).sum()/len(true_probs):.1f}%")
            print(f"    % > 0.5: {100*(true_probs > 0.5).sum()/len(true_probs):.1f}%")
            
            # What class is winning when it should be this class?
            pred_for_this_class = all_pred[true_class_mask]
            print(f"  When ground truth is {class_name}, predicted as:")
            for pred_class in range(4):
                count = (pred_for_this_class == pred_class).sum()
                pct = 100 * count / len(pred_for_this_class)
                print(f"    {class_names[pred_class]:12s}: {count:10,} ({pct:5.1f}%)")
    
    # Class distribution
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION")
    print("="*80)
    
    print("\nGround Truth:")
    for i, name in enumerate(class_names):
        count = (all_true == i).sum()
        pct = 100 * count / len(all_true)
        print(f"  {name:12s}: {count:10,} ({pct:5.2f}%)")
    
    print("\nPredictions:")
    for i, name in enumerate(class_names):
        count = (all_pred == i).sum()
        pct = 100 * count / len(all_pred)
        print(f"  {name:12s}: {count:10,} ({pct:5.2f}%)")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    # Check if IT has any probability
    it_true_mask = (all_true == 3)
    if it_true_mask.sum() > 0:
        it_probs = all_probs[it_true_mask, 3]
        if it_probs.max() < 0.3:
            print("\n⚠️  IT probabilities are very low (max < 0.3)")
            print("    → Model is not learning IT features properly")
            print("    → Recommendation: Unfreeze MORE layers or train from scratch")
        else:
            print("\n✓ IT probabilities exist but losing to other classes")
            print("  → Try custom thresholding or class weights")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
