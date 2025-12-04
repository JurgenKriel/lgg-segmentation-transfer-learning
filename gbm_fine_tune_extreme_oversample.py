import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, Reshape, Activation, 
                                      BatchNormalization, Dropout, Concatenate, 
                                      Conv2DTranspose)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical, Sequence
from PIL import Image
from tqdm import tqdm
import datetime

Image.MAX_IMAGE_PIXELS = None

# ==================================================================
# GBM MODEL ARCHITECTURE
# ==================================================================
def RELU_BN(x): 
    return BatchNormalization(axis=-1)(Activation('relu')(x))

def CONV(x, nf, sz, wd, p, stride=1):
    x = Conv2D(nf, (sz, sz), strides=(stride, stride), padding='same', 
               kernel_initializer='he_uniform', kernel_regularizer=l2(wd))(x)
    return Dropout(p)(x) if p else x

def CONV_RELU_BN(x, nf, sz=3, wd=0, p=0, stride=1):
    return CONV(RELU_BN(x), nf, sz, wd=wd, p=p, stride=stride)

def DENSE_BLOCK(n, x, growth_rate, p, wd):
    added = []
    for i in range(n):
        b = CONV_RELU_BN(x, growth_rate, p=p, wd=wd)
        x = Concatenate(axis=-1)([x, b])
        added.append(b)
    return x, added

def Transition_DOWN(x, p, wd):
    return CONV_RELU_BN(x, tf.keras.backend.int_shape(x)[-1], sz=1, p=p, wd=wd, stride=2)

def Down_Path(x, nb_layers, growth_rate, p, wd):
    skips = []
    for i, n in enumerate(nb_layers):
        x, added = DENSE_BLOCK(n, x, growth_rate, p, wd)
        skips.append(x)
        x = Transition_DOWN(x, p=p, wd=wd)
    return skips, added

def Transition_UP(added, wd=0):
    x = Concatenate(axis=-1)(added)
    _, r, c, ch = tf.keras.backend.int_shape(x)
    return Conv2DTranspose(ch, (3, 3), strides=(2, 2), padding='same', 
                           kernel_initializer='he_uniform',
                           kernel_regularizer=l2(wd))(x)

def Up_Path(added, skips, nb_layers, growth_rate, p, wd):
    for i, n in enumerate(nb_layers):
        x = Transition_UP(added, wd)
        x = Concatenate(axis=-1)([x, skips[i]])
        x, added = DENSE_BLOCK(n, x, growth_rate, p, wd)
    return x

def reverse(a): 
    return list(reversed(a))

def GBM_WSSM_Model(nb_classes, img_input, nb_dense_block=6,
                   growth_rate=16, nb_filter=48, 
                   nb_layers_per_block=[4, 5, 7, 10, 12, 15], 
                   p=0.2, wd=1e-4):
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else:
        nb_layers = [nb_layers_per_block] * nb_dense_block

    x = CONV(img_input, nb_filter, 3, wd, 0)
    skips, added = Down_Path(x, nb_layers, growth_rate, p, wd)
    x = Up_Path(added, reverse(skips[:-1]), reverse(nb_layers[:-1]), growth_rate, p, wd)

    x = CONV(x, nb_classes, 1, wd, 0)
    _, r, c, f = tf.keras.backend.int_shape(x)
    x = Reshape((-1, nb_classes))(x)
    return Activation('softmax')(x)

# ==================================================================
# STAIN NORMALIZATION
# ==================================================================
def normalize_stain_macenko(img, target_concentrations=None):
    img = img.astype(np.float32)
    img = np.maximum(img, 1e-6)
    od = -np.log(img / 255.0)
    
    od_hat = od[~np.any(od < 0.15, axis=1)]
    
    if len(od_hat) < 2:
        return img
    
    eigvals, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
    that = od_hat.dot(eigvecs[:, 1:3])
    phi = np.arctan2(that[:, 1], that[:, 0])
    
    min_phi = np.percentile(phi, 1)
    max_phi = np.percentile(phi, 99)
    
    v1 = eigvecs[:, 1:3].dot(np.array([np.cos(min_phi), np.sin(min_phi)]))
    v2 = eigvecs[:, 1:3].dot(np.array([np.cos(max_phi), np.sin(max_phi)]))
    
    if np.linalg.norm(v1) > 0:
        v1 = v1 / np.linalg.norm(v1)
    if np.linalg.norm(v2) > 0:
        v2 = v2 / np.linalg.norm(v2)
    
    he = np.array([v1, v2]).T
    conc = np.linalg.lstsq(he, od.T, rcond=None)[0].T
    
    if target_concentrations is None:
        maxC = np.percentile(conc, 99, axis=0)
        conc = conc / maxC
    else:
        conc = conc * target_concentrations
    
    od_norm = conc.dot(he.T)
    img_norm = np.exp(-od_norm) * 255
    img_norm = np.clip(img_norm, 0, 255).astype(np.uint8)
    
    return img_norm

def apply_stain_normalization(img_array):
    try:
        img_uint8 = (img_array * 255).astype(np.uint8)
        img_normalized = normalize_stain_macenko(img_uint8)
        return img_normalized.astype(np.float32) / 255.0
    except:
        return img_array

# ==================================================================
# LE-FOCUSED DATA GENERATOR
# ==================================================================
class LEFocusedDataGenerator(Sequence):
    def __init__(self, image_paths, mask_paths, batch_size, image_size, num_classes,
                 le_oversample_rate=5.0, it_oversample_rate=20.0, shuffle=True, augment=True, stain_norm=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment
        self.stain_norm = stain_norm
        
        self.le_images = []
        self.it_images = []
        self.other_images = []
        
        print("Analyzing LE and IT content in dataset...")
        for idx, mask_path in enumerate(tqdm(mask_paths)):
            try:
                mask = load_img(mask_path, color_mode="grayscale")
                mask_array = img_to_array(mask)
                unique_classes = np.unique(mask_array)
                
                has_le = 2 in unique_classes
                has_it = 3 in unique_classes
                
                if has_it:
                    self.it_images.append(idx)
                elif has_le:
                    self.le_images.append(idx)
                else:
                    self.other_images.append(idx)
            except:
                self.other_images.append(idx)
        
        print(f"  IT images: {len(self.it_images)}")
        print(f"  LE images: {len(self.le_images)}")
        print(f"  Other images: {len(self.other_images)}")
        
        self.le_oversample_rate = le_oversample_rate
        self.it_oversample_rate = it_oversample_rate
        self.le_oversample_rate = le_oversample_rate
        self.create_sampling_indices()
    
    def create_sampling_indices(self):
        """Create index list with IT and LE images heavily oversampled"""
        self.indexes = list(range(len(self.image_paths)))
        
        # Oversample IT images (most critical)
        extra_it = int(len(self.it_images) * (self.it_oversample_rate - 1))
        if extra_it > 0:
            extra_samples = np.random.choice(self.it_images, size=extra_it, replace=True)
            self.indexes.extend(extra_samples)
        
        # Oversample LE images
        extra_le = int(len(self.le_images) * (self.le_oversample_rate - 1))
        if extra_le > 0:
            extra_samples = np.random.choice(self.le_images, size=extra_le, replace=True)
            self.indexes.extend(extra_samples)
        
        print(f"Total samples per epoch: {len(self.indexes)}")
        print(f"  IT oversampled {self.it_oversample_rate}x")
        print(f"  LE oversampled {self.le_oversample_rate}x")
    
    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_paths = [self.image_paths[k] for k in batch_indexes]
        batch_mask_paths = [self.mask_paths[k] for k in batch_indexes]
        X, y = self.__data_generation(batch_image_paths, batch_mask_paths)
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def augment_image(self, image, mask):
        if not self.augment:
            return image, mask
        
        if np.random.rand() > 0.5:
            image = tf.image.random_hue(image, 0.08)
            image = tf.image.random_saturation(image, 0.6, 1.6)
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.7, 1.3)
        
        if np.random.rand() > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        
        if np.random.rand() > 0.5:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)
        
        k = np.random.randint(0, 4)
        if k > 0:
            image = tf.image.rot90(image, k=k)
            mask = tf.image.rot90(mask, k=k)
        
        return image, mask
    
    def __data_generation(self, batch_image_paths, batch_mask_paths):
        X = np.empty((self.batch_size, *self.image_size))
        y = np.empty((self.batch_size, self.image_size[0] * self.image_size[1], self.num_classes))
        
        for i, (img_path, mask_path) in enumerate(zip(batch_image_paths, batch_mask_paths)):
            img = load_img(img_path, target_size=self.image_size[:2])
            img_array = img_to_array(img) / 255.0
            
            if self.stain_norm:
                img_array = apply_stain_normalization(img_array)
            
            mask = load_img(mask_path, target_size=self.image_size[:2], color_mode="grayscale")
            mask_array = img_to_array(mask)
            
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
            mask_tensor = tf.convert_to_tensor(mask_array, dtype=tf.float32)
            img_tensor, mask_tensor = self.augment_image(img_tensor, mask_tensor)
            
            X[i,] = img_tensor.numpy()
            
            mask_categorical = to_categorical(mask_tensor.numpy(), num_classes=self.num_classes)
            y[i,] = mask_categorical.reshape(-1, self.num_classes)
        
        return X, y

# ==================================================================
# MODEL WITH 8→4 MAPPING
# ==================================================================
def create_model_with_mapping(weights_path, num_classes=4, input_shape=(1024, 1024, 3)):
    print("\nBuilding GBM model architecture...")
    img_input = Input(shape=input_shape, name='input')
    output = GBM_WSSM_Model(
        nb_classes=8,
        img_input=img_input,
        nb_dense_block=6,
        growth_rate=16,
        nb_filter=48,
        nb_layers_per_block=[4, 5, 7, 10, 12, 15],
        p=0.2,
        wd=1e-4
    )
    
    original_model = Model(inputs=img_input, outputs=output)
    
    print(f"Loading pre-trained weights from {weights_path}...")
    original_model.load_weights(weights_path)
    print("✓ Weights loaded")
    
    # Find last Conv2D
    conv_layer = None
    for layer in reversed(original_model.layers):
        if isinstance(layer, Conv2D):
            conv_layer = layer
            break
    
    feature_layer = original_model.layers[original_model.layers.index(conv_layer) - 1]
    feature_output = feature_layer.output
    
    # Add mapping layers
    x = Conv2D(16, (1, 1), activation='relu', name='mapping_intermediate')(feature_output)
    x = Conv2D(num_classes, (1, 1), name='lgg_4class_conv')(x)
    
    _, r, c, f = tf.keras.backend.int_shape(x)
    x = Reshape((-1, num_classes))(x)
    x = Activation('softmax')(x)
    
    new_model = Model(inputs=img_input, outputs=x)
    
    return new_model, original_model

# ==================================================================
# FOCAL LOSS
# ==================================================================
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        ce = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1 - y_pred, gamma)
        focal_loss_val = alpha * focal_weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal_loss_val, axis=-1))
    return loss_fn

# ==================================================================
# PROGRESSIVE UNFREEZING
# ==================================================================
def progressive_unfreeze_training(model, original_model, train_gen, val_gen, home_dir):
    """
    Phase 1: Train mapping head only (frozen encoder)
    Phase 2: Unfreeze last 10 layers
    Phase 3: Unfreeze last 25 layers
    Phase 4: Unfreeze last 50 layers
    """
    
    phases = [
        {"name": "Phase 1: Head Only", "unfreeze_last": 0, "epochs": 30, "lr": 1e-3},
        {"name": "Phase 2: Last 10 Layers", "unfreeze_last": 10, "epochs": 30, "lr": 1e-4},
        {"name": "Phase 3: Last 25 Layers", "unfreeze_last": 25, "epochs": 30, "lr": 1e-5},
        {"name": "Phase 4: Last 50 Layers", "unfreeze_last": 50, "epochs": 30, "lr": 1e-5},
    ]
    
    for phase in phases:
        print("\n" + "="*80)
        print(phase["name"])
        print("="*80)
        
        # Freeze all original layers
        for layer in original_model.layers:
            layer.trainable = False
        
        # Unfreeze last N layers
        if phase["unfreeze_last"] > 0:
            total_layers = len(original_model.layers)
            unfreeze_from = max(0, total_layers - phase["unfreeze_last"])
            for i in range(unfreeze_from, total_layers):
                original_model.layers[i].trainable = True
            print(f"Unfrozen layers {unfreeze_from} to {total_layers-1}")
        
        # Count trainable params
        trainable_params = sum([tf.size(v).numpy() for v in model.trainable_variables])
        print(f"Trainable params: {trainable_params:,}")
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=phase["lr"]),
            loss=focal_loss(gamma=2.0, alpha=0.25),
            metrics=['accuracy']
        )
        
        # Callbacks
        log_dir = os.path.join(home_dir, f"tensorboard_logs/progressive_{phase['name'].replace(' ', '_')}", 
                               datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'model_checkpoints/progressive-{phase["name"].replace(" ", "_")}-{{epoch:02d}}-{{val_loss:.4f}}.keras',
                monitor='val_loss', save_best_only=True, mode='min'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            ),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
            )
        ]
        
        # Train
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=phase["epochs"],
            callbacks=callbacks
        )
        
        # Save after each phase
        model_name = f'gbm_progressive_{phase["name"].replace(" ", "_").lower()}.keras'
        model.save(os.path.join(home_dir, model_name))
        print(f"✓ Saved: ~/{model_name}")

# ==================================================================
# MAIN
# ==================================================================
def main():
    print("="*80)
    print("EXTREME IT/LE OVERSAMPLING + PROGRESSIVE UNFREEZING")
    print("="*80)
    
    input_shape = (1024, 1024, 3)
    batch_size = 1
    num_classes = 4
    
    weights_path = "/vast/projects/Histology_Glioma_ML/GBM_WSSM/GBM_WSSM.h5"
    images_dir = "/vast/projects/Histology_Glioma_ML/Annotated_HE_ML_Validation/all_images/preprocessed"
    
    # Load dataset
    print("\n# Load and validate dataset")
    print("\nLoading dataset...")
    all_image_paths = sorted(glob.glob(os.path.join(images_dir, "*_preprocessed.png")))
    all_mask_paths = [p.replace('_preprocessed.png', '_mask.png') for p in all_image_paths]
    
    valid_pairs = []
    for img_path, mask_path in zip(all_image_paths, all_mask_paths):
        if os.path.exists(mask_path):
            valid_pairs.append((img_path, mask_path))
        else:
            print(f"  Skipping (no mask): {os.path.basename(img_path)}")
    
    image_paths = [p[0] for p in valid_pairs]
    mask_paths = [p[1] for p in valid_pairs]
    
    print(f"Found {len(image_paths)} valid image-mask pairs")
    
    val_split = 0.2
    val_size = int(len(image_paths) * val_split)
    train_images = image_paths[:-val_size]
    train_masks = mask_paths[:-val_size]
    val_images = image_paths[-val_size:]
    val_masks = mask_paths[-val_size:]
    
    print(f"Training: {len(train_images)}, Validation: {len(val_images)}")
    
    # Create generators
    print("\nCreating LE-focused data generators...")
    train_gen = LEFocusedDataGenerator(
        train_images, train_masks, batch_size, input_shape, num_classes,
        le_oversample_rate=5.0, it_oversample_rate=20.0, shuffle=True, augment=True, stain_norm=True
    )
    
    val_gen = LEFocusedDataGenerator(
        val_images, val_masks, batch_size, input_shape, num_classes,
        le_oversample_rate=1.0, shuffle=False, augment=False, stain_norm=True
    )
    
    # Create model
    model, original_model = create_model_with_mapping(weights_path, num_classes, input_shape)
    
    os.makedirs('model_checkpoints', exist_ok=True)
    home_dir = os.path.expanduser("~")
    
    # Progressive unfreezing
    progressive_unfreeze_training(model, original_model, train_gen, val_gen, home_dir)
    
    print("\n" + "="*80)
    print("PROGRESSIVE TRAINING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
