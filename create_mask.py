#!/usr/bin/env python3
"""
Generate a grayscale weight mask for face matching.
Bright areas (white) = important to match, Dark areas = less important
"""

import cv2
import numpy as np
import os

def create_face_mask(image_path, output_path, target_size=256):
    """
    Create a weight mask highlighting the face region.
    
    Args:
        image_path: Path to the source image
        output_path: Path to save the weight mask
        target_size: Output size (square)
    """
    print(f"Loading image: {image_path}")
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    print(f"Original image size: {img.shape[:2]}")
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create base mask (start with black background)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    
    # Try to detect face
    print("Detecting face...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        print(f"Found {len(faces)} face(s)!")
        # Use the largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        
        print(f"Face location: x={x}, y={y}, width={w}, height={h}")
        
        # Calculate center of face
        center_x = x + w // 2
        center_y = y + h // 2
        
        print("Creating simple gradient mask...")
        
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:img.shape[0], :img.shape[1]]
        
        # 1. Face zone (max brightness 255) - bigger to capture full face
        face_radius_x = int(w * 0.70)  # Increased from 0.55
        face_radius_y = int(h * 0.80)  # Increased from 0.65
        face_dist = np.sqrt(((x_coords - center_x) / face_radius_x) ** 2 + 
                           ((y_coords - center_y) / face_radius_y) ** 2)
        face_zone = 255 * np.exp(-2.0 * face_dist ** 2)
        
        # 2. Body/silhouette gradient (extends down and sides) - darker
        body_radius_x = int(w * 1.0)
        body_radius_y = int(h * 1.6)
        body_dist = np.sqrt(((x_coords - center_x) / body_radius_x) ** 2 + 
                           ((y_coords - center_y) / body_radius_y) ** 2)
        body_zone = 90 * np.exp(-1.5 * body_dist ** 2)  # Reduced from 120 and steeper falloff
        
        # Combine zones (take maximum)
        mask = np.maximum(face_zone, body_zone)
        
    else:
        print("No face detected! Using simple center gradient as fallback...")
        # Fallback: simple center-weighted gradient
        center_y, center_x = img.shape[0] // 2, img.shape[1] // 2
        
        y_coords, x_coords = np.ogrid[:img.shape[0], :img.shape[1]]
        
        # Face center
        face_radius = min(img.shape[0], img.shape[1]) // 5
        face_dist = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        face_zone = 255 * np.exp(-2.0 * (face_dist / face_radius) ** 2)
        
        # Body gradient
        body_radius = min(img.shape[0], img.shape[1]) // 2.5
        body_dist = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        body_zone = 120 * np.exp(-1.2 * (body_dist / body_radius) ** 2)
        
        mask = np.maximum(face_zone, body_zone)
    
    # Clean up: set very low values to pure black
    mask = np.clip(mask, 0, 255)
    mask[mask < 15] = 0  # Background threshold
    mask = mask.astype(np.uint8)
    
    # Resize to target size
    print(f"Resizing mask to {target_size}x{target_size}")
    mask_resized = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # Save the mask
    cv2.imwrite(output_path, mask_resized)
    print(f"✓ Mask saved to: {output_path}")
    
    # Create a visualization for preview
    preview_path = output_path.replace('.png', '_preview.png')
    
    # Resize original image to match
    img_resized = cv2.resize(img, (target_size, target_size))
    
    # Create colored overlay (red channel shows weight)
    overlay = img_resized.copy()
    overlay[:, :, 2] = cv2.addWeighted(overlay[:, :, 2], 0.6, mask_resized, 0.4, 0)
    
    # Stack side by side: original | mask | overlay
    comparison = np.hstack([img_resized, cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR), overlay])
    cv2.imwrite(preview_path, comparison)
    print(f"✓ Preview saved to: {preview_path}")
    
    # Print statistics
    print(f"\nMask Statistics:")
    print(f"  Min weight: {mask_resized.min()}")
    print(f"  Max weight: {mask_resized.max()}")
    print(f"  Mean weight: {mask_resized.mean():.1f}")
    
    return mask_resized


if __name__ == "__main__":
    # Paths
    source_image = "src/app/calculate/YammanappaSir.png"
    
    # Generate both 256 and 128 versions
    targets = [
        ("src/app/calculate/weights256.png", 256),
        ("src/app/calculate/weights128.png", 128),
    ]
    
    if not os.path.exists(source_image):
        print(f"ERROR: Source image not found at {source_image}")
        print("Please make sure YammanappaSir.png is in src/app/calculate/")
        exit(1)
    
    print("=" * 60)
    print("Creating Weight Masks for Yammanappa Sir's Image")
    print("=" * 60)
    
    try:
        for output_mask, size in targets:
            print(f"\n>>> Creating {size}x{size} mask...")
            mask = create_face_mask(source_image, output_mask, target_size=size)
            print(f"✓ {size}x{size} mask complete!")
        
        print("\n" + "=" * 60)
        print("✓ SUCCESS! All weight masks created.")
        print("=" * 60)
        print("\nFiles created:")
        print("  - src/app/calculate/weights256.png")
        print("  - src/app/calculate/weights128.png")
        print("\nPreviews:")
        print("  - src/app/calculate/weights256_preview.png")
        print("  - src/app/calculate/weights128_preview.png")
        print("\nNext steps:")
        print("1. Check the previews to verify the masks look good")
        print("2. Rebuild the app: cargo run --release")
        print("3. The bright areas will be matched more accurately in morphing")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
