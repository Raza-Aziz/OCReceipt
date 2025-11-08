from PIL import Image
import numpy as np
import cv2

def analyze_image_quality(image):
    """
    Analyze image to determine which preprocessing steps are needed.
    Returns a dict with quality metrics.
    """
    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate metrics
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Estimate noise level using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Check if image is already binary/high contrast
    unique_vals = len(np.unique(gray))
    is_binary = unique_vals < 20
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'noise_level': laplacian_var,
        'is_binary': is_binary,
        'needs_enhancement': contrast < 50 or brightness < 80 or brightness > 180
    }

def preprocess_image(image_bytes, debug=False):
    """
    Intelligently preprocess image for better OCR results.
    
    Args:
        image_bytes: Input image as bytes
        debug: If True, returns intermediate steps for visualization
    
    Returns:
        processed_image: Preprocessed image ready for OCR
        steps: Dict of intermediate images (if debug=True)
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    steps = {'original': image.copy()} if debug else {}
    
    # Analyze image quality
    quality = analyze_image_quality(image)
    
    # Step 1: Convert to grayscale (always beneficial for text OCR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug:
        steps['grayscale'] = gray.copy()
    
    # Step 2: Noise reduction (only if image is noisy)
    if quality['noise_level'] < 100:  # Low sharpness indicates noise
        # Use bilateral filter to preserve edges while removing noise
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        if debug:
            steps['denoised'] = denoised.copy()
        gray = denoised
    
    # Step 3: Contrast enhancement (if needed)
    if quality['needs_enhancement']:
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        if debug:
            steps['contrast_enhanced'] = enhanced.copy()
        gray = enhanced
    
    # Step 4: Adaptive thresholding for better text separation
    # This is crucial for payment receipts with varying backgrounds
    # Use adaptive threshold instead of global to handle uneven lighting
    binary = cv2.adaptiveThreshold(
        gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        2
    )
    if debug:
        steps['binary'] = binary.copy()
    
    # Step 5: Morphological operations to clean up text
    # Remove small noise and connect broken characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if debug:
        steps['morphological'] = morphed.copy()
    
      
    # Convert back to PIL Image for compatibility
    final_image = Image.fromarray(binary)
    
    if debug:
        steps['final'] = binary
        return final_image, steps
    
    return final_image, None

def convert_cv_to_pil(cv_image):
    """Convert OpenCV image to PIL Image"""
    if len(cv_image.shape) == 2:  # Grayscale
        return Image.fromarray(cv_image)
    else:  # Color
        return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))