
import cv2
import numpy as np
import sys
import os

"""
BEST PRACTICE FOR FAINT TEXT:
1. Division Normalization (Turns gray paper white, keeps text gray)
2. CLAHE (Makes faint text darker)
3. Grayscale Output (Tesseract reads this better than jagged black/white)
"""

def estimate_skew(gray):
    # Detect edges and find lines to determine rotation
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None: return 0.0
    
    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta - np.pi / 2) * (180 / np.pi)
        if -15 < angle < 15:
            angles.append(angle)
            
    return float(np.median(angles)) if angles else 0.0

def deskew(gray):
    angle = estimate_skew(gray)
    if abs(angle) < 0.1: return gray
    
    # Rotate to fix skew
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def process_image(input_path, output_path):
    print(f"[Preprocess] Enhancing {input_path}")
    img = cv2.imread(input_path)
    if img is None: raise FileNotFoundError(input_path)
    
    # 1. Convert to Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 2. Background Division (The "Whitener")
    # Blur heavily to get the background color, then divide.
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    clean = cv2.divide(gray, bg, scale=255)

    # 3. Boost Contrast (The "Ink Darkener")
    # CLAHE makes faint text pop without adding noise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(clean)

    # 4. Light Denoise & Deskew
    denoised = cv2.fastNlMeansDenoising(contrast, h=3)
    final = deskew(denoised)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess.py <input> <output>")
        sys.exit(1)
    process_image(sys.argv[1], sys.argv[2])


# import cv2
# import numpy as np
# import sys
# import os

# def get_skew_angle(image):
#     # 1. Binarize just for Angle Detection
#     _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
#     # 2. Find coordinates of all white pixels (text)
#     coords = np.column_stack(np.where(thresh > 0))
#     if len(coords) < 10:
#         return 0.0

#     # 3. Get the rotated rectangle that fits the text
#     angle = cv2.minAreaRect(coords)[-1]

#     # 4. Handle OpenCV's angle eccentricities (-90 to 0)
#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle
        
#     # --- SAFETY CLAMP ---
#     # Newspaper scans are rarely > 5-10 degrees crooked.
#     # If we detect a huge angle (like 89 deg), it's likely a bug 
#     # (confusing vertical lines for horizontal ones).
#     if abs(angle) > 10.0:
#         print(f"   [Warn] Detected suspicious angle {angle:.2f}°. Ignoring rotation.")
#         return 0.0

#     return angle

# def process_image(input_path, output_path):
#     print(f"[Preprocess] Cleaning {input_path} -> {output_path}")

#     img = cv2.imread(input_path)
#     if img is None:
#         raise FileNotFoundError(input_path)

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # 1. Median Blur - The "Static" Killer
#     # Removes small grain/noise before we threshold
#     blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)

#     # 2. Adaptive Threshold - The "Whitener"
#     # BlockSize=51 (large area), C=15 (aggressive whitening)
#     binary = cv2.adaptiveThreshold(
#         blurred, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         11, 2
#     )

#     # 3. Deskew (Safe Mode)
#     angle = get_skew_angle(binary)
    
#     if abs(angle) > 0.1:
#         print(f"   -> Rotating by {angle:.2f} degrees")
#         (h, w) = binary.shape[:2]
#         center = (w // 2, h // 2)
#         M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
#         # Rotate the binary image
#         # borderValue=255 ensures we fill the corners with white
#         final = cv2.warpAffine(binary, M, (w, h),
#                                flags=cv2.INTER_CUBIC,
#                                borderMode=cv2.BORDER_CONSTANT,
#                                borderValue=255)
#     else:
#         final = binary

#     cv2.imwrite(output_path, final)

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python preprocess.py <input.png> <output.png>")
#         sys.exit(1)

#     process_image(sys.argv[1], sys.argv[2])
