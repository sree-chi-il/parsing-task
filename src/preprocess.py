import cv2
import numpy as np
import sys
import os

def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    if len(coords) == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated

def process_image(input_path, output_path):
    print(f"[Preprocess] {input_path} -> {output_path}")

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(input_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoise = cv2.fastNlMeansDenoising(gray, h=10)

    thresh = cv2.adaptiveThreshold(
        denoise, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 15
    )

    inverted = cv2.bitwise_not(thresh)
    desk = deskew(inverted)
    final = cv2.bitwise_not(desk)

    cv2.imwrite(output_path, final)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess.py <input.png> <output.png>")
        sys.exit(1)

    process_image(sys.argv[1], sys.argv[2])
