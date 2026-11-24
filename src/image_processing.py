
import cv2
import numpy as np
from pathlib import Path


# ==========================================
# 1. LOAD IMAGE USING OPENCV
# ==========================================

def load_image(image_path):

    image_path = str(image_path)

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Unable to load image: {image_path}"
        )

    return image


# ==========================================
# 2. CONVERT BGR TO RGB
# ==========================================

def convert_to_rgb(image):

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return rgb_image


# ==========================================
# 3. RESIZE IMAGE
# ==========================================

def resize_image(image, size=(128, 128)):

    resized = cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_AREA
    )

    return resized


# ==========================================
# 4. GRAYSCALE CONVERSION
# ==========================================

def convert_to_grayscale(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return gray


# ==========================================
# 5. NOISE REDUCTION
# ==========================================

def remove_noise(image):

    denoised = cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )

    return denoised


# ==========================================
# 6. CONTRAST ENHANCEMENT
# ==========================================

def enhance_contrast(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l)

    enhanced_lab = cv2.merge(
        (enhanced_l, a, b)
    )

    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced


# ==========================================
# 7. MAIN IMAGE PROCESSING FUNCTION
# ==========================================

def process_image(image_path):

    image = load_image(image_path)

    rgb_image = convert_to_rgb(image)

    return rgb_image


# ==========================================
# 8. TEST THE PROCESSING FUNCTIONS
# ==========================================

if __name__ == "__main__":

    print("========== OPENCV IMAGE PROCESSING ==========")

    # Select one existing defect crop
    ROOT = Path(__file__).resolve().parent.parent

    image_path = (
        ROOT / "dataset" / "cnn_crops"
        / "silk_spot"
    )

    image_files = (
        list(image_path.glob("*.jpg"))
        + list(image_path.glob("*.png"))
    )

    if not image_files:

        print("No test images found!")

    else:

        test_image = load_image(image_files[0])

        rgb = convert_to_rgb(test_image)

        resized = resize_image(rgb)

        gray = convert_to_grayscale(test_image)

        denoised = remove_noise(test_image)

        enhanced = enhance_contrast(test_image)

        print("OpenCV version:", cv2.__version__)

        print("Original image:", test_image.shape)

        print("RGB image:", rgb.shape)

        print("Resized image:", resized.shape)

        print("Grayscale image:", gray.shape)

        print("Denoised image:", denoised.shape)

        print("Enhanced image:", enhanced.shape)

        print("\nOPENCV PROCESSING COMPLETED")
