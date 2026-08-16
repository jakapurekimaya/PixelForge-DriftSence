import cv2
import os
import numpy as np


# ============================================================
# PixelForge DriftSense — Phase 3
# Pixel Drift / Degradation Generator
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "train"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "drifted"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Drift settings
# ------------------------------------------------------------

NOISE_STD = 15
BRIGHTNESS_SHIFT = 15


# ------------------------------------------------------------
# Add Gaussian noise
# ------------------------------------------------------------

def add_gaussian_noise(image, std):

    noise = np.random.normal(
        0,
        std,
        image.shape
    )

    noisy_image = image.astype(np.float32) + noise

    noisy_image = np.clip(
        noisy_image,
        0,
        255
    )

    return noisy_image.astype(np.uint8)


# ------------------------------------------------------------
# Process all search images
# ------------------------------------------------------------

print()
print("=" * 60)
print("PixelForge DriftSense - Phase 3")
print("Pixel Drift Generator")
print("=" * 60)

processed = 0


for i in range(1, 31):

    filename = f"pair_{i:04d}_search.png"

    input_path = os.path.join(
        INPUT_DIR,
        filename
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    image = cv2.imread(input_path)

    if image is None:
        print(f"Skipping {filename} - image not found")
        continue


    # --------------------------------------------------------
    # Brightness drift
    # --------------------------------------------------------

    drifted = image.astype(np.int16)

    drifted = drifted + BRIGHTNESS_SHIFT

    drifted = np.clip(
        drifted,
        0,
        255
    ).astype(np.uint8)


    # --------------------------------------------------------
    # Gaussian pixel noise
    # --------------------------------------------------------

    drifted = add_gaussian_noise(
        drifted,
        NOISE_STD
    )


    # --------------------------------------------------------
    # Save drifted image
    # --------------------------------------------------------

    cv2.imwrite(
        output_path,
        drifted
    )

    processed += 1

    print(
        f"Pair {i}: Drift applied -> {filename}"
    )


# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

print()
print("=" * 60)
print("DRIFT GENERATION COMPLETED")
print("=" * 60)

print(f"Images processed : {processed}")
print(f"Noise STD        : {NOISE_STD}")
print(f"Brightness shift : +{BRIGHTNESS_SHIFT}")
print(f"Output directory : {OUTPUT_DIR}")

print("=" * 60)