import cv2
import os
import numpy as np


# ============================================================
# PixelForge DriftSense — Phase 3B
# Strong Pixel Drift Generator
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
    "drifted_strong"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Drift parameters
# ------------------------------------------------------------

NOISE_STD = 30
BRIGHTNESS_SHIFT = 20

BLUR_KERNEL = 5

CORRUPTION_RATIO = 0.02


# ------------------------------------------------------------
# Add Gaussian noise
# ------------------------------------------------------------

def add_noise(image, std):

    noise = np.random.normal(
        0,
        std,
        image.shape
    )

    result = image.astype(np.float32) + noise

    result = np.clip(
        result,
        0,
        255
    )

    return result.astype(np.uint8)


# ------------------------------------------------------------
# Add random pixel corruption
# ------------------------------------------------------------

def corrupt_pixels(image, ratio):

    result = image.copy()

    total_pixels = image.shape[0] * image.shape[1]

    number_of_pixels = int(
        total_pixels * ratio
    )

    height, width = image.shape[:2]

    ys = np.random.randint(
        0,
        height,
        number_of_pixels
    )

    xs = np.random.randint(
        0,
        width,
        number_of_pixels
    )

    for x, y in zip(xs, ys):

        result[y, x] = np.random.randint(
            0,
            256,
            3
        )

    return result


# ------------------------------------------------------------
# Process images
# ------------------------------------------------------------

print()
print("=" * 65)
print("PixelForge DriftSense - Phase 3B")
print("Strong Pixel Drift Generator")
print("=" * 65)

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

        print(
            f"Skipping Pair {i}: image not found"
        )

        continue


    # --------------------------------------------------------
    # 1. Brightness drift
    # --------------------------------------------------------

    drifted = image.astype(np.int16)

    drifted += BRIGHTNESS_SHIFT

    drifted = np.clip(
        drifted,
        0,
        255
    ).astype(np.uint8)


    # --------------------------------------------------------
    # 2. Gaussian noise
    # --------------------------------------------------------

    drifted = add_noise(
        drifted,
        NOISE_STD
    )


    # --------------------------------------------------------
    # 3. Blur
    # --------------------------------------------------------

    drifted = cv2.GaussianBlur(
        drifted,
        (BLUR_KERNEL, BLUR_KERNEL),
        0
    )


    # --------------------------------------------------------
    # 4. Pixel corruption
    # --------------------------------------------------------

    drifted = corrupt_pixels(
        drifted,
        CORRUPTION_RATIO
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    cv2.imwrite(
        output_path,
        drifted
    )

    processed += 1

    print(
        f"Pair {i}: Strong drift applied"
    )


# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

print()
print("=" * 65)
print("STRONG DRIFT GENERATION COMPLETED")
print("=" * 65)

print(f"Images processed    : {processed}")
print(f"Noise STD           : {NOISE_STD}")
print(f"Brightness shift    : +{BRIGHTNESS_SHIFT}")
print(f"Blur kernel         : {BLUR_KERNEL} x {BLUR_KERNEL}")
print(f"Pixel corruption    : {CORRUPTION_RATIO * 100:.1f}%")
print(f"Output directory    : {OUTPUT_DIR}")

print("=" * 65)