import cv2
import numpy as np
import os


# ============================================================
# DRIFT-SENSE - Phase 1
# Synthetic DRAM Dataset Generator
# Version 1: Basic Reference/Search Image Generation
# ============================================================


# -----------------------------
# Folder configuration
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
METADATA_DIR = os.path.join(DATASET_DIR, "metadata")


# Create folders if they don't exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)


# -----------------------------
# Image configuration
# -----------------------------

SEARCH_WIDTH = 1000
SEARCH_HEIGHT = 800

REFERENCE_WIDTH = 200
REFERENCE_HEIGHT = 150


# -----------------------------
# Generate synthetic DRAM pattern
# -----------------------------

def generate_dram_pattern(width, height):

    # Start with a slightly varying background
    image = np.random.randint(
        20,
        45,
        (height, width),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Horizontal word lines
    # --------------------------------------------------------
    for y in range(20, height, 40):

        # Small random intensity variation
        intensity = np.random.randint(150, 210)

        cv2.line(
            image,
            (0, y),
            (width, y),
            int(intensity),
            2
        )

    # --------------------------------------------------------
    # Vertical bit lines
    # --------------------------------------------------------
    for x in range(20, width, 50):

        intensity = np.random.randint(90, 160)

        cv2.line(
            image,
            (x, 0),
            (x, height),
            int(intensity),
            2
        )

    # --------------------------------------------------------
    # DRAM cell / contact structures
    # --------------------------------------------------------
    for y in range(20, height, 40):

        for x in range(20, width, 50):

            # Random radius
            radius = np.random.randint(3, 7)

            # Random brightness
            brightness = np.random.randint(180, 256)

            cv2.circle(
                image,
                (x, y),
                radius,
                int(brightness),
                -1
            )

    # --------------------------------------------------------
    # Add random small DRAM-like structures
    # --------------------------------------------------------
    for _ in range(250):

        x = np.random.randint(0, width)
        y = np.random.randint(0, height)

        length = np.random.randint(3, 15)

        intensity = np.random.randint(60, 180)

        cv2.line(
            image,
            (x, y),
            (
                min(x + length, width - 1),
                y
            ),
            int(intensity),
            1
        )

    # --------------------------------------------------------
    # Add small local variations
    # --------------------------------------------------------
    for _ in range(150):

        x = np.random.randint(0, width)
        y = np.random.randint(0, height)

        radius = np.random.randint(1, 3)

        brightness = np.random.randint(80, 220)

        cv2.circle(
            image,
            (x, y),
            radius,
            int(brightness),
            -1
        )

    return image

# -----------------------------
# Main dataset generation
# -----------------------------

def generate_pair(pair_id):

    # Generate large Search image
    search_image = generate_dram_pattern(
    SEARCH_WIDTH,
    SEARCH_HEIGHT
    )

# ------------------------------------------------------------
# Add subtle random variation
# ------------------------------------------------------------
    noise = np.random.randint(
    0,
    16,
    (SEARCH_HEIGHT, SEARCH_WIDTH),
    dtype=np.uint8
    )

    search_image = cv2.add(
    search_image,
    noise
    )


    # Select random location for Reference image
    max_x = SEARCH_WIDTH - REFERENCE_WIDTH
    max_y = SEARCH_HEIGHT - REFERENCE_HEIGHT

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    # Extract Reference image from Search image
    reference_image = search_image[
        y:y + REFERENCE_HEIGHT,
        x:x + REFERENCE_WIDTH
    ].copy()

    # Save images
    reference_path = os.path.join(
        TRAIN_DIR,
        f"pair_{pair_id:04d}_reference.png"
    )

    search_path = os.path.join(
        TRAIN_DIR,
        f"pair_{pair_id:04d}_search.png"
    )

    cv2.imwrite(reference_path, reference_image)
    cv2.imwrite(search_path, search_image)

    # Save ground truth information
    metadata_path = os.path.join(
        METADATA_DIR,
        f"pair_{pair_id:04d}.txt"
    )

    with open(metadata_path, "w") as file:
        file.write(f"pair_id: {pair_id}\n")
        file.write(f"architecture: DRAM\n")
        file.write(f"true_x: {x}\n")
        file.write(f"true_y: {y}\n")
        file.write(f"reference_width: {REFERENCE_WIDTH}\n")
        file.write(f"reference_height: {REFERENCE_HEIGHT}\n")

    print(
        f"Pair {pair_id}: "
        f"Reference location = ({x}, {y})"
    )


# -----------------------------
# Generate dataset
# -----------------------------

if __name__ == "__main__":

    NUMBER_OF_PAIRS = 30

    print("Generating synthetic DRAM dataset...")
    print()

    for pair_id in range(1, NUMBER_OF_PAIRS + 1):
        generate_pair(pair_id)

    print()
    print("Dataset generation completed!")
    print(f"Images saved in: {TRAIN_DIR}")
    print(f"Ground truth saved in: {METADATA_DIR}")