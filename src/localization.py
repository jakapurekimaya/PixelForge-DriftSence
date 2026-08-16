import cv2
import os


# ============================================================
# PixelForge DriftSense — Phase 2
# Baseline Localization using Template Matching
# ============================================================

# Project folders
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEARCH_IMAGE = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "train",
    "pair_0001_search.png"
)

REFERENCE_IMAGE = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "train",
    "pair_0001_reference.png"
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUTPUT_IMAGE = os.path.join(
    OUTPUT_DIR,
    "pair_0001_localized.png"
)


# ------------------------------------------------------------
# Ground-truth location from metadata
# ------------------------------------------------------------
METADATA_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "metadata",
    "pair_0001.txt"
)

TRUE_X = None
TRUE_Y = None

with open(METADATA_FILE, "r") as f:
    for line in f:
        line = line.strip()

        if line.startswith("true_x:"):
            TRUE_X = int(line.split(":")[1].strip())

        elif line.startswith("true_y:"):
            TRUE_Y = int(line.split(":")[1].strip())

if TRUE_X is None or TRUE_Y is None:
    raise ValueError(
        "Could not find true_x and true_y in metadata."
    )


# ------------------------------------------------------------
# Create output folder if it doesn't exist
# ------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Load images
# ------------------------------------------------------------
search = cv2.imread(SEARCH_IMAGE)
reference = cv2.imread(REFERENCE_IMAGE)

if search is None:
    raise FileNotFoundError(
        f"Could not load search image:\n{SEARCH_IMAGE}"
    )

if reference is None:
    raise FileNotFoundError(
        f"Could not load reference image:\n{REFERENCE_IMAGE}"
    )


# ------------------------------------------------------------
# Convert images to grayscale
# ------------------------------------------------------------
search_gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)


# ------------------------------------------------------------
# Template matching
# ------------------------------------------------------------
result = cv2.matchTemplate(
    search_gray,
    reference_gray,
    cv2.TM_CCOEFF_NORMED
)


# Find the best matching location
_, max_score, _, max_location = cv2.minMaxLoc(result)

predicted_x, predicted_y = max_location


# ------------------------------------------------------------
# Get reference dimensions
# ------------------------------------------------------------
reference_height, reference_width = reference_gray.shape


# ------------------------------------------------------------
# Calculate localization error
# ------------------------------------------------------------
error_x = predicted_x - TRUE_X
error_y = predicted_y - TRUE_Y

absolute_error_x = abs(error_x)
absolute_error_y = abs(error_y)

euclidean_error = (
    (error_x ** 2 + error_y ** 2) ** 0.5
)


# ------------------------------------------------------------
# Draw detected bounding box
# ------------------------------------------------------------
top_left = (predicted_x, predicted_y)

bottom_right = (
    predicted_x + reference_width,
    predicted_y + reference_height
)

cv2.rectangle(
    search,
    top_left,
    bottom_right,
    (0, 255, 0),
    2
)


# ------------------------------------------------------------
# Add information to image
# ------------------------------------------------------------
cv2.putText(
    search,
    f"Predicted: ({predicted_x}, {predicted_y})",
    (20, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

cv2.putText(
    search,
    f"Ground Truth: ({TRUE_X}, {TRUE_Y})",
    (20, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 0, 0),
    2
)

cv2.putText(
    search,
    f"Match Score: {max_score:.4f}",
    (20, 90),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 255),
    2
)

cv2.putText(
    search,
    f"Localization Error: {euclidean_error:.2f} px",
    (20, 120),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 255),
    2
)


# ------------------------------------------------------------
# Save result
# ------------------------------------------------------------
cv2.imwrite(OUTPUT_IMAGE, search)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------
print()
print("=" * 55)
print("PixelForge DriftSense - Localization Result")
print("=" * 55)

print(f"Reference image : pair_0001_reference.png")
print(f"Search image    : pair_0001_search.png")

print()
print(f"Ground truth    : ({TRUE_X}, {TRUE_Y})")
print(f"Predicted       : ({predicted_x}, {predicted_y})")

print()
print(f"X error         : {absolute_error_x} pixels")
print(f"Y error         : {absolute_error_y} pixels")
print(f"Euclidean error : {euclidean_error:.2f} pixels")

print()
print(f"Match score     : {max_score:.4f}")

print()
print(f"Output saved to : {OUTPUT_IMAGE}")

print("=" * 55)
print("Localization completed!")
print("=" * 55)