import cv2
import os


# ============================================================
# PixelForge DriftSense
# Visual Drift Report — Pair 0001
# ============================================================


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------

ORIGINAL_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "train",
    "pair_0001_search.png"
)

DRIFTED_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "drifted_strong",
    "pair_0001_search.png"
)

REFERENCE_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "train",
    "pair_0001_reference.png"
)

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "metadata",
    "pair_0001.txt"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "pair_0001_drift_report.png"
)


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Load images
# ------------------------------------------------------------

original = cv2.imread(ORIGINAL_PATH)
drifted = cv2.imread(DRIFTED_PATH)
reference = cv2.imread(REFERENCE_PATH)


if original is None:
    raise FileNotFoundError(
        "Original image could not be loaded."
    )

if drifted is None:
    raise FileNotFoundError(
        "Drifted image could not be loaded."
    )

if reference is None:
    raise FileNotFoundError(
        "Reference image could not be loaded."
    )


# ------------------------------------------------------------
# Read ground truth
# ------------------------------------------------------------

true_x = None
true_y = None

with open(METADATA_PATH, "r") as file:

    for line in file:

        line = line.strip()

        if line.startswith("true_x:"):
            true_x = int(
                line.split(":")[1].strip()
            )

        elif line.startswith("true_y:"):
            true_y = int(
                line.split(":")[1].strip()
            )


# ------------------------------------------------------------
# Convert to grayscale
# ------------------------------------------------------------

original_gray = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2GRAY
)

drifted_gray = cv2.cvtColor(
    drifted,
    cv2.COLOR_BGR2GRAY
)

reference_gray = cv2.cvtColor(
    reference,
    cv2.COLOR_BGR2GRAY
)


# ------------------------------------------------------------
# Locate reference in original image
# ------------------------------------------------------------

original_result = cv2.matchTemplate(
    original_gray,
    reference_gray,
    cv2.TM_CCOEFF_NORMED
)

_, original_score, _, original_location = cv2.minMaxLoc(
    original_result
)


# ------------------------------------------------------------
# Locate reference in drifted image
# ------------------------------------------------------------

drifted_result = cv2.matchTemplate(
    drifted_gray,
    reference_gray,
    cv2.TM_CCOEFF_NORMED
)

_, drifted_score, _, drifted_location = cv2.minMaxLoc(
    drifted_result
)


# ------------------------------------------------------------
# Predicted location
# ------------------------------------------------------------

predicted_x, predicted_y = drifted_location


# ------------------------------------------------------------
# Calculate drift
# ------------------------------------------------------------

score_drop = (
    original_score -
    drifted_score
)

drift_percentage = (
    score_drop /
    original_score
) * 100


# ------------------------------------------------------------
# Severity
# ------------------------------------------------------------

if drift_percentage < 2:

    severity = "LOW"

elif drift_percentage < 8:

    severity = "MEDIUM"

else:

    severity = "HIGH"


# ------------------------------------------------------------
# Localization error
# ------------------------------------------------------------

error_x = predicted_x - true_x
error_y = predicted_y - true_y

localization_error = (
    error_x ** 2 +
    error_y ** 2
) ** 0.5


# ------------------------------------------------------------
# Draw bounding boxes
# ------------------------------------------------------------

reference_height, reference_width = (
    reference_gray.shape
)


original_display = original.copy()
drifted_display = drifted.copy()


original_bottom_right = (
    original_location[0] + reference_width,
    original_location[1] + reference_height
)

drifted_bottom_right = (
    predicted_x + reference_width,
    predicted_y + reference_height
)


cv2.rectangle(
    original_display,
    original_location,
    original_bottom_right,
    (0, 255, 0),
    3
)

cv2.rectangle(
    drifted_display,
    drifted_location,
    drifted_bottom_right,
    (0, 255, 0),
    3
)


# ------------------------------------------------------------
# Add labels
# ------------------------------------------------------------

cv2.putText(
    original_display,
    "ORIGINAL",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.putText(
    drifted_display,
    "DRIFTED",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)


# ------------------------------------------------------------
# Resize images
# ------------------------------------------------------------

height = 500

original_ratio = (
    height / original_display.shape[0]
)

drifted_ratio = (
    height / drifted_display.shape[0]
)


original_display = cv2.resize(
    original_display,
    None,
    fx=original_ratio,
    fy=original_ratio
)

drifted_display = cv2.resize(
    drifted_display,
    None,
    fx=drifted_ratio,
    fy=drifted_ratio
)


# ------------------------------------------------------------
# Make same width
# ------------------------------------------------------------

display_width = max(
    original_display.shape[1],
    drifted_display.shape[1]
)


def pad_image(image, width):

    if image.shape[1] < width:

        padding = width - image.shape[1]

        image = cv2.copyMakeBorder(
            image,
            0,
            0,
            0,
            padding,
            cv2.BORDER_CONSTANT,
            value=(30, 30, 30)
        )

    return image


original_display = pad_image(
    original_display,
    display_width
)

drifted_display = pad_image(
    drifted_display,
    display_width
)


# ------------------------------------------------------------
# Create information panel
# ------------------------------------------------------------

panel_height = 230

panel = (
    30 *
    __import__("numpy").ones(
        (panel_height, display_width, 3),
        dtype="uint8"
    )
)


# ------------------------------------------------------------
# Add report information
# ------------------------------------------------------------

cv2.putText(
    panel,
    "PixelForge DriftSense - DRIFT REPORT",
    (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (255, 255, 255),
    2
)

cv2.putText(
    panel,
    f"Original Match Score : {original_score:.4f}",
    (20, 75),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255, 255, 255),
    2
)

cv2.putText(
    panel,
    f"Drifted Match Score  : {drifted_score:.4f}",
    (20, 105),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255, 255, 255),
    2
)

cv2.putText(
    panel,
    f"Drift                : {drift_percentage:.2f}%",
    (20, 135),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (0, 255, 255),
    2
)

cv2.putText(
    panel,
    f"Localization         : ({predicted_x}, {predicted_y})",
    (20, 165),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255, 255, 255),
    2
)

cv2.putText(
    panel,
    f"Localization Error   : {localization_error:.2f} px",
    (20, 195),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255, 255, 255),
    2
)


# ------------------------------------------------------------
# Severity indicator
# ------------------------------------------------------------

cv2.putText(
    panel,
    f"SEVERITY: {severity}",
    (display_width - 300, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 0, 255),
    3
)

cv2.putText(
    panel,
    "STATUS: DRIFT DETECTED",
    (display_width - 390, 130),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (0, 0, 255),
    2
)


# ------------------------------------------------------------
# Combine everything
# ------------------------------------------------------------

image_comparison = cv2.hconcat([
    original_display,
    drifted_display
])


# ------------------------------------------------------------
# Prepare images for vertical concatenation
# ------------------------------------------------------------

# Make sure both images have 3 color channels
if len(image_comparison.shape) == 2:
    image_comparison = cv2.cvtColor(
        image_comparison,
        cv2.COLOR_GRAY2BGR
    )

if len(panel.shape) == 2:
    panel = cv2.cvtColor(
        panel,
        cv2.COLOR_GRAY2BGR
    )


# Make panel width same as image comparison
target_width = image_comparison.shape[1]

panel_height = int(
    panel.shape[0] *
    target_width /
    panel.shape[1]
)

panel = cv2.resize(
    panel,
    (target_width, panel_height)
)


# Now both images have same width and type
final_report = cv2.vconcat([
    image_comparison,
    panel
])


# ------------------------------------------------------------
# Save final report
# ------------------------------------------------------------

cv2.imwrite(
    OUTPUT_PATH,
    final_report
)


# ------------------------------------------------------------
# Terminal result
# ------------------------------------------------------------

print()
print("=" * 65)
print("PixelForge DriftSense - Visual Report")
print("=" * 65)

print(
    f"Original score     : {original_score:.4f}"
)

print(
    f"Drifted score      : {drifted_score:.4f}"
)

print(
    f"Drift percentage   : {drift_percentage:.2f}%"
)

print(
    f"Localization       : "
    f"({predicted_x}, {predicted_y})"
)

print(
    f"Localization error : "
    f"{localization_error:.2f} px"
)

print(
    f"Severity            : {severity}"
)

print(
    f"Status              : DRIFT DETECTED"
)

print()
print(
    f"Report saved to     : {OUTPUT_PATH}"
)

print("=" * 65)