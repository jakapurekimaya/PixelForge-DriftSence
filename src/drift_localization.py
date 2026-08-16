import cv2
import os
import numpy as np


# ============================================================
# PixelForge DriftSense — Phase 3
# Localization on Drifted Images
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DRIFT_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "drifted_strong"
)

REFERENCE_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "train"
)

METADATA_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "metadata"
)


print()
print("=" * 65)
print("PixelForge DriftSense - Phase 3")
print("Localization on Drifted Images")
print("=" * 65)


results = []


# ------------------------------------------------------------
# Process all 30 pairs
# ------------------------------------------------------------

for i in range(1, 31):

    filename = f"pair_{i:04d}"

    drift_path = os.path.join(
        DRIFT_DIR,
        filename + "_search.png"
    )

    reference_path = os.path.join(
        REFERENCE_DIR,
        filename + "_reference.png"
    )

    metadata_path = os.path.join(
        METADATA_DIR,
        filename + ".txt"
    )


    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    drifted = cv2.imread(drift_path)
    reference = cv2.imread(reference_path)

    if drifted is None or reference is None:
        print(f"Skipping Pair {i}: image not found")
        continue


    # --------------------------------------------------------
    # Read ground truth
    # --------------------------------------------------------

    true_x = None
    true_y = None

    with open(metadata_path, "r") as file:

        for line in file:

            if line.startswith("true_x:"):
                true_x = int(line.split(":")[1].strip())

            elif line.startswith("true_y:"):
                true_y = int(line.split(":")[1].strip())


    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    drift_gray = cv2.cvtColor(
        drifted,
        cv2.COLOR_BGR2GRAY
    )

    reference_gray = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Template matching
    # --------------------------------------------------------

    result = cv2.matchTemplate(
        drift_gray,
        reference_gray,
        cv2.TM_CCOEFF_NORMED
    )


    # Find best match

    _, max_score, _, max_location = cv2.minMaxLoc(
        result
    )


    predicted_x, predicted_y = max_location


    # --------------------------------------------------------
    # Calculate error
    # --------------------------------------------------------

    error_x = predicted_x - true_x
    error_y = predicted_y - true_y

    euclidean_error = (
        (error_x ** 2 + error_y ** 2) ** 0.5
    )


    exact_match = (
        predicted_x == true_x and
        predicted_y == true_y
    )


    results.append({
        "pair": i,
        "true_x": true_x,
        "true_y": true_y,
        "predicted_x": predicted_x,
        "predicted_y": predicted_y,
        "error": euclidean_error,
        "score": max_score,
        "exact": exact_match
    })


    print(
        f"Pair {i:02d} | "
        f"GT=({true_x},{true_y}) | "
        f"Pred=({predicted_x},{predicted_y}) | "
        f"Error={euclidean_error:.2f}px | "
        f"Score={max_score:.4f}"
    )


# ------------------------------------------------------------
# Overall evaluation
# ------------------------------------------------------------

total = len(results)

exact_matches = sum(
    r["exact"] for r in results
)

errors = [
    r["error"] for r in results
]

scores = [
    r["score"] for r in results
]


accuracy = (
    exact_matches / total * 100
    if total > 0 else 0
)

mean_error = (
    np.mean(errors)
    if errors else 0
)

minimum_error = (
    np.min(errors)
    if errors else 0
)

maximum_error = (
    np.max(errors)
    if errors else 0
)

average_score = (
    np.mean(scores)
    if scores else 0
)


print()
print("=" * 65)
print("OVERALL RESULTS")
print("=" * 65)

print(f"Pairs evaluated      : {total}")
print(f"Exact matches        : {exact_matches}")
print(f"Localization accuracy: {accuracy:.2f}%")
print(f"Mean error           : {mean_error:.2f} pixels")
print(f"Minimum error        : {minimum_error:.2f} pixels")
print(f"Maximum error        : {maximum_error:.2f} pixels")
print(f"Average match score  : {average_score:.4f}")

print("=" * 65)