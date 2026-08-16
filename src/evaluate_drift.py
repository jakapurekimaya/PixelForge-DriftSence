import cv2
import os
import math


# ============================================================
# PixelForge DriftSense
# Phase 4 — Evaluate Localization on Drifted Images
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# ------------------------------------------------------------
# Dataset folders
# ------------------------------------------------------------

DRIFTED_DIR = os.path.join(
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


# ------------------------------------------------------------
# Store results
# ------------------------------------------------------------

results = []


# ============================================================
# Evaluate all 30 drifted images
# ============================================================

for pair_id in range(1, 31):

    pair_name = f"pair_{pair_id:04d}"

    drifted_path = os.path.join(
        DRIFTED_DIR,
        f"{pair_name}_search.png"
    )

    reference_path = os.path.join(
        REFERENCE_DIR,
        f"{pair_name}_reference.png"
    )

    metadata_path = os.path.join(
        METADATA_DIR,
        f"{pair_name}.txt"
    )


    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    drifted = cv2.imread(drifted_path)
    reference = cv2.imread(reference_path)

    if drifted is None:
        print(
            f"ERROR: Could not load {drifted_path}"
        )
        continue

    if reference is None:
        print(
            f"ERROR: Could not load {reference_path}"
        )
        continue


    # --------------------------------------------------------
    # Read ground truth
    # --------------------------------------------------------

    true_x = None
    true_y = None

    with open(metadata_path, "r") as f:

        for line in f:

            line = line.strip()

            if line.startswith("true_x:"):
                true_x = int(
                    line.split(":")[1].strip()
                )

            elif line.startswith("true_y:"):
                true_y = int(
                    line.split(":")[1].strip()
                )


    if true_x is None or true_y is None:

        print(
            f"ERROR: Ground truth missing for {pair_name}"
        )

        continue


    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    drifted_gray = cv2.cvtColor(
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

    match_result = cv2.matchTemplate(
        drifted_gray,
        reference_gray,
        cv2.TM_CCOEFF_NORMED
    )


    # --------------------------------------------------------
    # Find best match
    # --------------------------------------------------------

    _, match_score, _, max_location = cv2.minMaxLoc(
        match_result
    )

    predicted_x, predicted_y = max_location


    # --------------------------------------------------------
    # Calculate localization error
    # --------------------------------------------------------

    error_x = predicted_x - true_x
    error_y = predicted_y - true_y

    euclidean_error = math.sqrt(
        error_x ** 2 +
        error_y ** 2
    )


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append({

        "pair": pair_id,

        "true_x": true_x,
        "true_y": true_y,

        "predicted_x": predicted_x,
        "predicted_y": predicted_y,

        "error": euclidean_error,

        "score": match_score
    })


# ============================================================
# Print individual results
# ============================================================

print()
print("=" * 80)
print("PixelForge DriftSense - DRIFT EVALUATION")
print("=" * 80)

print()

print(
    f"{'Pair':<6}"
    f"{'Ground Truth':<20}"
    f"{'Predicted':<20}"
    f"{'Error(px)':<12}"
    f"{'Score':<10}"
)

print("-" * 80)


for result in results:

    ground_truth = (
        f"({result['true_x']}, "
        f"{result['true_y']})"
    )

    predicted = (
        f"({result['predicted_x']}, "
        f"{result['predicted_y']})"
    )

    print(
        f"{result['pair']:<6}"
        f"{ground_truth:<20}"
        f"{predicted:<20}"
        f"{result['error']:<12.2f}"
        f"{result['score']:<10.4f}"
    )


# ============================================================
# Overall statistics
# ============================================================

if len(results) > 0:

    errors = [
        result["error"]
        for result in results
    ]

    scores = [
        result["score"]
        for result in results
    ]


    mean_error = (
        sum(errors) / len(errors)
    )

    max_error = max(errors)

    min_error = min(errors)

    average_score = (
        sum(scores) / len(scores)
    )


    # Exact matches

    exact_matches = sum(
        1
        for error in errors
        if error == 0
    )


    accuracy = (
        exact_matches /
        len(results)
    ) * 100


    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("DRIFT EVALUATION RESULTS")
    print("=" * 80)

    print(
        f"Pairs evaluated       : {len(results)}"
    )

    print(
        f"Exact matches         : {exact_matches}"
    )

    print(
        f"Localization accuracy : {accuracy:.2f}%"
    )

    print(
        f"Mean error            : {mean_error:.2f} pixels"
    )

    print(
        f"Minimum error         : {min_error:.2f} pixels"
    )

    print(
        f"Maximum error         : {max_error:.2f} pixels"
    )

    print(
        f"Average match score   : {average_score:.4f}"
    )

    print("=" * 80)

else:

    print()
    print("No drift results were generated.")