import cv2
import os
import numpy as np


# ============================================================
# PixelForge DriftSense
# Phase 5 — Drift Detection and Severity Classification
# ============================================================


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


REFERENCE_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "train"
)


DRIFT_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "drifted_strong"
)


# ============================================================
# Drift detection settings
# ============================================================

DRIFT_THRESHOLD = 0.05


# ============================================================
# Storage
# ============================================================

results = []


# ============================================================
# Process all 30 image pairs
# ============================================================

for i in range(1, 31):

    filename = f"pair_{i:04d}"

    reference_path = os.path.join(
        REFERENCE_DIR,
        filename + "_reference.png"
    )

    original_search_path = os.path.join(
        REFERENCE_DIR,
        filename + "_search.png"
    )

    drifted_search_path = os.path.join(
        DRIFT_DIR,
        filename + "_search.png"
    )


    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    reference = cv2.imread(reference_path)
    original_search = cv2.imread(original_search_path)
    drifted_search = cv2.imread(drifted_search_path)


    if (
        reference is None or
        original_search is None or
        drifted_search is None
    ):

        print(
            f"Skipping Pair {i}: image not found"
        )

        continue


    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    reference_gray = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY
    )

    original_gray = cv2.cvtColor(
        original_search,
        cv2.COLOR_BGR2GRAY
    )

    drifted_gray = cv2.cvtColor(
        drifted_search,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Match reference with original search image
    # --------------------------------------------------------

    original_result = cv2.matchTemplate(
        original_gray,
        reference_gray,
        cv2.TM_CCOEFF_NORMED
    )

    _, original_score, _, _ = cv2.minMaxLoc(
        original_result
    )


    # --------------------------------------------------------
    # Match reference with drifted search image
    # --------------------------------------------------------

    drifted_result = cv2.matchTemplate(
        drifted_gray,
        reference_gray,
        cv2.TM_CCOEFF_NORMED
    )

    _, drifted_score, _, _ = cv2.minMaxLoc(
        drifted_result
    )


    # --------------------------------------------------------
    # Calculate degradation
    # --------------------------------------------------------

    score_drop = (
        original_score -
        drifted_score
    )


    drift_percentage = (
        score_drop /
        original_score
    ) * 100


    # --------------------------------------------------------
    # Classify drift severity
    # --------------------------------------------------------

    if drift_percentage < 2:

        severity = "LOW"

    elif drift_percentage < 8:

        severity = "MEDIUM"

    else:

        severity = "HIGH"


    drift_detected = (
        score_drop >
        DRIFT_THRESHOLD
    )


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append({

        "pair": i,

        "original_score":
            original_score,

        "drifted_score":
            drifted_score,

        "score_drop":
            score_drop,

        "drift_percentage":
            drift_percentage,

        "drift_detected":
            drift_detected,

        "severity":
            severity
    })


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        f"Pair {i:02d} | "
        f"Original={original_score:.4f} | "
        f"Drifted={drifted_score:.4f} | "
        f"Drop={score_drop:.4f} | "
        f"Drift={drift_percentage:.2f}% | "
        f"Severity={severity}"
    )


# ============================================================
# Overall results
# ============================================================

if results:

    average_original = np.mean([
        r["original_score"]
        for r in results
    ])


    average_drifted = np.mean([
        r["drifted_score"]
        for r in results
    ])


    average_drop = (
        average_original -
        average_drifted
    )


    average_drift_percentage = (
        average_drop /
        average_original
    ) * 100


    detected_count = sum(
        r["drift_detected"]
        for r in results
    )


    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()

    print("=" * 70)
    print("PixelForge DriftSense")
    print("DRIFT DETECTION REPORT")
    print("=" * 70)

    print(
        f"Pairs evaluated       : {len(results)}"
    )

    print(
        f"Original avg score    : "
        f"{average_original:.4f}"
    )

    print(
        f"Drifted avg score     : "
        f"{average_drifted:.4f}"
    )

    print(
        f"Average score drop    : "
        f"{average_drop:.4f}"
    )

    print(
        f"Average drift         : "
        f"{average_drift_percentage:.2f}%"
    )

    print(
        f"Drift detected        : "
        f"{detected_count}/{len(results)}"
    )

    print()

    if average_drift_percentage < 2:

        final_severity = "LOW"

    elif average_drift_percentage < 8:

        final_severity = "MEDIUM"

    else:

        final_severity = "HIGH"


    print(
        f"Overall drift severity: "
        f"{final_severity}"
    )

    print("=" * 70)


else:

    print(
        "No results were generated."
    )