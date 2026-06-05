#!/usr/bin/env python3
"""
Inter-Annotator Agreement (IAA) analysis for the cultural-entity validation study.

Reads the 13 batch CSV files (output_batch_1.csv ... output_batch_13.csv),
takes the first 10 questions from each (= 130 sampled questions), extracts the
two human annotators' Yes/No judgments, and computes:

  - the 2x2 contingency table
  - observed (percentage) agreement
  - Cohen's kappa (computed manually AND cross-checked with scikit-learn if available)
  - each annotator's marginal totals / accuracy
  - a Landis & Koch interpretation band

Outputs:
  - prints a full report to the screen
  - writes combined_annotations.csv  (the merged per-question labels = your proof)
  - writes iaa_results.txt            (the summary report)

"""

import csv
import os

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))   # folder this script lives in
NUM_BATCHES = 13                                     # output_batch_1 ... output_batch_13
QUESTIONS_PER_BATCH = 10                             # we validated the first 10 of each batch

# The annotator columns are at positions 4 and 5 in every file
# (the header NAMES are inconsistent across files, so we use position, not name).
ANNOTATOR_1_COL = 4
ANNOTATOR_2_COL = 5


def norm(value):
    """Normalise a raw cell to 'yes' / 'no' / None."""
    v = (value or "").strip().lower()
    if v in ("yes", "y", "1", "true"):
        return "yes"
    if v in ("no", "n", "0", "false"):
        return "no"
    return None  # blank or unexpected


def load_labels():
    """Return three parallel lists: question_ids, annotator1, annotator2."""
    qids, a1, a2 = [], [], []
    skipped = []

    for i in range(1, NUM_BATCHES + 1):
        path = os.path.join(HERE, f"output_batch_{i}.csv")
        # utf-8-sig strips any BOM that Excel sometimes adds
        with open(path, encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))

        # rows[0] is the header; take the first QUESTIONS_PER_BATCH data rows
        for row in rows[1 : 1 + QUESTIONS_PER_BATCH]:
            v1 = norm(row[ANNOTATOR_1_COL]) if len(row) > ANNOTATOR_1_COL else None
            v2 = norm(row[ANNOTATOR_2_COL]) if len(row) > ANNOTATOR_2_COL else None
            qid = f"batch{i}_q{row[0]}" if row else f"batch{i}_?"

            if v1 is None or v2 is None:
                skipped.append((qid, row[ANNOTATOR_1_COL:ANNOTATOR_2_COL + 1] if len(row) > ANNOTATOR_2_COL else row))
                continue

            qids.append(qid)
            a1.append(v1)
            a2.append(v2)

    return qids, a1, a2, skipped


def cohens_kappa(a1, a2):
    """Compute Cohen's kappa for two lists of 'yes'/'no' labels (manual formula)."""
    n = len(a1)

    both_yes = sum(1 for x, y in zip(a1, a2) if x == "yes" and y == "yes")
    both_no  = sum(1 for x, y in zip(a1, a2) if x == "no"  and y == "no")
    a1y_a2n  = sum(1 for x, y in zip(a1, a2) if x == "yes" and y == "no")
    a1n_a2y  = sum(1 for x, y in zip(a1, a2) if x == "no"  and y == "yes")

    # Observed agreement
    po = (both_yes + both_no) / n

    # Expected agreement by chance
    p1_yes = a1.count("yes") / n
    p2_yes = a2.count("yes") / n
    p1_no  = a1.count("no")  / n
    p2_no  = a2.count("no")  / n
    pe = (p1_yes * p2_yes) + (p1_no * p2_no)

    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else float("nan")

    table = {
        "both_yes": both_yes,
        "both_no": both_no,
        "a1yes_a2no": a1y_a2n,
        "a1no_a2yes": a1n_a2y,
    }
    return po, pe, kappa, table


def landis_koch(kappa):
    """Standard interpretation bands (Landis & Koch, 1977)."""
    if kappa < 0.00:  return "poor (less than chance)"
    if kappa < 0.20:  return "slight"
    if kappa < 0.40:  return "fair"
    if kappa < 0.60:  return "moderate"
    if kappa < 0.80:  return "substantial"
    return "almost perfect"


def main():
    qids, a1, a2, skipped = load_labels()
    n = len(a1)

    po, pe, kappa, t = cohens_kappa(a1, a2)

    # --- cross-check with scikit-learn if it's installed ---
    sklearn_kappa = None
    try:
        from sklearn.metrics import cohen_kappa_score
        sklearn_kappa = cohen_kappa_score(a1, a2)
    except Exception:
        pass

    lines = []
    lines.append("=" * 60)
    lines.append("INTER-ANNOTATOR AGREEMENT REPORT")
    lines.append("=" * 60)
    lines.append(f"Questions scored by BOTH annotators: {n}")
    if skipped:
        lines.append(f"Rows skipped (blank/invalid label): {len(skipped)}")
    lines.append("")
    lines.append("2x2 contingency table:")
    lines.append("                 A2 = Yes    A2 = No")
    lines.append(f"   A1 = Yes        {t['both_yes']:>4}       {t['a1yes_a2no']:>4}")
    lines.append(f"   A1 = No         {t['a1no_a2yes']:>4}       {t['both_no']:>4}")
    lines.append("")
    lines.append(f"Annotator 1: Yes={a1.count('yes')}  No={a1.count('no')}  "
                 f"(acc = {a1.count('yes')/n*100:.1f}%)")
    lines.append(f"Annotator 2: Yes={a2.count('yes')}  No={a2.count('no')}  "
                 f"(acc = {a2.count('yes')/n*100:.1f}%)")
    lines.append("")
    lines.append(f"Observed (percentage) agreement  Po = "
                 f"{t['both_yes'] + t['both_no']}/{n} = {po*100:.1f}%")
    lines.append(f"Expected-by-chance agreement     Pe = {pe:.4f}")
    lines.append(f"Cohen's kappa (manual)              = {kappa:.4f}")
    if sklearn_kappa is not None:
        lines.append(f"Cohen's kappa (scikit-learn check)  = {sklearn_kappa:.4f}")
    lines.append(f"Interpretation (Landis & Koch 1977) = {landis_koch(kappa)}")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    # --- write proof artifacts ---
    with open(os.path.join(HERE, "combined_annotations.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "annotator_1", "annotator_2", "agree"])
        for q, x, y in zip(qids, a1, a2):
            w.writerow([q, x, y, "yes" if x == y else "no"])

    with open(os.path.join(HERE, "iaa_results.txt"), "w") as f:
        f.write(report + "\n")

    print("\nSaved: combined_annotations.csv  (per-question merged labels)")
    print("Saved: iaa_results.txt           (this summary)")


if __name__ == "__main__":
    main()
