"""Ingest the three labellers' sheets into the real validation set.

Run:  python -m data.ingest_labels

Reads  data/labels_a.csv, labels_b.csv, labels_c.csv (returned to_label_X.csv sheets)
Writes data/real_validation.csv (string_id, raw_string, category, labeller_id)
       results/label_agreement.json (Fleiss' kappa etc. on the 30 anchors)

Rules (fixed in advance, see DECISIONS.md pre-registration):
- Any category not in src/config.CATEGORIES -> hard fail, listing the offending rows.
  Case/whitespace differences are tolerated; anything else is an error.
- Blank category -> hard fail (an unlabelled row is not a label).
- Anchors: majority vote (2 of 3). A 3-way split has no ground truth: it is logged
  to DECISIONS.md and EXCLUDED from real_validation.csv rather than guessed.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.stats import fleiss_kappa  # noqa: E402

LABELLERS = ["a", "b", "c"]
CANON = {c.strip().lower(): c for c in config.CATEGORY_NAMES}


def read_sheet(path: Path) -> pd.DataFrame:
    # Header lines start with '##'. Masked strings contain '#', so pandas' comment= can't be used.
    text = "\n".join(ln for ln in path.read_text(encoding="utf-8-sig").splitlines() if not ln.startswith("##"))
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


def main():
    frames, errors = [], []
    for lab in LABELLERS:
        path = config.DATA_DIR / f"labels_{lab}.csv"
        if not path.exists():
            print(f"SKIPPED: {path.name} not found. Real validation needs all three sheets.")
            return 0
        df = read_sheet(path)
        df["labeller_id"] = lab
        canon = df["category"].str.strip().str.lower().map(CANON)
        for _, r in df[canon.isna()].iterrows():
            errors.append(f"labels_{lab}.csv {r.string_id}: invalid category {r.category!r}")
        df["category"] = canon
        frames.append(df)
    if errors:
        print("HARD FAIL: unknown or blank categories. Fix these rows and re-run:")
        print("\n".join("  " + e for e in errors))
        print("Valid names: " + " | ".join(config.CATEGORY_NAMES))
        return 1

    allrows = pd.concat(frames, ignore_index=True)
    anchors = allrows[allrows["string_id"].str.startswith("ANC")]
    wide = anchors.pivot(index="string_id", columns="labeller_id", values="category")[LABELLERS]
    agreement = fleiss_kappa(wide.to_numpy(), config.CATEGORY_NAMES)

    resolved, ties = [], []
    raw_by_id = anchors.drop_duplicates("string_id").set_index("string_id")["raw_string"]
    for sid, row in wide.iterrows():
        vals, counts = np.unique(row.to_numpy(), return_counts=True)
        if counts.max() >= 2:
            resolved.append({"string_id": sid, "raw_string": raw_by_id[sid],
                             "category": vals[counts.argmax()], "labeller_id": "majority"})
        else:
            ties.append((sid, raw_by_id[sid], dict(row)))

    uniques = allrows[~allrows["string_id"].str.startswith("ANC")][["string_id", "raw_string", "category", "labeller_id"]]
    out = pd.concat([uniques, pd.DataFrame(resolved)], ignore_index=True).sort_values("string_id")
    out.to_csv(config.REAL_VALIDATION_CSV, index=False)

    agreement.update({"anchors_resolved_by_majority": len(resolved), "anchors_excluded_3way_tie": len(ties),
                      "n_real_validation_rows": int(len(out))})
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "label_agreement.json").write_text(json.dumps(agreement, indent=2))

    if ties:
        with open(config.ROOT / "DECISIONS.md", "a", encoding="utf-8") as f:
            f.write("\n### Anchor 3-way ties (auto-logged by data/ingest_labels.py)\n")
            f.write("No majority means no ground truth, so these are excluded from real_validation.csv.\n\n")
            for sid, s, labs in ties:
                f.write(f"- {sid} `{s}`: " + ", ".join(f"{k}={v}" for k, v in labs.items()) + "\n")

    print(f"Wrote {config.REAL_VALIDATION_CSV.name}: {len(out)} rows")
    print(f"Anchor agreement ({agreement['n_items']} strings x 3 labellers):")
    print(f"  Fleiss' kappa           = {agreement['fleiss_kappa']:.3f}")
    print(f"  mean pairwise agreement = {agreement['mean_pairwise_agreement']:.3f}")
    print(f"  unanimous               = {agreement['unanimous_agreement']:.3f}")
    print(f"  majority-resolved={len(resolved)}  3-way ties excluded={len(ties)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
