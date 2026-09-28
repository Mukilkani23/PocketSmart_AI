"""Build the human labelling sheets for the REAL validation set.

Run:  python -m data.labelling_sheet

Input : data/real_strings_raw.txt: one real bank-statement merchant string per line
        (pasted by Kani from real statements). Gitignored. Never committed.
Output: data/to_label_a.csv, to_label_b.csv, to_label_c.csv: one per labeller,
        130 rows each = 100 unique strings + the same 30 ANCHOR strings.
        data/to_label.csv: master list (330 unique strings, who labels what).

Anchors (string_id ANC01..ANC30) are labelled by all three people. That overlap is what
lets us measure human agreement (Fleiss' kappa), i.e. the ceiling for the model.

PRIVACY: runs of >=4 digits (phone numbers, UPI refs, account numbers) are masked to
'#' BEFORE anything is written. The model applies the same mask to every string, so
nothing is lost for classification.

Labellers: open your to_label_X.csv, fill the `category` column with EXACTLY one of
the 10 names in the header, save as labels_X.csv in data/. Don't edit other columns.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.features import mask_digits  # noqa: E402

N_UNIQUE_PER = 100
N_ANCHORS = 30
LABELLERS = ["a", "b", "c"]


def header_lines() -> list[str]:
    lines = ["## PocketSmart labelling sheet. Fill the `category` column with EXACTLY one of these 10 names:"]
    lines += [f"## {name} = {desc}" for name, desc in config.CATEGORIES.items()]
    lines += ["## If unsure, pick your best guess. Do not invent new categories. Lines starting ## are ignored."]
    return lines


def load_strings() -> tuple[list[str], str]:
    if config.REAL_STRINGS_RAW.exists():
        raw = [ln.strip() for ln in config.REAL_STRINGS_RAW.read_text(encoding="utf-8").splitlines()]
        strings = list(dict.fromkeys(mask_digits(s) for s in raw if s))  # dedupe, keep order
        return strings, "real"
    print("\n" + "!" * 78)
    print("!! WARNING: data/real_strings_raw.txt NOT FOUND.")
    print("!! Building the sheet from HELD-OUT SYNTHETIC strings so the pipeline is testable.")
    print("!! These sheets are NOT a real validation set. Paste real strings and re-run.")
    print("!" * 78 + "\n")
    df = pd.read_csv(config.TRANSACTIONS_CSV)
    cutoff = df["date"].sort_values().iloc[int(len(df) * (1 - config.TEST_FRACTION))]
    test = df[df["date"] >= cutoff]
    strings = list(dict.fromkeys(mask_digits(s) for s in test["merchant_raw"]))
    return strings, "synthetic_fallback"


def main():
    strings, source = load_strings()
    rng = np.random.default_rng(config.SEED)
    strings = [strings[i] for i in rng.permutation(len(strings))]
    need = N_ANCHORS + N_UNIQUE_PER * len(LABELLERS)
    if len(strings) < need:
        print(f"WARNING: only {len(strings)} unique strings (want {need}). Using all of them; "
              f"anchors stay at {min(N_ANCHORS, len(strings))}.")
    anchors = strings[:N_ANCHORS]
    rest = strings[N_ANCHORS:need]
    per = int(np.ceil(len(rest) / len(LABELLERS)))

    master = [{"string_id": f"ANC{i + 1:02d}", "raw_string": s, "labeller_id": "abc"} for i, s in enumerate(anchors)]
    for k, lab in enumerate(LABELLERS):
        chunk = rest[k * per:(k + 1) * per]
        master += [{"string_id": f"L{lab.upper()}{i + 1:03d}", "raw_string": s, "labeller_id": lab}
                   for i, s in enumerate(chunk)]
    master_df = pd.DataFrame(master)
    master_df["source"] = source

    config.DATA_DIR.mkdir(exist_ok=True)
    master_df.to_csv(config.DATA_DIR / "to_label.csv", index=False)
    for lab in LABELLERS:
        sheet = master_df[master_df["labeller_id"].isin(["abc", lab])][["string_id", "raw_string"]].copy()
        # anchors are shuffled in among the unique rows so labellers can't treat them differently
        sheet = sheet.sample(frac=1, random_state=config.SEED + ord(lab)).reset_index(drop=True)
        sheet["category"] = ""
        path = config.DATA_DIR / f"to_label_{lab}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(header_lines()) + "\n")
            sheet.to_csv(f, index=False)
        print(f"Wrote {path.name}: {len(sheet)} rows ({(sheet.string_id.str.startswith('ANC')).sum()} anchors)")
    print(f"Wrote to_label.csv: {len(master_df)} unique strings, source={source}")
    print("\nNext: send to_label_a/b/c.csv to the three labellers; they return labels_a/b/c.csv,")
    print("then run:  python -m data.ingest_labels")


if __name__ == "__main__":
    main()
