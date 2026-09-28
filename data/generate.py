"""Synthetic generator: 24 months of messy Indian bank-statement transactions.

Run:  python -m data.generate

Output: data/transactions.csv with columns
    txn_id, date, merchant_raw, amount, direction, category, is_anomaly

Design notes (defended in VIVA.md / DECISIONS.md):
- The label must NOT be recoverable from the string format alone. Every category
  mostly pays via the same UPI formats, and several merchants are shared across
  categories (Amazon Pay, Paytm, PhonePe, Flipkart, Swiggy, Tata, Reliance...).
- A share of every category is paid to *individuals* (kirana owner, auto driver,
  tuition teacher, landlord) whose names come from ONE shared pool. Those rows are
  genuinely ambiguous from text alone; they set the ceiling on accuracy.
- Merchants appear under brand names AND legal entity names (Swiggy = BUNDL
  TECHNOLOGIES, Ola = ANI TECHNOLOGIES), with truncation, casing and junk noise.
- `is_anomaly` is ground truth for evaluating the detector. It is never a model input.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

# --------------------------------------------------------------------------
# Ambiguity knobs. These are the ONLY things tuned when the leakage gate fires.
# Every change is logged in DECISIONS.md with the accuracy it produced.
# --------------------------------------------------------------------------
PERSON_SHARE = {  # fraction of a category's txns paid to an individual (shared name pool)
    "Food & Dining": 0.05, "Groceries": 0.10, "Transport": 0.10, "Shopping": 0.03,
    "Bills & Utilities": 0.04, "Entertainment": 0.02, "Health": 0.06,
    "Education": 0.10, "Transfers": 0.85, "Miscellaneous": 0.12,
}
TRUNCATE_P = 0.20       # probability a string is cut to 18-32 chars
ANOMALY_RATE = 0.02

# Shared platforms: (aliases) — reused across categories below.
AMAZON = ["AMAZON PAY INDIA", "AMAZONPAY", "AMZN PAY", "AMAZON PAY IN", "AMAZON SELLER SERVICES"]
FLIPKART = ["FLIPKART", "FLIPKART INTERNET", "FKRT", "FLIPKART PAYMENTS"]
PAYTM = ["PAYTM", "ONE97 COMMUNICATIONS", "PAYTM PAYMENTS", "PAYTMQR"]
PHONEPE = ["PHONEPE", "PHONEPE PVT LTD", "PHONEPE RECHARGE"]
SWIGGY = ["SWIGGY", "SWIGY", "BUNDL TECHNOLOGIES", "SWIGGY LTD"]
ZOMATO = ["ZOMATO", "ZOMATO LTD", "ETERNAL LTD", "ZOMATO ONLINE"]
TATA = ["TATA", "TATA DIGITAL", "TATA NEU"]
RELIANCE = ["RELIANCE", "RELIANCE RETAIL", "RELIANCE SMART"]
GPAY = ["GOOGLE PAY", "GPAY", "GOOGLEPAY"]
CRED = ["CRED", "DREAMPLUG TECHNOLOGIES", "CRED CLUB"]

# category -> list of (aliases, weight). Shared platforms appear in several lists.
MERCHANTS: dict[str, list[tuple[list[str], float]]] = {
    "Food & Dining": [
        (SWIGGY, 5), (ZOMATO, 5), (["DOMINOS PIZZA", "JUBILANT FOODWORKS", "DOMINOS"], 1.5),
        (["MCDONALDS", "HARDCASTLE RESTAURANTS", "MCD"], 1), (["STARBUCKS", "TATA STARBUCKS"], 0.8),
        (["SARAVANA BHAVAN", "HOTEL SARAVANA BHAVAN"], 1), (["A2B ADYAR ANANDA BHAVAN", "A2B"], 1),
        (["KFC", "DEVYANI INTERNATIONAL"], 0.8), (["CHAAYOS", "SUNSHINE TEASHOP"], 0.5),
        (["BARBEQUE NATION"], 0.4), (["JUNIOR KUPPANNA", "KUPPANNA"], 0.6),
        (PAYTM, 0.8), (GPAY, 0.4), (AMAZON, 0.2), (TATA, 0.2),
    ],
    "Groceries": [
        (["BIGBASKET", "SUPERMARKET GROCERY SUPPLIES", "BB NOW", "INNOVATIVE RETAIL"], 3),
        (["BLINKIT", "BLINK COMMERCE", "GROFERS"], 2.5), (["ZEPTO", "KIRANAKART TECHNOLOGIES"], 2.5),
        (SWIGGY, 1.5), (ZOMATO, 0.4), (AMAZON, 1.2), (FLIPKART, 0.8),
        (["DMART", "AVENUE SUPERMARTS"], 2), (RELIANCE, 1.5), (["MORE RETAIL", "MORE SUPERMARKET"], 1),
        (["NILGIRIS", "NILGIRIS DAIRY FARM"], 0.6), (["AAVIN", "AAVIN MILK"], 0.8),
        (["SPENCERS RETAIL"], 0.5), (TATA, 0.5), (PAYTM, 0.8), (PHONEPE, 0.3),
    ],
    "Transport": [
        (["UBER", "UBER INDIA SYSTEMS", "UBER TRIP"], 3), (["OLA", "ANI TECHNOLOGIES", "OLACABS"], 3),
        (["RAPIDO", "ROPPEN TRANSPORTATION"], 2), (["IRCTC", "INDIAN RAILWAY CATERING", "IRCTC UTS"], 1.5),
        (["INDIAN OIL", "IOCL", "INDIANOIL PETROL"], 1.5), (["HP PETROL", "HPCL"], 1),
        (["BHARAT PETROLEUM", "BPCL"], 1), (["CHENNAI METRO RAIL", "CMRL"], 0.8),
        (["REDBUS", "IBIBO GROUP"], 0.6), (["FASTAG", "NETC FASTAG", "ICICI FASTAG"], 0.8),
        (["NAMMA YATRI", "JUSPAY"], 0.5), (PAYTM, 1.2), (PHONEPE, 0.4), (AMAZON, 0.2),
    ],
    "Shopping": [
        (AMAZON, 5), (FLIPKART, 4), (["MYNTRA", "MYNTRA DESIGNS"], 2), (["AJIO", "RELIANCE AJIO"], 1),
        (["NYKAA", "FSN ECOMMERCE"], 1), (["MEESHO", "FASHNEAR TECHNOLOGIES"], 1),
        (["DECATHLON", "DECATHLON SPORTS"], 0.6), (["CROMA", "INFINITI RETAIL"], 0.6),
        (["RELIANCE DIGITAL", "RELIANCE TRENDS"], 0.8), (RELIANCE, 0.5), (TATA, 0.8),
        (["POTHYS", "CHENNAI SILKS", "THE CHENNAI SILKS"], 0.8), (["SARAVANA STORES"], 0.8),
        (["LIFESTYLE", "MAX FASHION", "LANDMARK GROUP"], 0.6), (["IKEA"], 0.3), (PAYTM, 0.5), (GPAY, 0.3),
    ],
    "Bills & Utilities": [
        (["TANGEDCO", "TNEB", "TN ELECTRICITY BOARD", "TNPDCL"], 2),
        (["AIRTEL", "BHARTI AIRTEL", "AIRTEL PREPAID"], 2), (["JIO", "RELIANCE JIO INFOCOMM", "JIO PREPAID"], 2),
        (["ACT FIBERNET", "ATRIA CONVERGENCE"], 1), (["TATA PLAY", "TATA SKY"], 0.8),
        (["LIC OF INDIA", "LIFE INSURANCE CORP"], 0.6), (["HDFC ERGO", "STAR HEALTH INSURANCE"], 0.5),
        (CRED, 1.2), (["CHENNAI METRO WATER", "CMWSSB"], 0.4), (["INDANE GAS", "INDANE"], 0.6),
        (AMAZON, 1), (PAYTM, 1.5), (PHONEPE, 1.2), (GPAY, 0.5), (RELIANCE, 0.3),
    ],
    "Entertainment": [
        (["NETFLIX", "NETFLIX COM"], 1.5), (["SPOTIFY", "SPOTIFY INDIA"], 1.2),
        (["HOTSTAR", "JIOSTAR", "NOVI DIGITAL", "JIOHOTSTAR"], 1.2), (["BOOKMYSHOW", "BIGTREE ENTERTAINMENT"], 2),
        (["PVR INOX", "PVR CINEMAS", "INOX LEISURE"], 1.5), (["YOUTUBE PREMIUM", "GOOGLE YOUTUBE"], 0.6),
        (["STEAM", "VALVE STEAM"], 0.3), (["DISTRICT", "PAYTM INSIDER"], 0.6), (["SONYLIV", "CULVER MAX"], 0.4),
        (AMAZON, 0.6), (PAYTM, 0.6), (TATA, 0.3), (GPAY, 0.3),
    ],
    "Health": [
        (["APOLLO PHARMACY", "APOLLO HOSPITALS", "APOLLO 247", "APOLLO"], 3),
        (["MEDPLUS", "OPTIVAL HEALTH"], 2), (["PHARMEASY", "API HOLDINGS"], 1), (["TATA 1MG", "1MG"], 1),
        (["NETMEDS"], 0.6), (["THYROCARE"], 0.5), (["KAUVERY HOSPITAL"], 0.5), (["MGM HEALTHCARE"], 0.4),
        (["PRACTO"], 0.5), (TATA, 0.4), (AMAZON, 0.3), (RELIANCE, 0.3), (PAYTM, 0.4),
    ],
    "Education": [
        (["BYJUS", "THINK AND LEARN"], 1), (["UNACADEMY", "SORTING HAT TECHNOLOGIES"], 1),
        (["COURSERA"], 1), (["UDEMY"], 1), (["PHYSICSWALLAH", "PW"], 0.8),
        (["ANNA UNIVERSITY", "ANNA UNIV EXAM FEE"], 0.6), (["SONA COLLEGE", "SONA COLLEGE OF TECH"], 0.8),
        (["HIGGINBOTHAMS"], 0.4), (["NPTEL", "IIT MADRAS NPTEL"], 0.4), (AMAZON, 0.5), (FLIPKART, 0.3),
        (PAYTM, 0.5), (GPAY, 0.3),
    ],
    "Transfers": [
        (GPAY, 1), (PHONEPE, 1), (PAYTM, 0.8), (["SELF TRANSFER", "OWN ACCOUNT"], 0.8), (CRED, 0.2),
    ],
    "Miscellaneous": [
        (["ATM WDL"], 3), (["CHARGES", "SMS CHARGES", "ANNUAL FEE", "DEBIT CARD FEE"], 1),
        (["GIVEINDIA", "DONATION", "TEMPLE TRUST"], 0.5), (["COURIER", "DTDC", "INDIA POST"], 0.5),
        (["XEROX", "STATIONERY"], 0.4), (PAYTM, 0.6), (GPAY, 0.5), (PHONEPE, 0.5), (AMAZON, 0.3),
    ],
}

FIRST = ["RAJESH", "KUMAR", "SELVAM", "PRIYA", "KARTHIK", "LAKSHMI", "MURUGAN", "DIVYA", "ARUN",
         "SANTHOSH", "MEENA", "VIGNESH", "GANESH", "KAVITHA", "SURESH", "ANAND", "DEEPA", "RAMESH",
         "SARAVANAN", "BALAJI", "NIRMALA", "VIJAY", "SENTHIL", "REVATHI", "MANI", "PANDIAN", "ANITHA"]
LAST = ["K", "M", "S", "R", "P", "KUMAR", "RAJ", "N", "V", "SUBRAMANIAN", "PANDI", "RAJAN", "A", "T"]
# Optional free-text notes a payer types — deliberately weak, overlapping hints.
NOTES = ["", "", "", "", "PAYMENT", "UPI", "SENT", "PAID", "BILL", "FEES", "ORDER", "MONTH", "THANKS", "OK"]
BANKS = ["YESB", "HDFC", "ICIC", "SBIN", "UTIB", "KKBK", "IOBA", "CNRB", "BARB", "PUNB"]
CITIES = ["SALEM TN", "CHENNAI TN", "COIMBATORE", "BANGALORE", "BANGA", "MADURAI TN", "ERODE TN",
          "TRICHY", "MUMBAI", "HYDERABAD"]
VPA_SUFFIX = ["@ybl", "@okicici", "@oksbi", "@paytm", "@axl", "@ibl", "@okhdfcbank", "@upi"]

# Daily base rates (expected txns/day) and lognormal (median INR, sigma).
DAILY_RATE = {
    "Food & Dining": 3.2, "Groceries": 1.9, "Transport": 2.6, "Shopping": 1.0, "Bills & Utilities": 0.35,
    "Entertainment": 0.55, "Health": 0.45, "Education": 0.25, "Transfers": 1.0, "Miscellaneous": 0.7,
}
AMOUNT = {
    "Food & Dining": (260, 0.6), "Groceries": (550, 0.75), "Transport": (160, 0.8), "Shopping": (1300, 0.9),
    "Bills & Utilities": (650, 0.8), "Entertainment": (420, 0.7), "Health": (480, 0.9),
    "Education": (1800, 0.9), "Transfers": (1500, 1.0), "Miscellaneous": (450, 0.9),
}
FESTIVALS = {  # date -> (days before, multipliers)
    "2024-11-01": (10, {"Shopping": 3.0, "Groceries": 1.8, "Food & Dining": 1.3}),   # Diwali 2024
    "2025-10-20": (10, {"Shopping": 3.0, "Groceries": 1.8, "Food & Dining": 1.3}),   # Diwali 2025
    "2025-01-14": (5, {"Groceries": 2.0, "Shopping": 1.8}),                          # Pongal 2025
    "2026-01-14": (5, {"Groceries": 2.0, "Shopping": 1.8}),                          # Pongal 2026
}
WEEKEND_BOOST = {"Food & Dining": 1.6, "Entertainment": 1.9, "Shopping": 1.3}
# Fixed monthly bills on the 1st-5th: (merchant aliases, median, sigma, payee-is-person)
LANDLORD = "SELVARAJ M"  # rent goes to a person via UPI/IMPS, so it looks like a Transfer
MONTHLY_BILLS = [
    (None, 14000, 0.02, True),                                    # rent to landlord (a person)
    (["TANGEDCO", "TNEB", "TN ELECTRICITY BOARD"], 1400, 0.35, False),
    (["AIRTEL", "BHARTI AIRTEL"], 599, 0.05, False),
    (["ACT FIBERNET", "ATRIA CONVERGENCE"], 1049, 0.03, False),
    (CRED, 9000, 0.5, False),                                     # credit-card bill
]


def _ref(rng, n):
    return "".join(rng.choice(list("0123456789"), size=n))


def person_name(rng):
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def render(rng, payee: str, category: str, is_person: bool) -> str:
    """Wrap a payee in one of several bank-statement formats, then add noise."""
    bank = rng.choice(BANKS)
    ifsc = f"{bank}000{_ref(rng, 4)}"
    note = rng.choice(NOTES)
    if payee == "ATM WDL":
        s = f"ATM WDL {_ref(rng, 4)} {rng.choice(CITIES)}"
    else:
        vpa = payee.lower().replace(" ", "")[:12] + rng.choice(VPA_SUFFIX) if not is_person \
            else _ref(rng, 10) + rng.choice(VPA_SUFFIX)
        fmts = [
            (f"UPI/{_ref(rng, 12)}/{payee}/{bank}{_ref(rng, 4)}", 3.0),
            (f"UPI-{payee}-{vpa}-{ifsc}-{_ref(rng, 12)}-{note}", 2.5),
            (f"UPI/DR/{_ref(rng, 12)}/{payee[:20]}/{bank}/{vpa}/{note}", 1.5),
            (f"POS {_ref(rng, 4)}*{payee} {rng.choice(CITIES)}", 0.0 if is_person else 1.2),
            (f"NEFT-DR-{bank}-{payee}", 0.8),
            (f"IMPS/P2A/{_ref(rng, 12)}/{payee}/{note}", 1.2 if is_person else 0.3),
            (f"ACH D- {payee}-{_ref(rng, 9)}", 0.0 if is_person else 0.5),
            (f"BIL/ONL/{_ref(rng, 9)}/{payee}/{_ref(rng, 6)}", 0.0 if is_person else 0.5),
            (f"{payee} {rng.choice(['ONLINE', 'INDIA', 'PVT LTD', 'IN', ''])} {rng.choice(CITIES)[:5]}", 0.0 if is_person else 0.8),
            (f"ECOM PUR {payee} {_ref(rng, 8)}", 0.0 if is_person else 0.6),
        ]
        w = np.array([f[1] for f in fmts])
        s = fmts[rng.choice(len(fmts), p=w / w.sum())][0]
    # noise: truncation, casing, trailing junk
    if rng.random() < TRUNCATE_P:
        s = s[: rng.integers(18, 33)]
    r = rng.random()
    if r < 0.15:
        s = s.lower()
    elif r < 0.25:
        s = s.title()
    if rng.random() < 0.15:
        s += rng.choice([" ", "  IN", " /", "-", " XX", " 00"])
    return s


def pick_payee(rng, category):
    if rng.random() < PERSON_SHARE[category]:
        return person_name(rng), True
    opts = MERCHANTS[category]
    w = np.array([o[1] for o in opts])
    aliases = opts[rng.choice(len(opts), p=w / w.sum())][0]
    return rng.choice(aliases), False


def generate(seed: int = config.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range(config.DATA_START, config.DATA_END, freq="D")
    fest_mult: dict[pd.Timestamp, dict[str, float]] = {}
    for d, (before, mult) in FESTIVALS.items():
        for k in range(before + 1):
            fest_mult[pd.Timestamp(d) - pd.Timedelta(days=k)] = mult

    rows = []
    for day in days:
        weekend = day.dayofweek >= 5
        for cat, base in DAILY_RATE.items():
            lam = base * (WEEKEND_BOOST.get(cat, 1.0) if weekend else 1.0)
            lam *= fest_mult.get(day, {}).get(cat, 1.0)
            for _ in range(rng.poisson(lam)):
                payee, is_person = pick_payee(rng, cat)
                med, sig = AMOUNT[cat]
                amt = rng.lognormal(np.log(med), sig)
                if payee == "ATM WDL":
                    amt = float(rng.choice([500, 1000, 2000, 3000, 5000]))
                rows.append((day, render(rng, payee, cat, is_person), amt, "debit", cat))
        # monthly salary credit on the 1st, fixed bills on the 1st-5th
        if day.day == 1:
            s = f"NEFT-CR-HDFC0000{_ref(rng, 3)}-ACME TECH SOLUTIONS PVT LTD-SAL {day.strftime('%b').upper()}"
            rows.append((day, s, round(rng.normal(72000, 500), 2), "credit", "Transfers"))
            # each fixed bill is paid once a month on a random day between the 1st and the 5th
            for aliases, med, sig, is_person in MONTHLY_BILLS:
                pay_day = day + pd.Timedelta(days=int(rng.integers(0, 5)))
                payee = LANDLORD if is_person else rng.choice(aliases)
                txt = render(rng, payee, "Bills & Utilities", is_person)
                rows.append((pay_day, txt, rng.lognormal(np.log(med), sig), "debit", "Bills & Utilities"))

    df = pd.DataFrame(rows, columns=["date", "merchant_raw", "amount", "direction", "category"])
    df = df[df["date"] <= pd.Timestamp(config.DATA_END)]
    df = df.sort_values(["date", "category", "merchant_raw"], kind="mergesort").reset_index(drop=True)

    # Inject ~2% amount outliers within category (debits only). Ground truth only.
    df["is_anomaly"] = 0
    debit_idx = df.index[df["direction"] == "debit"].to_numpy()
    n_anom = int(round(ANOMALY_RATE * len(debit_idx)))
    anom = rng.choice(debit_idx, size=n_anom, replace=False)
    df.loc[anom, "amount"] *= rng.uniform(4.0, 10.0, size=n_anom)
    df.loc[anom, "is_anomaly"] = 1

    df["amount"] = df["amount"].round(2)
    df.insert(0, "txn_id", [f"T{i:06d}" for i in range(len(df))])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def main():
    df = generate()
    config.DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(config.TRANSACTIONS_CSV, index=False)
    print(f"Wrote {config.TRANSACTIONS_CSV}  rows={len(df):,}  span={df.date.min()} -> {df.date.max()}")
    print("\nCategory mix:")
    print(df["category"].value_counts().to_string())
    print(f"\nDirection: {df['direction'].value_counts().to_dict()}")
    print(f"Anomalies injected: {df.is_anomaly.sum()} ({df.is_anomaly.mean():.2%} of all rows)")
    print("\n15 sample merchant strings:")
    for _, r in df.sample(15, random_state=1).iterrows():
        print(f"  {r.category:<18} {r.merchant_raw}")


if __name__ == "__main__":
    main()
