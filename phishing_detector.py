#!/usr/bin/env python3
"""
Phishing Email Detection Model
--------------------------------
Trains a scikit-learn model to classify emails as "Phishing" or
"Safe" (legitimate) based on their text content and simple URL /
keyword-based features.

Usage:
    python3 phishing_detector.py                     # uses built-in sample dataset
    python3 phishing_detector.py --data emails.csv    # uses your own CSV

Your CSV (if provided) must have two columns:
    text  -> the email subject+body as a single string
    label -> "phishing" or "safe" (or 1/0)

Recommended real-world datasets (download and pass with --data):
    - Kaggle: "Phishing Email Detection" dataset
    - Kaggle: "Nazario Phishing Corpus" + Enron "legitimate" emails
    - UCI: Phishing Websites Dataset (URL-focused variant)
"""

import argparse
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save the image
import matplotlib.pyplot as plt

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


# ----------------------------------------------------------------------
# 1. Sample dataset (used when no --data CSV is supplied)
# ----------------------------------------------------------------------

def create_sample_dataset():
    """
    Small built-in dataset of templated phishing vs. legitimate emails,
    with light randomized variation, so the pipeline runs end-to-end
    without needing an external download. Replace with a real dataset
    (see module docstring) for a stronger, more credible result.
    """
    phishing_templates = [
        "Urgent: Your account will be suspended! Click here to verify your account now: http://{d}/verify?id={n}",
        "Dear customer, we detected suspicious activity. Verify your identity immediately at http://{d}/login-{n}",
        "Congratulations! You have won a prize. Claim your reward now at http://{d}/claim/{n} before it expires!",
        "Your payment failed. Update your billing information at http://{d}/billing?ref={n} or your account will be closed.",
        "Security Alert: unusual sign-in detected. Confirm your password at http://{d}/secure-{n} within 24 hours.",
        "ACTION REQUIRED: Your mailbox is full. Click http://{d}/upgrade-{n} to increase storage immediately!!!",
        "We could not verify your recent transaction. Please login at http://{d}/verify-account-{n} to avoid suspension.",
        "Your Netflix subscription payment was declined. Update your card now: http://{d}/renew?user={n}",
        "IRS Notice: You have a pending refund. Submit your details at http://{d}/refund/{n} to receive payment.",
        "Your package could not be delivered. Confirm your address at http://{d}/delivery-{n} within 48 hours.",
    ]

    legit_templates = [
        "Hi team, attaching the quarterly report for review. Let me know if you have questions before Friday's meeting.",
        "Reminder: our project sync is scheduled for 3 PM tomorrow. Agenda is attached.",
        "Thanks for your email. I'll review the document and get back to you by end of week.",
        "Here are the meeting notes from today's call. Please add any corrections in the shared doc.",
        "Your order has shipped and is on its way. Estimated delivery is in 3-5 business days. Track it from your account dashboard.",
        "Just following up on our conversation last week regarding the budget proposal.",
        "The invoice for last month's services is attached for your records.",
        "Looking forward to catching up at the conference next month. Let me know your travel dates.",
        "Please find attached the updated syllabus for next semester's course.",
        "Happy birthday! Hope you have a wonderful day, let's grab lunch soon.",
    ]

    domains = ["secure-update.info", "account-verify.net", "login-alert.com", "confirm-now.biz"]

    rows = []
    rng = np.random.default_rng(42)

    for i in range(120):
        tmpl = phishing_templates[i % len(phishing_templates)]
        text = tmpl.format(d=rng.choice(domains), n=rng.integers(1000, 9999))
        rows.append({"text": text, "label": "phishing"})

    for i in range(120):
        tmpl = legit_templates[i % len(legit_templates)]
        # occasionally add a normal, non-suspicious link to add realistic variety
        if rng.random() < 0.2:
            tmpl += " More info: https://company-internal.com/docs"
        rows.append({"text": tmpl, "label": "safe"})

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------------------

SUSPICIOUS_KEYWORDS = [
    "urgent", "verify", "suspend", "suspended", "click here", "confirm",
    "password", "update your", "act now", "limited time", "winner",
    "congratulations", "claim", "security alert", "unusual activity",
    "immediately", "account will be", "expire", "refund",
]

URL_PATTERN = re.compile(r"https?://\S+")
IP_URL_PATTERN = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}")


def extract_numeric_features(texts):
    """Hand-crafted features that complement the TF-IDF text vectors."""
    feats = []
    for t in texts:
        t_lower = t.lower()
        urls = URL_PATTERN.findall(t)
        num_urls = len(urls)
        has_ip_url = 1 if IP_URL_PATTERN.search(t) else 0
        keyword_hits = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in t_lower)
        exclamations = t.count("!")
        length = len(t)
        upper_chars = sum(1 for c in t if c.isupper())
        upper_ratio = upper_chars / length if length > 0 else 0
        feats.append([num_urls, has_ip_url, keyword_hits, exclamations, length, upper_ratio])
    cols = ["num_urls", "has_ip_url", "keyword_hits", "exclamations", "length", "upper_ratio"]
    return pd.DataFrame(feats, columns=cols)


def build_features(texts, vectorizer, fit=False):
    """Combine TF-IDF text vectors with numeric engineered features."""
    if fit:
        tfidf = vectorizer.fit_transform(texts)
    else:
        tfidf = vectorizer.transform(texts)

    numeric_df = extract_numeric_features(texts)
    numeric_sparse = csr_matrix(numeric_df.values.astype(float))
    combined = hstack([tfidf, numeric_sparse])
    return combined


# ----------------------------------------------------------------------
# 3. Main pipeline
# ----------------------------------------------------------------------

def normalize_label(label):
    if isinstance(label, (int, np.integer)):
        return "phishing" if label == 1 else "safe"
    label = str(label).strip().lower()
    if label in ("1", "phishing", "spam", "malicious"):
        return "phishing"
    return "safe"


def main():
    parser = argparse.ArgumentParser(description="Phishing Email Detection Model (scikit-learn)")
    parser.add_argument("--data", default=None,
                         help="Path to a CSV with 'text' and 'label' columns. "
                              "If omitted, a built-in sample dataset is used.")
    parser.add_argument("--out", default="confusion_matrix.png",
                         help="Filename for the saved confusion matrix image")
    args = parser.parse_args()

    if args.data:
        print(f"[*] Loading dataset from {args.data}")
        df = pd.read_csv(args.data)
        if "text" not in df.columns or "label" not in df.columns:
            print("[!] CSV must have 'text' and 'label' columns.")
            sys.exit(1)
    else:
        print("[*] No --data provided; using built-in sample dataset.")
        print("    (For a stronger result, download a real dataset — see script docstring — "
              "and pass it with --data your_file.csv)")
        df = create_sample_dataset()

    df["label"] = df["label"].apply(normalize_label)
    df = df.dropna(subset=["text", "label"])

    print(f"[*] Dataset size: {len(df)} emails "
          f"({(df['label'] == 'phishing').sum()} phishing, "
          f"{(df['label'] == 'safe').sum()} safe)")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    print("[*] Extracting TF-IDF + keyword/URL features...")
    vectorizer = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
    X_train = build_features(X_train_text, vectorizer, fit=True)
    X_test = build_features(X_test_text, vectorizer, fit=False)

    print("[*] Training Logistic Regression classifier...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 50)
    print(f"Accuracy: {acc * 100:.2f}%")
    print("=" * 50)
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred))

    labels = ["phishing", "safe"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(f"           {labels[0]:>10} {labels[1]:>10}")
    for i, row_label in enumerate(labels):
        print(f"{row_label:>10} {cm[i][0]:>10} {cm[i][1]:>10}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix (Accuracy: {acc*100:.1f}%)")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\n[*] Confusion matrix image saved to: {args.out}")


if __name__ == "__main__":
    main()
