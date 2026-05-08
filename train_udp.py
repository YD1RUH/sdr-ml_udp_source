#!/usr/bin/env python3
"""
train_udp.py  —  Training script pengganti train_validated.py
-------------------------------------------------------------
Langsung baca dari: datasets_validated(slash)datasets_validated
Simpan model ke   : models

Cara pakai:
    python train_udp.py

Atau dengan path custom:
    python train_udp.py --data-dir datasets_validated(slash)datasets_validated
"""

import os
import sys
import argparse
import numpy as np
import joblib
from pathlib import Path

# Tambahkan src/ ke path agar signal_features bisa di-import
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from signal_features import SignalFeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_dataset(data_dir: str):
    """Muat semua file .npy dari subfolder label."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"[ERROR] Folder tidak ditemukan: {data_path.resolve()}")
        sys.exit(1)

    extractor = SignalFeatureExtractor()
    X, y = [], []
    
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    if not class_dirs:
        print(f"[ERROR] Tidak ada subfolder kelas di: {data_path.resolve()}")
        print("  Pastikan struktur folder:")
        print("  datasets_validated\\datasets_validated\\")
        print("    APRS\\  FM_broadcast\\  FRS_GMRS\\  ...")
        sys.exit(1)

    print(f"Ditemukan {len(class_dirs)} kelas: {[d.name for d in class_dirs]}")
    print()

    for class_dir in class_dirs:
        npy_files = list(class_dir.glob("*.npy"))
        label = class_dir.name
        loaded = 0
        errors = 0

        for fpath in npy_files:
            try:
                data = np.load(str(fpath), allow_pickle=True).item()
                
                # Support berbagai format key
                if "samples" in data:
                    samples = data["samples"]
                elif "iq" in data:
                    samples = data["iq"]
                else:
                    # Coba ambil array pertama
                    for v in data.values():
                        if isinstance(v, np.ndarray) and np.iscomplexobj(v):
                            samples = v
                            break
                    else:
                        errors += 1
                        continue

                samples = samples.astype(np.complex64)
                features = extractor.extract_features(samples)
                
                if features is not None and len(features) > 0:
                    X.append(features)
                    y.append(label)
                    loaded += 1

            except Exception as e:
                errors += 1
                if errors <= 3:  # Hanya tampilkan 3 error pertama
                    print(f"  [WARN] {fpath.name}: {e}")

        print(f"  {label:20s}: {loaded:3d} sampel dimuat"
              + (f"  ({errors} error)" if errors else ""))

    if len(X) == 0:
        print("\n[ERROR] Tidak ada sampel yang berhasil dimuat!")
        print("  Cek apakah file .npy bisa dibaca dengan:")
        print("  python -c \"import numpy as np; d=np.load('path/ke/file.npy',allow_pickle=True); print(d)\"")
        sys.exit(1)

    return np.array(X), np.array(y)


def train(data_dir: str, model_dir: str, test_size: float = 0.2):
    print("=" * 70)
    print("TRAINING RF CLASSIFIER — rtl-ml UDP mode")
    print("=" * 70)
    print(f"Data  : {Path(data_dir).resolve()}")
    print(f"Model : {Path(model_dir).resolve()}")
    print()

    # ── Load ──
    X, y = load_dataset(data_dir)
    print(f"\nTotal: {len(X)} sampel, {len(np.unique(y))} kelas")

    # ── Encode label ──
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"Label : {list(le.classes_)}")

    # ── Split train/test ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=42, stratify=y_enc
    )
    print(f"Train : {len(X_train)} | Test: {len(X_test)}")

    # ── Scale ──
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Train ──
    print("\nTraining Random Forest (100 trees) …")
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    clf.fit(X_train_sc, y_train)

    # ── Evaluate ──
    y_pred = clf.predict(X_test_sc)
    acc = (y_pred == y_test).mean() * 100
    print(f"\nAkurasi : {acc:.1f}%")
    print()
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Confusion matrix ──
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    header = f"{'':15s}" + "".join(f"{c[:8]:>10s}" for c in le.classes_)
    print(header)
    for i, row in enumerate(cm):
        print(f"{le.classes_[i][:15]:15s}" + "".join(f"{v:>10d}" for v in row))

    # ── Simpan model ──
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf,    os.path.join(model_dir, "rf_classifier.pkl"))
    joblib.dump(le,     os.path.join(model_dir, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

    print(f"\n[OK] Model disimpan ke {model_dir}/")
    print(f"     rf_classifier.pkl")
    print(f"     label_encoder.pkl")
    print(f"     scaler.pkl")
    print()
    print("Sekarang jalankan:")
    print("  python src/classify_live_udp.py --port 5000 --loop")


def main():
    parser = argparse.ArgumentParser(description="Training rtl-ml dari dataset lokal")
    parser.add_argument(
        "--data-dir",
        default=os.path.join("datasets_validated", "datasets_validated"),
        help="Path ke folder dataset"
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Path output model (default: models/)"
    )
    parser.add_argument(
        "--test-size",
        type=float, default=0.2,
        help="Proporsi data test (default: 0.2)"
    )
    args = parser.parse_args()
    train(args.data_dir, args.model_dir, args.test_size)


if __name__ == "__main__":
    main()