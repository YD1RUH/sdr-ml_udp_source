#!/usr/bin/env python3
"""
classify_live_udp.py  —  rtl-ml classifier dengan sumber IQ dari UDP
--------------------------------------------------------------------
Cara pakai:
    python src/classify_live_udp.py --port 5000 --loop

Perubahan dari versi asal:
  - DC blocker ditingkatkan (IIR high-pass Butterworth + mean removal)
  - Bandwidth sanity-check: sinyal >150 kHz otomatis di-override ke FM_broadcast
  - Kolom bandwidth estimasi ditampilkan di output
"""

import argparse
import socket
import time
import numpy as np
import joblib
import os
import sys
from scipy import signal as sp_signal

# Tambahkan direktori src/ langsung ke path (hindari konflik package "src" sistem)
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from signal_features import SignalFeatureExtractor

# ── Konstanta ──────────────────────────────────────────────────────────────────
BYTES_PER_SAMPLE = 8    # complex64 = 4 byte I + 4 byte Q
MTU              = 1472  # bytes per UDP packet

# Threshold bandwidth untuk FM broadcast (kHz).
# FM broadcast ~200 kHz, APRS ~25 kHz, pager ~25 kHz.
FM_BW_THRESHOLD_KHZ = 150.0

# Minimum SNR (peak PSD vs median noise floor) agar override aktif.
# Noise memiliki spektrum datar → SNR ≈ 0-3 dB (tidak ada puncak nyata).
# FM broadcast memiliki puncak jelas → SNR ≥ 10-20 dB.
# Override hanya aktif jika KEDUA kondisi terpenuhi: BW > threshold DAN SNR > min.
FM_MIN_SNR_DB = 8.0


# ── UDP IQ Receiver ────────────────────────────────────────────────────────────

class UDPIQReceiver:
    def __init__(self, host="127.0.0.1", port=5000, timeout=10.0):
        self.host    = host
        self.port    = port
        self.timeout = timeout
        self.sock    = None

    def open(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(self.timeout)
        print(f"[UDP] Mendengarkan di {self.host}:{self.port} ...")

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def read_samples(self, num_samples):
        buf = np.empty(num_samples, dtype=np.complex64)
        collected = 0
        while collected < num_samples:
            try:
                data, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                raise TimeoutError(
                    f"[UDP] Tidak ada data setelah {self.timeout}s. "
                    "Pastikan GNU Radio flowgraph sudah berjalan."
                )
            n_floats  = len(data) // 4
            n_complex = n_floats // 2
            if n_complex == 0:
                continue
            floats  = np.frombuffer(data[:n_complex * 8], dtype=np.float32)
            samples = floats[0::2] + 1j * floats[1::2]
            take = min(n_complex, num_samples - collected)
            buf[collected:collected + take] = samples[:take]
            collected += take
        return buf


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(samples, samp_rate=1.024e6):
    """
    Preprocessing IQ samples:
      1. Hapus DC offset kasar via mean subtraction
      2. DC blocker IIR (Butterworth high-pass orde 1, cutoff 1 kHz)
         -> menghilangkan spike DC residual dari BladeRF / RTL-SDR
      3. Normalisasi amplitudo ke puncak = 1.0
    """
    # Langkah 1: Hapus rata-rata (DC kasar)
    samples = samples - np.mean(samples)

    # Langkah 2: IIR high-pass untuk DC blocker
    # Cutoff 1 kHz jauh di bawah konten IQ baseband sehingga tidak merusak sinyal.
    cutoff_hz = 1000.0
    nyq       = samp_rate / 2.0
    b, a      = sp_signal.butter(1, cutoff_hz / nyq, btype='high')

    # scipy.lfilter hanya menerima array real, filter I dan Q terpisah
    i_filt  = sp_signal.lfilter(b, a, samples.real).astype(np.float32)
    q_filt  = sp_signal.lfilter(b, a, samples.imag).astype(np.float32)
    samples = i_filt + 1j * q_filt

    # Langkah 3: Normalisasi
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples = samples / peak

    return samples


# ── Estimasi Bandwidth ─────────────────────────────────────────────────────────

def estimate_signal_metrics(samples, samp_rate=1.024e6, fft_size=8192,
                             db_threshold=20.0):
    """
    Estimasi bandwidth dan SNR sinyal dari PSD.

    Mengembalikan (bw_khz, snr_db):
      bw_khz  — lebar -db_threshold dB dari puncak (dalam kHz)
      snr_db  — selisih peak PSD vs median noise floor (dB)

    Cara membedakan noise vs FM broadcast:
      Noise       : spektrum datar  -> snr_db ~0-4 dB,  bw_khz ~samp_rate penuh
      FM broadcast: puncak lokal    -> snr_db ~10-25 dB, bw_khz ~150-220 kHz
      APRS/Pager  : puncak sempit   -> snr_db ~8-20 dB,  bw_khz ~15-40 kHz

    Referensi bandwidth:
      FM broadcast  : 150-220 kHz
      APRS          :  15-30  kHz
      Pager         :  15-30  kHz
      ISM sensor    :  <50    kHz (burst)
    """
    # Pakai segmen tengah untuk menghindari transien di ujung buffer
    n   = min(fft_size, len(samples))
    mid = len(samples) // 2
    seg = samples[mid - n // 2 : mid + n // 2]

    window  = np.blackman(len(seg))
    fft_out = np.fft.fftshift(np.fft.fft(seg * window, fft_size))
    psd_db  = 10.0 * np.log10(np.abs(fft_out) ** 2 + 1e-12)

    peak_db        = np.max(psd_db)
    noise_floor_db = np.median(psd_db)   # median = robust noise floor estimate
    snr_db         = peak_db - noise_floor_db

    threshold = peak_db - db_threshold
    above     = psd_db > threshold

    if not np.any(above):
        return 0.0, snr_db

    freqs  = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / samp_rate))
    bw_hz  = freqs[above][-1] - freqs[above][0]
    return max(0.0, bw_hz / 1000.0), snr_db


# ── Klasifikasi ────────────────────────────────────────────────────────────────

def classify_from_udp(receiver, extractor, model, label_enc, scaler,
                      num_samples, samp_rate=1.024e6):
    t0      = time.time()
    samples = receiver.read_samples(num_samples)
    t1      = time.time()

    samples = preprocess(samples, samp_rate=samp_rate)

    # Estimasi bandwidth DAN SNR sebelum feature extraction
    bw_khz, snr_db = estimate_signal_metrics(samples, samp_rate=samp_rate)

    features    = extractor.extract_features(samples)
    features_2d = features.reshape(1, -1)

    if scaler is not None:
        features_2d = scaler.transform(features_2d)

    pred_idx   = model.predict(features_2d)[0]
    pred_prob  = model.predict_proba(features_2d)[0]
    label      = label_enc.inverse_transform([pred_idx])[0]
    confidence = pred_prob[pred_idx]

    # ── Bandwidth + SNR sanity-check ──────────────────────────────────────────
    # Override ke FM hanya jika SEMUA kondisi terpenuhi:
    #   1. Bandwidth  > FM_BW_THRESHOLD_KHZ  (~150 kHz)
    #   2. SNR        > FM_MIN_SNR_DB         (~8 dB) — noise punya SNR ~0-3 dB
    #   3. Model BUKAN prediksi 'noise'       — kalau sudah noise, percayai model
    #   4. Model punya sedikit confidence FM  — jika FM = 0.0%, jangan paksa
    #
    # Ini mencegah dua false-positive:
    #   a) Noise (spektrum datar, BW besar tapi SNR rendah)  → kondisi 2 menolak
    #   b) Noise dengan SNR kebetulan tinggi                  → kondisi 3&4 menolak
    bw_override    = False
    original_label = label
    is_noise       = 'noise' in label.lower()

    fm_candidates  = [c for c in label_enc.classes_
                      if 'FM' in c or 'fm' in c.lower() or 'broadcast' in c.lower()]
    fm_conf_in_model = sum(pred_prob[i] for i, c in enumerate(label_enc.classes_)
                           if c in fm_candidates)

    if (not is_noise
            and bw_khz >= FM_BW_THRESHOLD_KHZ
            and snr_db  >= FM_MIN_SNR_DB
            and fm_conf_in_model > 0.05          # model minimal 5% yakin FM
            and fm_candidates
            and label not in fm_candidates):
        label       = fm_candidates[0]
        confidence  = min(0.99, confidence + 0.40)
        bw_override = True
        print(f"   [BW-OVERRIDE] Model prediksi '{original_label}' "
              f"tapi BW={bw_khz:.0f} kHz SNR={snr_db:.1f} dB FM-conf={fm_conf_in_model*100:.0f}% "
              f"-> di-override ke '{label}'")

    return label, confidence, t1 - t0, pred_prob, label_enc.classes_, bw_khz, snr_db, bw_override


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="rtl-ml classifier — sumber IQ via UDP (GNU Radio)"
    )
    parser.add_argument("--host",        default="127.0.0.1")
    parser.add_argument("--port",        type=int, default=5000)
    parser.add_argument("--sample-rate", type=float, default=1.024e6)
    parser.add_argument("--num-samples", type=int, default=512000)
    parser.add_argument("--timeout",     type=float, default=10.0)
    parser.add_argument("--loop",        action="store_true")
    parser.add_argument("--model-dir",   default="models")
    args = parser.parse_args()

    # ── Muat model ──
    model_path  = os.path.join(args.model_dir, "rf_classifier.pkl")
    labels_path = os.path.join(args.model_dir, "label_encoder.pkl")
    scaler_path = os.path.join(args.model_dir, "scaler.pkl")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        print("  Jalankan 'python train_udp.py' terlebih dahulu.")
        sys.exit(1)

    print(f"[INFO] Memuat model ...")
    model     = joblib.load(model_path)
    label_enc = joblib.load(labels_path)
    scaler    = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    if scaler is not None:
        print(f"[INFO] Scaler dimuat dari {scaler_path}")
    else:
        print(f"[WARN] scaler.pkl tidak ditemukan, fitur tidak di-scale")

    extractor = SignalFeatureExtractor()

    duration = args.num_samples / args.sample_rate
    print(f"[INFO] UDP          : {args.host}:{args.port}")
    print(f"[INFO] Sample rate  : {args.sample_rate/1e6:.3f} MSPS")
    print(f"[INFO] Durasi burst : {duration:.2f} detik ({args.num_samples:,} sampel)")
    print(f"[INFO] Loop mode    : {'Ya' if args.loop else 'Tidak'}")
    print(f"[INFO] FM BW thresh : {FM_BW_THRESHOLD_KHZ:.0f} kHz (sanity-check)")
    print()

    receiver = UDPIQReceiver(host=args.host, port=args.port, timeout=args.timeout)
    receiver.open()

    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"-- Klasifikasi #{iteration}")
            print(f"   Mengumpulkan {args.num_samples:,} sampel ...")

            try:
                label, conf, elapsed, all_probs, classes, bw_khz, snr_db, bw_override = \
                    classify_from_udp(
                        receiver, extractor, model, label_enc, scaler,
                        args.num_samples, samp_rate=args.sample_rate
                    )
            except TimeoutError as e:
                print(f"   {e}")
                if not args.loop:
                    break
                continue

            bar    = "#" * int(conf * 20)
            bw_tag = " [BW-OVERRIDE]" if bw_override else ""
            print(f"   +------------------------------------------+")
            print(f"   |  >> PREDIKSI : {label:<27}")
            print(f"   |     Keyakinan: {conf*100:5.1f}%  {bar:<20}")
            print(f"   |     Bandwidth: {bw_khz:6.1f} kHz / SNR {snr_db:5.1f} dB{bw_tag:<5}")
            print(f"   |     Waktu    : {elapsed:.3f} detik")
            print(f"   +------------------------------------------+")

            sorted_idx = sorted(range(len(all_probs)),
                                key=lambda i: all_probs[i], reverse=True)
            for i in sorted_idx:
                prob = all_probs[i]
                cls  = classes[i]
                bar2 = "#" * int(prob * 30)
                marker = " <<" if cls == label else "   "
                print(f"   {cls:<20} {prob*100:5.1f}%  {bar2}{marker}")
            print()

            if not args.loop:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan oleh pengguna.")
    finally:
        receiver.close()


if __name__ == "__main__":
    main()