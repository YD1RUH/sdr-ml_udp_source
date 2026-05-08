# sdr-ml_udp_source
sdr-ml_udp_source, Versi modifikasi dari repositori pengklasifikasi sinyal radio AI ini mengadaptasi alur kerja aslinya untuk mendukung perangkat keras bladeRF dan integrasi jaringan menggunakan GNU Radio. Alih-alih mengandalkan pengambilan sampel data IQ mentah secara langsung melalui skrip Python bawaan (seperti pyrtlsdr), sistem ini dirancang untuk menerima aliran data secara real-time melalui koneksi UDP dari sinyal yang telah diproses terlebih dahulu oleh flowgraph GNU Radio. Perubahan arsitektur (pipeline) ini memberikan fleksibilitas prapemrosesan sinyal yang lebih tangguh pada SDR kelas atas sebelum data tersebut masuk ke tahap ekstraksi fitur dan diprediksi oleh model Random Forest. Meskipun antarmuka penangkapan datanya dirombak, proyek ini tetap mempertahankan inti utamanya sebagai sistem klasifikasi 7 jenis sinyal radio berbasis machine learning yang efisien dan ringan untuk dijalankan pada perangkat komputasi edge.

## Skema
| Nama Skema | Gambar |
| :--- | :--- |
| **rtl-ml** | ![SKEMA1](https://github.com/YD1RUH/sdr-ml_udp_source/blob/main/SKEMA1.png?raw=true) |
| **blade-ml** | ![SKEMA12](https://github.com/YD1RUH/sdr-ml_udp_source/blob/main/SKEMA2.png?raw=true) |

---

## Perhatian !!!
- letakkan ```classify_live_udp.py``` ke dalam directory src
- letakkan ```iq_to_udp.grc```, ```iq_to_udp_fosphor.grc```, ```train_udp.py``` di root repository sdr-ml (TrevTron)

## Step-by-Step
clone repo
```git clone https://github.com/TrevTron/rtl-ml.git```

## Install Depedencies
- ```python3 -m pip install -r requirements.txt```
- ```python3 -m pip install huggingface-hub```

## Download dataset
Buka CMD kemudian jalankan command
- ```python3```
- ```from huggingface_hub import snapshot_download```
- ```snapshot_download(repo_id="TrevTron/rtl-ml-dataset", repo_type="dataset", local_dir="datasets_validated")```

## TRAINING RF CLASSIFIER
```python3 train_udp.py```

output:
- rf_classifier.pkl
- label_encoder.pkl
- scaler.pkl

## Simulasi
#### buka gnuradio companion kemudian run workflow graph 
- ```iq_to_udp_fosphor.grc``` Jika ingin menggunakan GUI
- ```iq_to_udp.grc``` Jika tidak menggunakan GUI

### gunakan model yang sudah di training untuk memprediksi modulasi Sinyal yang di sampling
run:
```python3 src/classify_live_udp.py --port 5000 --loop```

