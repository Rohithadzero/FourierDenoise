# 🎵 FourierDenoise — Audio Signal Denoising via Fourier + Linear Algebra

A Windows desktop application that demonstrates how **Fourier Transform** and **Linear Algebra** can denoise audio signals. Load a noisy audio file, apply mathematical denoising, and hear the difference!

## ✨ Features

- **3 Denoising Methods** — Spectral Gate, SVD Subspace Projection, Wiener Filter
- **Record or Load Audio** — Record from microphone or load WAV/FLAC files
- **Add Synthetic Noise** — White, Pink, or Brown noise with adjustable SNR
- **Rich Visualizations** — Waveform, Frequency Spectrum, Spectrogram, SVD Analysis
- **Math Explainer** — Step-by-step linear algebra explanations for each method
- **Before/After Playback** — Hear the denoising results instantly
- **Export** — Save denoised audio as WAV

## 🧮 Mathematical Concepts

| Concept | Application |
|---|---|
| **Vector Spaces** | Audio signals as vectors in ℝⁿ |
| **Orthogonal Bases** | DFT basis {e^(2πikn/N)} — orthonormal decomposition |
| **Projection** | Denoising = projecting onto clean subspace |
| **Fourier Transform** | FFT as a linear map F: ℝⁿ → ℂⁿ |
| **SVD** | Rank-k approximation for optimal subspace projection |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Windows 10/11

### Setup & Run
```bash
# 1. Run the build script (creates venv, installs deps)
build.bat

# 2. Launch the application
run.bat
```

### Or manually:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## 🎯 How To Use

1. **Load Audio** — Click "Load Audio" or "Test Signal" to get started
2. **Add Noise** — Select noise type and SNR, click "Inject Noise"
3. **Choose Method** — Pick Spectral Gate, SVD, or Wiener Filter
4. **Denoise** — Adjust strength and click "DENOISE"
5. **Compare** — Play before/after audio and explore the visualizations
6. **Export** — Save the denoised result as a WAV file

## 🛠 Tech Stack

- **Python** — Core language
- **CustomTkinter** — Modern dark-mode GUI
- **NumPy / SciPy** — FFT, SVD, signal processing
- **Matplotlib** — Embedded visualizations
- **SoundFile / SoundDevice** — Audio I/O and playback

## 📄 License

MIT License
