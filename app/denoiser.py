"""
FourierDenoise — Core Denoising Algorithms
Implements three denoising methods based on Fourier Transform + Linear Algebra:

This is just the core part of the whole application(Denoiser.py), There are many files inculded in the application
for the whole application to run and excecute all the other files should be downloaded and all the dependecies must be installed!.
1. Spectral Gate  — Threshold projection in frequency space
2. SVD Denoise    — Rank-k approximation of the spectrogram matrix
3. Wiener Filter  — Optimal linear estimator in frequency domain

Mathematical Foundation:
- Audio signals live in ℝⁿ (vector space of n samples)
- DFT basis {e^(2πikn/N)} is an orthonormal basis for ℂⁿ
- FFT computes x̂ = F·x  (linear map from time → frequency domain)
- Denoising = projecting onto a "clean" subspace of the frequency space
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.linalg import svd
import time


class DenoiseResult:
    """Container for denoising results and metadata."""

    def __init__(self, denoised: np.ndarray, sr: int, method: str):
        self.denoised = denoised
        self.sr = sr
        self.method = method
        self.processing_time = 0.0
        self.snr_before = 0.0
        self.snr_after = 0.0
        self.snr_improvement = 0.0

        # Method-specific data
        self.noise_profile = None          # Estimated noise spectrum
        self.frequency_response = None     # Filter frequency response
        self.singular_values = None        # SVD singular values
        self.rank_used = None              # SVD rank used
        self.total_rank = None             # Total possible rank
        self.stft_before = None            # STFT of noisy signal
        self.stft_after = None             # STFT of denoised signal
        self.freqs = None                  # Frequency axis
        self.times = None                  # Time axis for STFT


class AudioDenoiser:
    """
    Audio denoising engine using Fourier Transform and Linear Algebra.

    The core insight: audio signals are vectors in a high-dimensional space.
    The DFT provides an orthonormal basis for this space. Noise tends to
    spread across all frequencies, while meaningful signal concentrates
    in specific frequency bands. Denoising = projecting onto the subspace
    spanned by the "signal" frequencies.
    """

    # ─── Method 1: Spectral Gate ──────────────────────────────────

    @staticmethod
    def spectral_gate(signal_data: np.ndarray, sr: int,
                      threshold_factor: float = 1.5,
                      noise_segment_duration: float = 0.5) -> DenoiseResult:
        """
        Spectral Gating — Threshold Projection in Frequency Space.

        Math:
            1. Compute STFT: X(t,f) = Σ x(n) · w(n-t) · e^(-j2πfn/N)
            2. Estimate noise spectrum from initial segment:
               N(f) = E[|X_noise(t,f)|²]
            3. Compute gate mask: G(t,f) = 1 if |X(t,f)|² > α·N(f), else 0
            4. Apply: Y(t,f) = G(t,f) · X(t,f)
            5. Inverse STFT: y(n) = ISTFT(Y)

        This is a PROJECTION: we project the signal onto the subspace
        of frequency components that exceed the noise floor.

        Args:
            signal_data: Noisy input signal (vector in ℝⁿ)
            sr: Sample rate
            threshold_factor: α — multiplier for noise threshold (1.0-5.0)
            noise_segment_duration: Duration of initial noise sample (seconds)

        Returns:
            DenoiseResult with denoised signal and metadata
        """
        t_start = time.time()
        result = DenoiseResult(None, sr, "Spectral Gate")

        n_fft = 2048
        hop_length = n_fft // 4
        window = scipy_signal.windows.hann(n_fft)

        # ── Step 1: Compute STFT (Linear Map: time → time-frequency) ──
        freqs, times, stft_matrix = scipy_signal.stft(
            signal_data, fs=sr, window=window,
            nperseg=n_fft, noverlap=n_fft - hop_length
        )
        result.stft_before = np.abs(stft_matrix)
        result.freqs = freqs
        result.times = times

        # ── Step 2: Estimate noise spectrum from initial segment ──
        noise_samples = int(noise_segment_duration * sr)
        noise_samples = min(noise_samples, len(signal_data) // 4)

        noise_segment = signal_data[:noise_samples]
        _, _, noise_stft = scipy_signal.stft(
            noise_segment, fs=sr, window=window,
            nperseg=n_fft, noverlap=n_fft - hop_length
        )

        # Noise power spectrum: E[|N(f)|²]
        noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)
        result.noise_profile = np.sqrt(noise_power.flatten())

        # ── Step 3: Compute gate mask (Projection operator) ──
        signal_power = np.abs(stft_matrix) ** 2
        gate_mask = (signal_power > threshold_factor * noise_power).astype(float)

        # Smooth the mask to reduce musical noise artifacts
        from scipy.ndimage import uniform_filter
        gate_mask = uniform_filter(gate_mask, size=(3, 5))
        gate_mask = np.clip(gate_mask, 0, 1)

        result.frequency_response = np.mean(gate_mask, axis=1)

        # ── Step 4: Apply gate (Projection onto clean subspace) ──
        clean_stft = gate_mask * stft_matrix

        result.stft_after = np.abs(clean_stft)

        # ── Step 5: Inverse STFT (Linear Map: frequency → time) ──
        _, denoised = scipy_signal.istft(
            clean_stft, fs=sr, window=window,
            nperseg=n_fft, noverlap=n_fft - hop_length
        )

        # Match lengths
        min_len = min(len(signal_data), len(denoised))
        denoised = denoised[:min_len]

        result.denoised = denoised
        result.processing_time = time.time() - t_start

        return result

    # ─── Method 2: SVD Denoise ────────────────────────────────────

    @staticmethod
    def svd_denoise(signal_data: np.ndarray, sr: int,
                    rank_ratio: float = 0.3) -> DenoiseResult:
        """
        SVD-Based Subspace Projection — Rank-k Matrix Approximation.

        Math:
            1. Compute STFT → Spectrogram matrix S ∈ ℂ^(F×T)
            2. SVD decomposition: S = U Σ Vᴴ
               - U: left singular vectors (frequency patterns)
               - Σ: singular values (importance weights)
               - Vᴴ: right singular vectors (temporal patterns)
            3. Keep top-k singular values: S̃ = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᴴ
            4. This is the BEST rank-k approximation (Eckart-Young theorem)
            5. Signal concentrates in top singular values; noise spreads across all

        The rank-k approximation IS a projection onto the k-dimensional
        subspace spanned by the top-k singular vectors. This is the
        most direct application of linear algebra to denoising.

        Args:
            signal_data: Noisy input signal
            sr: Sample rate
            rank_ratio: Fraction of singular values to keep (0.0-1.0)

        Returns:
            DenoiseResult with denoised signal and metadata
        """
        t_start = time.time()
        result = DenoiseResult(None, sr, "SVD Subspace Projection")

        n_fft = 2048
        hop_length = n_fft // 4
        window = scipy_signal.windows.hann(n_fft)

        # ── Step 1: STFT → Spectrogram matrix S ∈ ℂ^(F×T) ──
        freqs, times, stft_matrix = scipy_signal.stft(
            signal_data, fs=sr, window=window,
            nperseg=n_fft, noverlap=n_fft - hop_length
        )
        result.stft_before = np.abs(stft_matrix)
        result.freqs = freqs
        result.times = times

        # ── Step 2: SVD Decomposition: S = U Σ Vᴴ ──
        U, sigma, Vh = svd(stft_matrix, full_matrices=False)

        result.singular_values = sigma.copy()
        total_rank = len(sigma)
        result.total_rank = total_rank

        # ── Step 3: Rank-k approximation ──
        k = max(1, int(rank_ratio * total_rank))
        result.rank_used = k

        # Zero out small singular values → projection
        sigma_truncated = np.zeros_like(sigma)
        sigma_truncated[:k] = sigma[:k]

        # ── Step 4: Reconstruct: S̃ = U · diag(σ̃) · Vᴴ ──
        clean_stft = U @ np.diag(sigma_truncated) @ Vh

        result.stft_after = np.abs(clean_stft)

        # Compute frequency response (ratio of power retained per freq bin)
        power_before = np.mean(np.abs(stft_matrix) ** 2, axis=1)
        power_after = np.mean(np.abs(clean_stft) ** 2, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            result.frequency_response = np.where(
                power_before > 0,
                power_after / power_before,
                0
            )

        # ── Step 5: Inverse STFT ──
        _, denoised = scipy_signal.istft(
            clean_stft, fs=sr, window=window,
            nperseg=n_fft, noverlap=n_fft - hop_length
        )

        min_len = min(len(signal_data), len(denoised))
        denoised = denoised[:min_len]

        result.denoised = denoised
        result.processing_time = time.time() - t_start

        return result

    # ─── Method 3: Wiener Filter ──────────────────────────────────

    @staticmethod
    def wiener_filter(signal_data: np.ndarray, sr: int,
                      noise_segment_duration: float = 0.5,
                      smoothing: float = 0.5) -> DenoiseResult:
        """
        Wiener Filter — Optimal Linear Estimator in Frequency Domain.

        Math:
            1. Estimate noise power: P_N(f) = E[|N(f)|²]
            2. Estimate noisy signal power: P_X(f) = |X(f)|²
            3. Estimate clean signal power: P_S(f) = max(P_X(f) - P_N(f), 0)
            4. Wiener gain: H(f) = P_S(f) / (P_S(f) + P_N(f))
               - This is the MMSE (minimum mean square error) linear filter
               - H(f) → 1 where signal dominates (keep those frequencies)
               - H(f) → 0 where noise dominates (suppress those frequencies)
            5. Apply: Y(f) = H(f) · X(f)
            6. Inverse STFT: y(n) = ISTFT(Y)

        The Wiener filter is the OPTIMAL linear filter — it minimizes
        E[|s(n) - ŝ(n)|²], the mean squared error between the true
        signal and the estimate. This is a weighted projection.

        Args:
            signal_data: Noisy input signal
            sr: Sample rate
            noise_segment_duration: Duration of initial noise sample
            smoothing: Smoothing factor for the Wiener gain (0-1)

        Returns:
            DenoiseResult with denoised signal and metadata
        """
        t_start = time.time()
        result = DenoiseResult(None, sr, "Wiener Filter")

        n_fft = 2048
        hop_length = n_fft // 4
        window = scipy_signal.windows.hann(n_fft)

        # ── Step 1: STFT of the full signal ──
        freqs, times, stft_matrix = scipy_signal.stft(
            signal_data, fs=sr, window=window,
            nperseg=n_fft, noverlap=n_fft - hop_length
        )
        result.stft_before = np.abs(stft_matrix)
        result.freqs = freqs
        result.times = times

        # ── Step 2: Estimate noise power spectrum ──
        noise_samples = int(noise_segment_duration * sr)
        noise_samples = min(noise_samples, len(signal_data) // 4)

        noise_segment = signal_data[:noise_samples]
        _, _, noise_stft = scipy_signal.stft(
            noise_segment, fs=sr, window=window,
            nperseg=n_fft, noverlap=n_fft - hop_length
        )

        noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)
        result.noise_profile = np.sqrt(noise_power.flatten())

        # ── Step 3: Compute Wiener gain H(f) ──
        signal_power = np.abs(stft_matrix) ** 2

        # Estimate clean signal power (spectral subtraction)
        clean_power_est = np.maximum(signal_power - noise_power, 0)

        # Wiener gain: H = P_S / (P_S + P_N)
        wiener_gain = clean_power_est / (clean_power_est + noise_power + 1e-10)

        # Apply smoothing to reduce musical noise
        if smoothing > 0:
            from scipy.ndimage import uniform_filter
            kernel_size = max(1, int(smoothing * 10))
            wiener_gain = uniform_filter(
                wiener_gain, size=(kernel_size, kernel_size)
            )
            wiener_gain = np.clip(wiener_gain, 0, 1)

        result.frequency_response = np.mean(wiener_gain, axis=1)

        # ── Step 4: Apply Wiener filter ──
        clean_stft = wiener_gain * stft_matrix

        result.stft_after = np.abs(clean_stft)

        # ── Step 5: Inverse STFT ──
        _, denoised = scipy_signal.istft(
            clean_stft, fs=sr, window=window,
            nperseg=n_fft, noverlap=n_fft - hop_length
        )

        min_len = min(len(signal_data), len(denoised))
        denoised = denoised[:min_len]

        result.denoised = denoised
        result.processing_time = time.time() - t_start

        return result


# ─── Convenience Function ─────────────────────────────────────────

def denoise(signal_data: np.ndarray, sr: int, method: str = "spectral_gate",
            **kwargs) -> DenoiseResult:
    """
    Convenience function to denoise a signal with the specified method.

    Args:
        signal_data: Noisy input signal
        sr: Sample rate
        method: "spectral_gate", "svd", or "wiener"
        **kwargs: Method-specific parameters

    Returns:
        DenoiseResult
    """
    denoiser = AudioDenoiser()

    if method == "spectral_gate":
        return denoiser.spectral_gate(signal_data, sr, **kwargs)
    elif method == "svd":
        return denoiser.svd_denoise(signal_data, sr, **kwargs)
    elif method == "wiener":
        return denoiser.wiener_filter(signal_data, sr, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
