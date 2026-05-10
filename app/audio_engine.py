"""
FourierDenoise — Audio Engine
Handles loading, saving, playing, recording, and manipulating audio signals.
"""

import numpy as np
import soundfile as sf
import sounddevice as sd
import threading
import os


class AudioEngine:
    """Core audio I/O and manipulation engine."""

    def __init__(self):
        self.current_playback = None
        self._stop_event = threading.Event()
        self.is_playing = False
        self.playback_callback = None  # GUI callback for playback position

    # ─── Load / Save ──────────────────────────────────────────────

    @staticmethod
    def load_audio(filepath: str) -> tuple:
        """
        Load an audio file and return (signal, sample_rate).
        Converts stereo to mono and normalizes to [-1, 1].
        """
        signal, sr = sf.read(filepath, dtype='float64')

        # Convert stereo to mono
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)

        # Normalize
        peak = np.max(np.abs(signal))
        if peak > 0:
            signal = signal / peak

        return signal, sr

    @staticmethod
    def save_audio(filepath: str, signal: np.ndarray, sr: int):
        """Save audio signal to a WAV file."""
        # Normalize before saving
        peak = np.max(np.abs(signal))
        if peak > 0:
            signal = signal / peak * 0.95  # Leave small headroom

        sf.write(filepath, signal, sr, subtype='PCM_16')

    # ─── Playback ─────────────────────────────────────────────────

    def play_audio(self, signal: np.ndarray, sr: int, on_complete=None):
        """Play audio signal in a non-blocking thread."""
        self.stop_audio()
        self._stop_event.clear()
        self.is_playing = True

        def _play():
            try:
                # Use a small blocksize for responsive stopping
                blocksize = 1024
                stream = sd.OutputStream(
                    samplerate=sr,
                    channels=1,
                    dtype='float32',
                    blocksize=blocksize
                )
                stream.start()

                signal_f32 = signal.astype(np.float32)
                total_samples = len(signal_f32)
                pos = 0

                while pos < total_samples and not self._stop_event.is_set():
                    end = min(pos + blocksize, total_samples)
                    chunk = signal_f32[pos:end]
                    stream.write(chunk.reshape(-1, 1))

                    # Report position
                    if self.playback_callback:
                        progress = pos / total_samples
                        self.playback_callback(progress)

                    pos = end

                stream.stop()
                stream.close()
            except Exception as e:
                print(f"Playback error: {e}")
            finally:
                self.is_playing = False
                if on_complete and not self._stop_event.is_set():
                    on_complete()

        thread = threading.Thread(target=_play, daemon=True)
        thread.start()

    def stop_audio(self):
        """Stop any currently playing audio."""
        self._stop_event.set()
        self.is_playing = False
        try:
            sd.stop()
        except Exception:
            pass

    # ─── Recording ────────────────────────────────────────────────

    def record_audio(self, duration: float, sr: int = 44100,
                     on_complete=None) -> np.ndarray:
        """
        Record audio from microphone for given duration.
        Returns the recorded signal.
        """
        self._stop_event.clear()

        def _record():
            try:
                recording = sd.rec(
                    int(duration * sr),
                    samplerate=sr,
                    channels=1,
                    dtype='float64'
                )
                sd.wait()
                signal = recording.flatten()

                # Normalize
                peak = np.max(np.abs(signal))
                if peak > 0:
                    signal = signal / peak

                if on_complete:
                    on_complete(signal, sr)

                return signal
            except Exception as e:
                print(f"Recording error: {e}")
                if on_complete:
                    on_complete(None, sr)
                return None

        thread = threading.Thread(target=_record, daemon=True)
        thread.start()

    # ─── Noise Injection ──────────────────────────────────────────

    @staticmethod
    def add_noise(signal: np.ndarray, noise_type: str = "white",
                  snr_db: float = 10.0) -> np.ndarray:
        """
        Add synthetic noise to a clean signal.

        Args:
            signal: Clean audio signal
            noise_type: "white", "pink", or "brown"
            snr_db: Signal-to-noise ratio in dB (lower = more noise)

        Returns:
            Noisy signal
        """
        n = len(signal)

        if noise_type == "white":
            noise = np.random.randn(n)

        elif noise_type == "pink":
            # Pink noise: 1/f spectrum
            # Generate white noise in frequency domain, shape with 1/sqrt(f)
            white = np.fft.rfft(np.random.randn(n))
            freqs = np.fft.rfftfreq(n)
            freqs[0] = 1  # Avoid division by zero
            pink_filter = 1.0 / np.sqrt(freqs)
            pink_filter[0] = 0
            noise = np.fft.irfft(white * pink_filter, n=n)

        elif noise_type == "brown":
            # Brown noise: 1/f² spectrum (cumulative sum of white noise)
            noise = np.cumsum(np.random.randn(n))
            noise = noise - np.mean(noise)

        else:
            noise = np.random.randn(n)

        # Normalize noise
        noise = noise / (np.max(np.abs(noise)) + 1e-10)

        # Scale noise to achieve desired SNR
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power > 0 and signal_power > 0:
            snr_linear = 10 ** (snr_db / 10)
            scale = np.sqrt(signal_power / (snr_linear * noise_power))
            noise = noise * scale

        noisy = signal + noise

        # Normalize to prevent clipping
        peak = np.max(np.abs(noisy))
        if peak > 1.0:
            noisy = noisy / peak

        return noisy

    # ─── Utilities ────────────────────────────────────────────────

    @staticmethod
    def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
        """Compute Signal-to-Noise Ratio in dB."""
        noise = noisy - clean
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power < 1e-10:
            return float('inf')

        return 10 * np.log10(signal_power / noise_power)

    @staticmethod
    def estimate_snr(signal: np.ndarray) -> float:
        """
        Estimate SNR from a single signal using spectral methods.
        Assumes noise is relatively flat in spectrum.
        """
        spectrum = np.abs(np.fft.rfft(signal))
        sorted_spec = np.sort(spectrum)

        # Estimate noise floor as median of lower half of spectrum
        n = len(sorted_spec)
        noise_floor = np.mean(sorted_spec[:n // 4]) ** 2
        signal_power = np.mean(spectrum ** 2)

        if noise_floor < 1e-10:
            return float('inf')

        return 10 * np.log10((signal_power - noise_floor) / noise_floor)

    @staticmethod
    def get_duration(signal: np.ndarray, sr: int) -> float:
        """Get duration in seconds."""
        return len(signal) / sr

    @staticmethod
    def generate_test_signal(sr: int = 44100, duration: float = 3.0) -> np.ndarray:
        """
        Generate a clean test signal: sum of sine waves (like a chord).
        Perfect for demonstrating Fourier decomposition.
        """
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # Musical chord: A4 (440Hz), C#5 (554Hz), E5 (659Hz)
        signal = (
            0.5 * np.sin(2 * np.pi * 440 * t) +     # A4
            0.3 * np.sin(2 * np.pi * 554 * t) +     # C#5
            0.2 * np.sin(2 * np.pi * 659 * t) +     # E5
            0.15 * np.sin(2 * np.pi * 880 * t)      # A5 (octave)
        )

        # Add gentle envelope (fade in/out)
        envelope = np.ones_like(t)
        fade_len = int(0.1 * sr)
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
        signal *= envelope

        # Normalize
        signal = signal / np.max(np.abs(signal))

        return signal
