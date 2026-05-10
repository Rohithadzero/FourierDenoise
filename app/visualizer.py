"""
FourierDenoise — Visualization Engine
Creates matplotlib plots for waveforms, spectra, spectrograms, and SVD values.
All plots use a dark theme matching the application's aesthetic.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend first
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as ticker


# ─── Dark Theme Configuration ────────────────────────────────────

COLORS = {
    'bg':         '#0a0e1a',
    'panel':      '#1a1f35',
    'grid':       '#2a3050',
    'text':       '#e0e4f0',
    'text_dim':   '#8892b0',
    'cyan':       '#00d4ff',
    'magenta':    '#ff006e',
    'green':      '#00ff88',
    'gold':       '#ffd700',
    'orange':     '#ff8c00',
    'purple':     '#b388ff',
    'noisy':      '#ff4466',
    'clean':      '#00d4ff',
    'denoised':   '#00ff88',
}


def apply_dark_theme(fig, ax_or_axes):
    """Apply the dark theme to figure and axes."""
    fig.patch.set_facecolor(COLORS['bg'])

    axes = ax_or_axes if hasattr(ax_or_axes, '__iter__') else [ax_or_axes]

    for ax in axes:
        ax.set_facecolor(COLORS['panel'])
        ax.tick_params(colors=COLORS['text_dim'], labelsize=8)
        ax.xaxis.label.set_color(COLORS['text_dim'])
        ax.yaxis.label.set_color(COLORS['text_dim'])
        ax.title.set_color(COLORS['text'])
        ax.title.set_fontsize(10)
        ax.title.set_fontweight('bold')

        for spine in ax.spines.values():
            spine.set_color(COLORS['grid'])
            spine.set_linewidth(0.5)

        ax.grid(True, alpha=0.15, color=COLORS['grid'], linewidth=0.5)


class AudioVisualizer:
    """Creates embedded matplotlib visualizations for the GUI."""

    @staticmethod
    def create_waveform_plot(fig: Figure, noisy: np.ndarray,
                             denoised: np.ndarray, sr: int):
        """
        Plot time-domain waveforms: noisy (red) vs denoised (green).
        Shows the before/after comparison.
        """
        fig.clear()
        ax = fig.add_subplot(111)
        apply_dark_theme(fig, ax)

        t_noisy = np.arange(len(noisy)) / sr
        t_denoised = np.arange(len(denoised)) / sr

        # Plot noisy signal (translucent red)
        ax.plot(t_noisy, noisy, color=COLORS['noisy'],
                alpha=0.35, linewidth=0.4, label='Noisy Signal')

        # Plot denoised signal (solid green)
        ax.plot(t_denoised, denoised, color=COLORS['denoised'],
                alpha=0.9, linewidth=0.5, label='Denoised Signal')

        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=9)
        ax.set_title('⟨ Waveform Comparison ⟩', fontsize=11, pad=10)

        legend = ax.legend(loc='upper right', fontsize=8,
                          framealpha=0.3, edgecolor=COLORS['grid'])
        for text in legend.get_texts():
            text.set_color(COLORS['text'])

        ax.set_xlim(0, max(t_noisy[-1], t_denoised[-1]))
        ax.set_ylim(-1.1, 1.1)

        fig.tight_layout(pad=1.5)

    @staticmethod
    def create_spectrum_plot(fig: Figure, noisy: np.ndarray,
                            denoised: np.ndarray, sr: int,
                            noise_profile: np.ndarray = None):
        """
        Plot frequency spectrum (FFT magnitude) for noisy and denoised signals.
        Optionally shows the estimated noise floor.
        """
        fig.clear()
        ax = fig.add_subplot(111)
        apply_dark_theme(fig, ax)

        # Compute FFT
        n = len(noisy)
        freqs = np.fft.rfftfreq(n, d=1/sr)

        # Magnitude spectrum (in dB)
        spec_noisy = 20 * np.log10(np.abs(np.fft.rfft(noisy)) + 1e-10)
        spec_denoised = 20 * np.log10(np.abs(np.fft.rfft(denoised)) + 1e-10)

        # Smooth for visualization
        from scipy.ndimage import uniform_filter1d
        smooth_size = max(1, len(freqs) // 500)
        spec_noisy_smooth = uniform_filter1d(spec_noisy, smooth_size)
        spec_denoised_smooth = uniform_filter1d(spec_denoised, smooth_size)

        ax.fill_between(freqs, spec_noisy_smooth, -100,
                        color=COLORS['noisy'], alpha=0.15)
        ax.plot(freqs, spec_noisy_smooth, color=COLORS['noisy'],
                alpha=0.6, linewidth=0.7, label='Noisy')

        ax.fill_between(freqs, spec_denoised_smooth, -100,
                        color=COLORS['denoised'], alpha=0.1)
        ax.plot(freqs, spec_denoised_smooth, color=COLORS['denoised'],
                alpha=0.9, linewidth=0.8, label='Denoised')

        # Noise floor
        if noise_profile is not None:
            noise_db = 20 * np.log10(noise_profile + 1e-10)
            noise_freqs = np.linspace(0, sr/2, len(noise_db))
            ax.plot(noise_freqs, noise_db, color=COLORS['gold'],
                    alpha=0.7, linewidth=1.2, linestyle='--',
                    label='Noise Floor')

        ax.set_xlabel('Frequency (Hz)', fontsize=9)
        ax.set_ylabel('Magnitude (dB)', fontsize=9)
        ax.set_title('⟨ Frequency Spectrum — FFT Magnitude ⟩', fontsize=11, pad=10)

        # Limit to audible range
        ax.set_xlim(0, min(sr / 2, 10000))
        y_min = min(np.min(spec_denoised_smooth), np.min(spec_noisy_smooth))
        ax.set_ylim(max(y_min, -80), 5)

        legend = ax.legend(loc='upper right', fontsize=8,
                          framealpha=0.3, edgecolor=COLORS['grid'])
        for text in legend.get_texts():
            text.set_color(COLORS['text'])

        fig.tight_layout(pad=1.5)

    @staticmethod
    def create_spectrogram_plot(fig: Figure,
                                stft_before: np.ndarray,
                                stft_after: np.ndarray,
                                freqs: np.ndarray,
                                times: np.ndarray):
        """
        Side-by-side spectrograms: before and after denoising.
        Shows the time-frequency representation.
        """
        fig.clear()
        ax1, ax2 = fig.subplots(1, 2, sharey=True)
        apply_dark_theme(fig, [ax1, ax2])

        # Convert to dB scale
        vmin = -60
        vmax = 0

        spec_before_db = 20 * np.log10(stft_before + 1e-10)
        spec_before_db = spec_before_db - np.max(spec_before_db)

        spec_after_db = 20 * np.log10(stft_after + 1e-10)
        spec_after_db = spec_after_db - np.max(spec_after_db)

        # Limit frequency range for display
        max_freq_idx = np.searchsorted(freqs, min(freqs[-1], 8000))

        im1 = ax1.pcolormesh(
            times, freqs[:max_freq_idx], spec_before_db[:max_freq_idx],
            shading='gouraud', cmap='magma', vmin=vmin, vmax=vmax
        )
        ax1.set_title('Before (Noisy)', fontsize=10, pad=8)
        ax1.set_xlabel('Time (s)', fontsize=8)
        ax1.set_ylabel('Frequency (Hz)', fontsize=8)

        im2 = ax2.pcolormesh(
            times, freqs[:max_freq_idx], spec_after_db[:max_freq_idx],
            shading='gouraud', cmap='magma', vmin=vmin, vmax=vmax
        )
        ax2.set_title('After (Denoised)', fontsize=10, pad=8)
        ax2.set_xlabel('Time (s)', fontsize=8)

        # Colorbar
        cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, pad=0.02)
        cbar.set_label('Power (dB)', color=COLORS['text_dim'], fontsize=8)
        cbar.ax.tick_params(colors=COLORS['text_dim'], labelsize=7)

        fig.suptitle('⟨ Spectrogram — Time-Frequency Analysis ⟩',
                     color=COLORS['text'], fontsize=11, fontweight='bold', y=0.98)
        fig.tight_layout(pad=1.5, rect=[0, 0, 1, 0.95])

    @staticmethod
    def create_svd_plot(fig: Figure, singular_values: np.ndarray,
                        rank_used: int):
        """
        Bar chart of singular values with the rank-k cutoff.
        Shows how signal energy concentrates in top singular values.
        """
        fig.clear()
        ax = fig.add_subplot(111)
        apply_dark_theme(fig, ax)

        n = len(singular_values)
        display_n = min(n, 80)  # Show at most 80 bars
        sv = singular_values[:display_n]
        indices = np.arange(display_n)

        # Color bars: green for kept, red for discarded
        colors = [COLORS['cyan'] if i < rank_used else COLORS['noisy']
                  for i in range(display_n)]
        alphas = [0.9 if i < rank_used else 0.3 for i in range(display_n)]

        bars = ax.bar(indices, sv, color=colors, width=0.8)
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)

        # Cutoff line
        if rank_used < display_n:
            ax.axvline(x=rank_used - 0.5, color=COLORS['gold'],
                      linewidth=2, linestyle='--', alpha=0.8,
                      label=f'Rank cutoff (k={rank_used})')

        # Energy retained
        total_energy = np.sum(singular_values ** 2)
        kept_energy = np.sum(singular_values[:rank_used] ** 2)
        pct = (kept_energy / total_energy * 100) if total_energy > 0 else 0

        ax.text(0.98, 0.95,
                f'Energy retained: {pct:.1f}%\nRank: {rank_used}/{n}',
                transform=ax.transAxes, fontsize=9,
                color=COLORS['gold'], ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.4',
                         facecolor=COLORS['bg'], alpha=0.8,
                         edgecolor=COLORS['gold']))

        ax.set_xlabel('Singular Value Index', fontsize=9)
        ax.set_ylabel('σᵢ (Magnitude)', fontsize=9)
        ax.set_title('⟨ SVD Singular Values — Subspace Analysis ⟩',
                     fontsize=11, pad=10)

        legend = ax.legend(loc='upper right', fontsize=8,
                          framealpha=0.3, edgecolor=COLORS['grid'])
        if legend:
            for text in legend.get_texts():
                text.set_color(COLORS['text'])

        fig.tight_layout(pad=1.5)

    @staticmethod
    def create_initial_plot(fig: Figure, signal: np.ndarray, sr: int,
                           title: str = "Loaded Signal"):
        """Create a simple waveform plot for a single signal."""
        fig.clear()
        ax = fig.add_subplot(111)
        apply_dark_theme(fig, ax)

        t = np.arange(len(signal)) / sr
        ax.plot(t, signal, color=COLORS['cyan'], alpha=0.8, linewidth=0.5)
        ax.fill_between(t, signal, 0, color=COLORS['cyan'], alpha=0.1)

        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=9)
        ax.set_title(f'⟨ {title} ⟩', fontsize=11, pad=10)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-1.1, 1.1)

        # Show duration and sample rate
        duration = len(signal) / sr
        ax.text(0.98, 0.95,
                f'Duration: {duration:.2f}s\nSample Rate: {sr} Hz\nSamples: {len(signal):,}',
                transform=ax.transAxes, fontsize=8,
                color=COLORS['text_dim'], ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.4',
                         facecolor=COLORS['bg'], alpha=0.8,
                         edgecolor=COLORS['grid']))

        fig.tight_layout(pad=1.5)

    @staticmethod
    def create_empty_plot(fig: Figure, message: str = "Load an audio file to begin"):
        """Create a placeholder plot with a message."""
        fig.clear()
        ax = fig.add_subplot(111)
        apply_dark_theme(fig, ax)

        ax.text(0.5, 0.5, message,
                transform=ax.transAxes, fontsize=14,
                color=COLORS['text_dim'], ha='center', va='center',
                fontfamily='sans-serif')

        ax.text(0.5, 0.38, '~ ~ ~',
                transform=ax.transAxes, fontsize=24,
                color=COLORS['cyan'], alpha=0.4,
                ha='center', va='center')

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.tight_layout()
