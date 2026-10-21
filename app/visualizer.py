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
    'figure':     '#111629',  # matches the GUI panel so plots blend into their tab
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
    fig.patch.set_facecolor(COLORS['figure'])

    axes = ax_or_axes if hasattr(ax_or_axes, '__iter__') else [ax_or_axes]

    for ax in axes:
        ax.set_facecolor(COLORS['panel'])
        ax.tick_params(colors=COLORS['text_dim'], labelsize=10)
        ax.xaxis.label.set_color(COLORS['text_dim'])
        ax.yaxis.label.set_color(COLORS['text_dim'])
        ax.title.set_color(COLORS['text'])
        ax.title.set_fontsize(12)
        ax.title.set_fontweight('bold')

        for spine in ax.spines.values():
            spine.set_color(COLORS['grid'])
            spine.set_linewidth(0.5)

        ax.grid(True, alpha=0.15, color=COLORS['grid'], linewidth=0.5)


def draw_waveform(ax, signal: np.ndarray, sr: int, color: str,
                  alpha: float, label: str = None, columns: int = 2000):
    """
    Draw a waveform as a min/max envelope over ~`columns` time slices.

    Plotting every sample of a long file overflows Agg's path renderer
    ("Exceeded cell block limit") and takes seconds even when it doesn't;
    the envelope is what is visible at screen resolution anyway.
    """
    n = len(signal)
    if n <= 2 * columns:
        ax.plot(np.arange(n) / sr, signal, color=color, alpha=alpha,
                linewidth=0.6, label=label)
        return

    per_col = n // columns
    trimmed = signal[:per_col * columns].reshape(columns, per_col)
    t = (np.arange(columns) + 0.5) * per_col / sr
    ax.fill_between(t, trimmed.min(axis=1), trimmed.max(axis=1),
                    color=color, alpha=alpha, linewidth=0, label=label)


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

        # Noisy signal (translucent red) behind the denoised signal (solid green)
        draw_waveform(ax, noisy, sr, COLORS['noisy'], 0.35, 'Noisy Signal')
        draw_waveform(ax, denoised, sr, COLORS['denoised'], 0.8, 'Denoised Signal')

        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Amplitude', fontsize=11)
        ax.set_title('⟨ Waveform Comparison ⟩', fontsize=13, pad=8)

        legend = ax.legend(loc='upper right', fontsize=10,
                          framealpha=0.3, edgecolor=COLORS['grid'])
        for text in legend.get_texts():
            text.set_color(COLORS['text'])

        ax.set_xlim(0, max(len(noisy), len(denoised)) / sr)
        ax.set_ylim(-1.1, 1.1)

        fig.tight_layout(pad=0.8)

    @staticmethod
    def create_spectrum_plot(fig: Figure, noisy: np.ndarray,
                            denoised: np.ndarray, sr: int,
                            noise_profile: np.ndarray = None):
        """
        Plot frequency spectrum (FFT magnitude) for noisy and denoised signals.
        With denoised=None, shows only the input signal's spectrum.
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

        # Smooth for visualization
        from scipy.ndimage import uniform_filter1d
        smooth_size = max(1, len(freqs) // 500)
        spec_noisy_smooth = uniform_filter1d(spec_noisy, smooth_size)

        # The curve is already smoothed over smooth_size bins, so plotting every bin only
        # costs render time (hundreds of thousands of points for longer files)
        step = max(1, smooth_size // 2)
        freqs = freqs[::step]
        spec_noisy_smooth = spec_noisy_smooth[::step]

        if denoised is None:
            input_color, input_label, input_alpha = COLORS['cyan'], 'Input Signal', 0.9
        else:
            input_color, input_label, input_alpha = COLORS['noisy'], 'Noisy', 0.6

        ax.fill_between(freqs, spec_noisy_smooth, -100,
                        color=input_color, alpha=0.15)
        ax.plot(freqs, spec_noisy_smooth, color=input_color,
                alpha=input_alpha, linewidth=0.7, label=input_label)
        y_min, y_max = np.min(spec_noisy_smooth), np.max(spec_noisy_smooth)

        if denoised is not None:
            spec_denoised = 20 * np.log10(np.abs(np.fft.rfft(denoised)) + 1e-10)
            spec_denoised_smooth = uniform_filter1d(spec_denoised, smooth_size)[::step]
            ax.fill_between(freqs, spec_denoised_smooth, -100,
                            color=COLORS['denoised'], alpha=0.1)
            ax.plot(freqs, spec_denoised_smooth, color=COLORS['denoised'],
                    alpha=0.9, linewidth=0.8, label='Denoised')
            y_min = min(y_min, np.min(spec_denoised_smooth))
            y_max = max(y_max, np.max(spec_denoised_smooth))

        # Noise floor
        if noise_profile is not None:
            # noise_profile is a per-frame STFT magnitude (scipy scales by 1/Σw, Hann window);
            # rescale to the full-signal rfft magnitude plotted above: × √N · Σw / √Σw²
            n_fft = 2 * (len(noise_profile) - 1)
            win = np.hanning(n_fft)
            scale = np.sqrt(n) * win.sum() / np.sqrt(np.sum(win ** 2))
            noise_db = 20 * np.log10(noise_profile * scale + 1e-10)
            noise_freqs = np.linspace(0, sr/2, len(noise_db))
            ax.plot(noise_freqs, noise_db, color=COLORS['gold'],
                    alpha=0.7, linewidth=1.2, linestyle='--',
                    label='Noise Floor')

        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('Magnitude (dB)', fontsize=11)
        ax.set_title('⟨ Frequency Spectrum — FFT Magnitude ⟩', fontsize=13, pad=8)

        # Limit to audible range
        ax.set_xlim(0, min(sr / 2, 10000))
        # FFT magnitude grows with signal length, so fit the axis to the data (90 dB range max)
        ax.set_ylim(max(y_min, y_max - 90), y_max + 5)

        legend = ax.legend(loc='upper right', fontsize=10,
                          framealpha=0.3, edgecolor=COLORS['grid'])
        for text in legend.get_texts():
            text.set_color(COLORS['text'])

        fig.tight_layout(pad=0.8)

    @staticmethod
    def create_spectrogram_plot(fig: Figure,
                                stft_before: np.ndarray,
                                stft_after: np.ndarray,
                                freqs: np.ndarray,
                                times: np.ndarray):
        """
        Side-by-side spectrograms: before and after denoising.
        With stft_after=None, shows a single spectrogram of the input signal.
        Shows the time-frequency representation.
        """
        fig.clear()
        # Dedicated colorbar column so tight_layout can place it without overlapping the plots
        if stft_after is None:
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.02])
            ax1 = fig.add_subplot(gs[0])
            ax2 = None
            cax = fig.add_subplot(gs[1])
            apply_dark_theme(fig, ax1)
        else:
            gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.04])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharey=ax1)
            cax = fig.add_subplot(gs[2])
            ax2.tick_params(labelleft=False)
            apply_dark_theme(fig, [ax1, ax2])

        # Convert to dB scale
        vmin = -60
        vmax = 0

        spec_before_db = 20 * np.log10(stft_before + 1e-10)
        spec_before_db = spec_before_db - np.max(spec_before_db)

        # Limit frequency range for display
        max_freq_idx = np.searchsorted(freqs, min(freqs[-1], 8000))

        im1 = ax1.pcolormesh(
            times, freqs[:max_freq_idx], spec_before_db[:max_freq_idx],
            shading='gouraud', cmap='magma', vmin=vmin, vmax=vmax
        )
        ax1.set_title('Input Signal' if ax2 is None else 'Before (Noisy)', fontsize=12, pad=6)
        ax1.set_xlabel('Time (s)', fontsize=10)
        ax1.set_ylabel('Frequency (Hz)', fontsize=10)

        if ax2 is not None:
            spec_after_db = 20 * np.log10(stft_after + 1e-10)
            spec_after_db = spec_after_db - np.max(spec_after_db)
            ax2.pcolormesh(
                times, freqs[:max_freq_idx], spec_after_db[:max_freq_idx],
                shading='gouraud', cmap='magma', vmin=vmin, vmax=vmax
            )
            ax2.set_title('After (Denoised)', fontsize=12, pad=6)
            ax2.set_xlabel('Time (s)', fontsize=10)

        # Colorbar (both panels use the same fixed dB scale)
        cbar = fig.colorbar(im1, cax=cax)
        cbar.set_label('Power (dB)', color=COLORS['text_dim'], fontsize=10)
        cbar.ax.tick_params(colors=COLORS['text_dim'], labelsize=9)

        fig.suptitle('⟨ Spectrogram — Time-Frequency Analysis ⟩',
                     color=COLORS['text'], fontsize=13, fontweight='bold', y=0.98)
        fig.tight_layout(pad=0.8, rect=[0, 0, 1, 0.95])

    @staticmethod
    def create_svd_plot(fig: Figure, singular_values: np.ndarray,
                        rank_used: int = None):
        """
        Bar chart of singular values with the rank-k cutoff.
        With rank_used=None (before denoising), shows all values with no cutoff.
        Shows how signal energy concentrates in top singular values.
        """
        fig.clear()
        ax = fig.add_subplot(111)
        apply_dark_theme(fig, ax)

        n = len(singular_values)
        preview = rank_used is None
        if preview:
            rank_used = n
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

        if preview:
            top_k = min(10, n)
            top_pct = (np.sum(singular_values[:top_k] ** 2) / total_energy * 100) if total_energy > 0 else 0
            info = f'Top {top_k} values hold {top_pct:.1f}% of energy\nFull rank: {n}'
        else:
            info = f'Energy retained: {pct:.1f}%\nRank: {rank_used}/{n}'
        ax.text(0.98, 0.95, info,
                transform=ax.transAxes, fontsize=10,
                color=COLORS['gold'], ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.4',
                         facecolor=COLORS['bg'], alpha=0.8,
                         edgecolor=COLORS['gold']))

        ax.set_xlabel('Singular Value Index', fontsize=11)
        ax.set_ylabel('σᵢ (Magnitude)', fontsize=11)
        ax.set_title('⟨ SVD Singular Values — Subspace Analysis ⟩',
                     fontsize=13, pad=8)

        # The only labelled artist is the cutoff line, which is absent when every bar is kept
        if ax.get_legend_handles_labels()[0]:
            legend = ax.legend(loc='upper left', fontsize=10,
                              framealpha=0.3, edgecolor=COLORS['grid'])
            for text in legend.get_texts():
                text.set_color(COLORS['text'])

        fig.tight_layout(pad=0.8)

    @staticmethod
    def create_initial_plot(fig: Figure, signal: np.ndarray, sr: int,
                           title: str = "Loaded Signal"):
        """Create a simple waveform plot for a single signal."""
        fig.clear()
        ax = fig.add_subplot(111)
        apply_dark_theme(fig, ax)

        draw_waveform(ax, signal, sr, COLORS['cyan'], 0.8)

        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Amplitude', fontsize=11)
        ax.set_title(f'⟨ {title} ⟩', fontsize=13, pad=8)
        ax.set_xlim(0, len(signal) / sr)
        ax.set_ylim(-1.1, 1.1)

        # Show duration and sample rate
        duration = len(signal) / sr
        ax.text(0.98, 0.95,
                f'Duration: {duration:.2f}s\nSample Rate: {sr} Hz\nSamples: {len(signal):,}',
                transform=ax.transAxes, fontsize=10,
                color=COLORS['text_dim'], ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.4',
                         facecolor=COLORS['bg'], alpha=0.8,
                         edgecolor=COLORS['grid']))

        fig.tight_layout(pad=0.8)

    @staticmethod
    def create_empty_plot(fig: Figure, message: str = "Load an audio file to begin"):
        """Create a placeholder plot with a message."""
        fig.clear()
        ax = fig.add_subplot(111)
        apply_dark_theme(fig, ax)
        ax.set_facecolor(COLORS['figure'])
        ax.grid(False)

        ax.text(0.5, 0.5, message,
                transform=ax.transAxes, fontsize=15,
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

        fig.tight_layout(pad=0.4)
