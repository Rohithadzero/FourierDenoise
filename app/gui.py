"""
FourierDenoise — Main GUI
Premium dark-mode CustomTkinter application for audio signal denoising.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import threading
import os
import sys

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from app.audio_engine import AudioEngine
from app.denoiser import AudioDenoiser
from app.visualizer import AudioVisualizer, COLORS
from app.math_explainer import format_explanation_text

# ─── Theme Setup ──────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

FONT_FAMILY = "Segoe UI"
MONO_FONT = "Consolas"
BG = "#0a0e1a"
PANEL = "#111629"
PANEL_BORDER = "#1e2545"
ACCENT = "#00d4ff"
ACCENT_HOVER = "#33dfff"
MAGENTA = "#ff006e"
GREEN = "#00ff88"
GOLD = "#ffd700"
TEXT = "#e0e4f0"
TEXT_DIM = "#8892b0"
DANGER = "#ff4466"


class FourierDenoiseApp(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.title("FourierDenoise — Audio Signal Denoiser")
        self.geometry("1320x780")
        self.minsize(1100, 650)
        self.configure(fg_color=BG)

        # State
        self.audio_engine = AudioEngine()
        self.denoiser = AudioDenoiser()
        self.original_signal = None
        self.noisy_signal = None
        self.denoised_signal = None
        self.sample_rate = 44100
        self.last_result = None

        self._build_ui()
        self._show_empty_plot()

    # ═══════════════════════════════════════════════════════════════
    #  UI Construction
    # ═══════════════════════════════════════════════════════════════

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=52)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)

        ctk.CTkLabel(header, text="🎵  FourierDenoise",
                     font=(FONT_FAMILY, 18, "bold"),
                     text_color=ACCENT).pack(side="left", padx=20)
        ctk.CTkLabel(header, text="Audio Signal Denoising via Fourier Transform + Linear Algebra",
                     font=(FONT_FAMILY, 11), text_color=TEXT_DIM).pack(side="left", padx=10)

        # Main content
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=10, pady=8)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        # Left sidebar
        self._build_sidebar(content)

        # Right area
        right = ctk.CTkFrame(content, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._build_viz_area(right)
        self._build_status_bar()

    def _build_sidebar(self, parent):
        sidebar = ctk.CTkScrollableFrame(parent, width=265, fg_color=PANEL,
                                          corner_radius=12, border_width=1,
                                          border_color=PANEL_BORDER)
        sidebar.grid(row=0, column=0, sticky="nsew")

        # ── Input Section ──
        self._section_label(sidebar, "INPUT")

        btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=2)

        self._make_btn(btn_frame, "📂  Load Audio", self._load_audio, ACCENT).pack(fill="x", pady=2)
        self._make_btn(btn_frame, "🎙  Record (3s)", self._record_audio, "#6c63ff").pack(fill="x", pady=2)
        self._make_btn(btn_frame, "🎹  Test Signal", self._generate_test, "#ff8c00").pack(fill="x", pady=2)

        self.file_label = ctk.CTkLabel(sidebar, text="No file loaded",
                                        font=(FONT_FAMILY, 10), text_color=TEXT_DIM,
                                        wraplength=230)
        self.file_label.pack(padx=10, pady=(0, 5))

        # ── Noise Section ──
        self._section_label(sidebar, "ADD NOISE")

        noise_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        noise_frame.pack(fill="x", padx=10)

        ctk.CTkLabel(noise_frame, text="Type:", font=(FONT_FAMILY, 10),
                     text_color=TEXT_DIM).pack(anchor="w")
        self.noise_type = ctk.CTkSegmentedButton(
            noise_frame, values=["White", "Pink", "Brown"],
            font=(FONT_FAMILY, 10), selected_color=ACCENT,
            selected_hover_color=ACCENT_HOVER)
        self.noise_type.set("White")
        self.noise_type.pack(fill="x", pady=2)

        ctk.CTkLabel(noise_frame, text="SNR (dB):", font=(FONT_FAMILY, 10),
                     text_color=TEXT_DIM).pack(anchor="w", pady=(4, 0))
        self.snr_slider = ctk.CTkSlider(noise_frame, from_=0, to=30,
                                         number_of_steps=30,
                                         button_color=MAGENTA,
                                         button_hover_color="#ff3388",
                                         progress_color=MAGENTA)
        self.snr_slider.set(10)
        self.snr_slider.pack(fill="x", pady=2)
        self.snr_label = ctk.CTkLabel(noise_frame, text="10 dB",
                                       font=(MONO_FONT, 10), text_color=MAGENTA)
        self.snr_label.pack(anchor="e")
        self.snr_slider.configure(command=lambda v: self.snr_label.configure(
            text=f"{int(v)} dB"))

        self._make_btn(noise_frame, "🔊  Inject Noise", self._add_noise, MAGENTA).pack(fill="x", pady=4)

        # ── Denoise Section ──
        self._section_label(sidebar, "DENOISE METHOD")

        method_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        method_frame.pack(fill="x", padx=10)

        self.method_var = tk.StringVar(value="spectral_gate")
        methods = [
            ("Spectral Gate", "spectral_gate"),
            ("SVD Projection", "svd"),
            ("Wiener Filter", "wiener"),
        ]
        for label, val in methods:
            ctk.CTkRadioButton(method_frame, text=label, variable=self.method_var,
                               value=val, font=(FONT_FAMILY, 11),
                               text_color=TEXT, fg_color=ACCENT,
                               hover_color=ACCENT_HOVER,
                               command=self._on_method_change
                               ).pack(anchor="w", pady=2)

        # Parameter slider
        ctk.CTkLabel(method_frame, text="Strength:", font=(FONT_FAMILY, 10),
                     text_color=TEXT_DIM).pack(anchor="w", pady=(8, 0))
        self.strength_slider = ctk.CTkSlider(method_frame, from_=0.05, to=1.0,
                                              number_of_steps=19,
                                              button_color=GREEN,
                                              button_hover_color="#33ff99",
                                              progress_color=GREEN)
        self.strength_slider.set(0.4)
        self.strength_slider.pack(fill="x", pady=2)
        self.strength_label = ctk.CTkLabel(method_frame, text="0.40",
                                            font=(MONO_FONT, 10), text_color=GREEN)
        self.strength_label.pack(anchor="e")
        self.strength_slider.configure(command=lambda v: self.strength_label.configure(
            text=f"{v:.2f}"))

        self.denoise_btn = self._make_btn(method_frame, "⚡  DENOISE", self._run_denoise, GREEN)
        self.denoise_btn.pack(fill="x", pady=(8, 4))

        # ── Playback Section ──
        self._section_label(sidebar, "PLAYBACK")

        play_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        play_frame.pack(fill="x", padx=10)

        self._make_btn(play_frame, "▶  Play Original / Noisy", self._play_noisy, ACCENT).pack(fill="x", pady=2)
        self._make_btn(play_frame, "▶  Play Denoised", self._play_denoised, GREEN).pack(fill="x", pady=2)
        self._make_btn(play_frame, "⏹  Stop", self._stop_playback, DANGER).pack(fill="x", pady=2)

        # ── Export Section ──
        self._section_label(sidebar, "EXPORT")
        export_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        export_frame.pack(fill="x", padx=10, pady=(0, 10))
        self._make_btn(export_frame, "💾  Save Denoised WAV", self._export_audio, GOLD).pack(fill="x", pady=2)

    def _build_viz_area(self, parent):
        # Tab view for visualizations
        self.tabview = ctk.CTkTabview(parent, fg_color=PANEL,
                                       segmented_button_fg_color=PANEL_BORDER,
                                       segmented_button_selected_color=ACCENT,
                                       segmented_button_selected_hover_color=ACCENT_HOVER,
                                       segmented_button_unselected_color=PANEL,
                                       corner_radius=12, border_width=1,
                                       border_color=PANEL_BORDER)
        self.tabview.grid(row=0, column=0, sticky="nsew", pady=(0, 4))

        tab_waveform = self.tabview.add("Waveform")
        tab_spectrum = self.tabview.add("Spectrum")
        tab_spectro = self.tabview.add("Spectrogram")
        tab_svd = self.tabview.add("SVD Analysis")
        tab_math = self.tabview.add("Math Explained")

        # Matplotlib figures
        self.fig_wave = Figure(figsize=(7, 3.5), dpi=100)
        self.canvas_wave = FigureCanvasTkAgg(self.fig_wave, master=tab_waveform)
        self.canvas_wave.get_tk_widget().pack(fill="both", expand=True)

        self.fig_spec = Figure(figsize=(7, 3.5), dpi=100)
        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=tab_spectrum)
        self.canvas_spec.get_tk_widget().pack(fill="both", expand=True)

        self.fig_spectro = Figure(figsize=(7, 3.5), dpi=100)
        self.canvas_spectro = FigureCanvasTkAgg(self.fig_spectro, master=tab_spectro)
        self.canvas_spectro.get_tk_widget().pack(fill="both", expand=True)

        self.fig_svd = Figure(figsize=(7, 3.5), dpi=100)
        self.canvas_svd = FigureCanvasTkAgg(self.fig_svd, master=tab_svd)
        self.canvas_svd.get_tk_widget().pack(fill="both", expand=True)

        # Math explainer tab (text-based)
        self.math_text = ctk.CTkTextbox(tab_math, font=(MONO_FONT, 11),
                                         fg_color=BG, text_color=TEXT,
                                         border_width=0, wrap="word")
        self.math_text.pack(fill="both", expand=True, padx=5, pady=5)
        self._update_math_text()

    def _build_status_bar(self):
        self.status_bar = ctk.CTkFrame(self, fg_color=PANEL, height=32,
                                        corner_radius=0)
        self.status_bar.pack(fill="x", side="bottom")
        self.status_bar.pack_propagate(False)

        self.status_label = ctk.CTkLabel(self.status_bar, text="Ready — Load an audio file to begin",
                                          font=(FONT_FAMILY, 10), text_color=TEXT_DIM)
        self.status_label.pack(side="left", padx=15)

        self.metrics_label = ctk.CTkLabel(self.status_bar, text="",
                                           font=(MONO_FONT, 10), text_color=GOLD)
        self.metrics_label.pack(side="right", padx=15)

    # ═══════════════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════════════

    def _section_label(self, parent, text):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=10, pady=(12, 4))
        ctk.CTkLabel(frame, text=text, font=(FONT_FAMILY, 9, "bold"),
                     text_color=TEXT_DIM).pack(anchor="w")
        sep = ctk.CTkFrame(frame, height=1, fg_color=PANEL_BORDER)
        sep.pack(fill="x", pady=(2, 0))

    def _make_btn(self, parent, text, command, color):
        return ctk.CTkButton(parent, text=text, command=command,
                              font=(FONT_FAMILY, 12, "bold"),
                              fg_color=color, hover_color=self._lighten(color),
                              text_color="#000000" if color in (GREEN, GOLD, "#ff8c00") else "#ffffff",
                              corner_radius=8, height=34)

    @staticmethod
    def _lighten(hex_color, factor=0.25):
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:], 16)
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _set_status(self, text, color=None):
        self.status_label.configure(text=text, text_color=color or TEXT_DIM)

    def _show_empty_plot(self):
        AudioVisualizer.create_empty_plot(self.fig_wave, "Load an audio file or generate a test signal")
        self.canvas_wave.draw_idle()
        AudioVisualizer.create_empty_plot(self.fig_spec, "Frequency spectrum will appear here")
        self.canvas_spec.draw_idle()
        AudioVisualizer.create_empty_plot(self.fig_spectro, "Spectrogram will appear here")
        self.canvas_spectro.draw_idle()
        AudioVisualizer.create_empty_plot(self.fig_svd, "SVD analysis will appear here")
        self.canvas_svd.draw_idle()

    def _update_math_text(self):
        method = self.method_var.get()
        text = format_explanation_text(method)
        self.math_text.configure(state="normal")
        self.math_text.delete("1.0", "end")
        self.math_text.insert("1.0", text)
        self.math_text.configure(state="disabled")

    def _on_method_change(self):
        self._update_math_text()

    # ═══════════════════════════════════════════════════════════════
    #  Actions
    # ═══════════════════════════════════════════════════════════════

    def _load_audio(self):
        path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.flac *.ogg *.mp3"),
                       ("WAV Files", "*.wav"), ("All Files", "*.*")])
        if not path:
            return

        try:
            self._set_status(f"Loading {os.path.basename(path)}...")
            signal, sr = AudioEngine.load_audio(path)
            self.original_signal = signal
            self.noisy_signal = signal.copy()
            self.sample_rate = sr
            self.denoised_signal = None
            self.last_result = None

            duration = len(signal) / sr
            self.file_label.configure(text=f"✓ {os.path.basename(path)}\n"
                                           f"  {duration:.2f}s • {sr} Hz • {len(signal):,} samples")

            AudioVisualizer.create_initial_plot(self.fig_wave, signal, sr,
                                                f"Loaded: {os.path.basename(path)}")
            self.canvas_wave.draw_idle()
            self.tabview.set("Waveform")

            self._set_status(f"Loaded: {os.path.basename(path)} ({duration:.1f}s)", ACCENT)
            self.metrics_label.configure(text=f"Duration: {duration:.2f}s  |  SR: {sr} Hz")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio:\n{e}")
            self._set_status(f"Error: {e}", DANGER)

    def _record_audio(self):
        if self.audio_engine.is_playing:
            self.audio_engine.stop_audio()

        self._set_status("🎙 Recording... Speak now! (3 seconds)", MAGENTA)
        self.update()

        def on_done(signal, sr):
            if signal is not None:
                self.original_signal = signal
                self.noisy_signal = signal.copy()
                self.sample_rate = sr
                self.denoised_signal = None
                self.last_result = None

                self.after(0, lambda: self._on_record_complete(signal, sr))

        self.audio_engine.record_audio(3.0, 44100, on_complete=on_done)

    def _on_record_complete(self, signal, sr):
        duration = len(signal) / sr
        self.file_label.configure(text=f"✓ Recorded audio\n"
                                       f"  {duration:.2f}s • {sr} Hz • {len(signal):,} samples")
        AudioVisualizer.create_initial_plot(self.fig_wave, signal, sr, "Recorded Audio")
        self.canvas_wave.draw_idle()
        self.tabview.set("Waveform")
        self._set_status("Recording complete!", GREEN)
        self.metrics_label.configure(text=f"Duration: {duration:.2f}s  |  SR: {sr} Hz")

    def _generate_test(self):
        signal = AudioEngine.generate_test_signal(sr=44100, duration=3.0)
        self.original_signal = signal
        self.noisy_signal = signal.copy()
        self.sample_rate = 44100
        self.denoised_signal = None
        self.last_result = None

        self.file_label.configure(text="✓ Test signal (A-major chord)\n"
                                       "  3.00s • 44100 Hz • 132,300 samples")
        AudioVisualizer.create_initial_plot(self.fig_wave, signal, 44100,
                                            "Test Signal: A-Major Chord (440+554+659+880 Hz)")
        self.canvas_wave.draw_idle()
        self.tabview.set("Waveform")
        self._set_status("Generated test signal — try adding noise!", GREEN)
        self.metrics_label.configure(text="Duration: 3.00s  |  SR: 44100 Hz")

    def _add_noise(self):
        if self.original_signal is None:
            messagebox.showwarning("No Audio", "Load an audio file first.")
            return

        noise_type = self.noise_type.get().lower()
        snr_db = self.snr_slider.get()

        self.noisy_signal = AudioEngine.add_noise(
            self.original_signal, noise_type, snr_db)
        self.denoised_signal = None
        self.last_result = None

        AudioVisualizer.create_initial_plot(self.fig_wave, self.noisy_signal,
                                            self.sample_rate,
                                            f"Noisy Signal ({noise_type} noise, SNR={int(snr_db)}dB)")
        self.canvas_wave.draw_idle()
        self.tabview.set("Waveform")

        actual_snr = AudioEngine.compute_snr(self.original_signal, self.noisy_signal)
        self._set_status(f"Added {noise_type} noise (SNR ≈ {actual_snr:.1f} dB)", MAGENTA)

    def _run_denoise(self):
        if self.noisy_signal is None:
            messagebox.showwarning("No Audio", "Load an audio file and add noise first.")
            return

        method = self.method_var.get()
        strength = self.strength_slider.get()
        self._set_status(f"⚡ Denoising with {method.replace('_', ' ').title()}...", GOLD)
        self.denoise_btn.configure(state="disabled", text="Processing...")
        self.update()

        def _process():
            try:
                if method == "spectral_gate":
                    result = self.denoiser.spectral_gate(
                        self.noisy_signal, self.sample_rate,
                        threshold_factor=1.0 + strength * 4.0)
                elif method == "svd":
                    result = self.denoiser.svd_denoise(
                        self.noisy_signal, self.sample_rate,
                        rank_ratio=max(0.05, 1.0 - strength))
                elif method == "wiener":
                    result = self.denoiser.wiener_filter(
                        self.noisy_signal, self.sample_rate,
                        smoothing=strength)
                else:
                    return

                self.denoised_signal = result.denoised
                self.last_result = result

                # Compute SNR metrics
                if self.original_signal is not None:
                    min_len = min(len(self.original_signal), len(self.noisy_signal),
                                 len(self.denoised_signal))
                    result.snr_before = AudioEngine.compute_snr(
                        self.original_signal[:min_len], self.noisy_signal[:min_len])
                    result.snr_after = AudioEngine.compute_snr(
                        self.original_signal[:min_len], self.denoised_signal[:min_len])
                    result.snr_improvement = result.snr_after - result.snr_before

                self.after(0, lambda: self._on_denoise_complete(result))

            except Exception as e:
                self.after(0, lambda: self._on_denoise_error(str(e)))

        threading.Thread(target=_process, daemon=True).start()

    def _on_denoise_complete(self, result):
        self.denoise_btn.configure(state="normal", text="⚡  DENOISE")

        # Update all visualizations
        min_len = min(len(self.noisy_signal), len(result.denoised))
        noisy_trim = self.noisy_signal[:min_len]
        denoised_trim = result.denoised[:min_len]

        AudioVisualizer.create_waveform_plot(
            self.fig_wave, noisy_trim, denoised_trim, self.sample_rate)
        self.canvas_wave.draw_idle()

        noise_profile = result.noise_profile if result.noise_profile is not None else None
        AudioVisualizer.create_spectrum_plot(
            self.fig_spec, noisy_trim, denoised_trim,
            self.sample_rate, noise_profile)
        self.canvas_spec.draw_idle()

        if result.stft_before is not None and result.stft_after is not None:
            AudioVisualizer.create_spectrogram_plot(
                self.fig_spectro, result.stft_before, result.stft_after,
                result.freqs, result.times)
            self.canvas_spectro.draw_idle()

        if result.singular_values is not None:
            AudioVisualizer.create_svd_plot(
                self.fig_svd, result.singular_values, result.rank_used)
            self.canvas_svd.draw_idle()

        # Status and metrics
        method_name = result.method
        t = result.processing_time
        metrics_parts = [f"Method: {method_name}", f"Time: {t:.2f}s"]

        if result.snr_before != 0:
            metrics_parts.append(f"SNR: {result.snr_before:.1f} → {result.snr_after:.1f} dB")
            metrics_parts.append(f"Δ: +{result.snr_improvement:.1f} dB")

        if result.rank_used is not None:
            metrics_parts.append(f"Rank: {result.rank_used}/{result.total_rank}")

        self.metrics_label.configure(text="  |  ".join(metrics_parts))
        self._set_status(f"✓ Denoising complete! Play the result to hear the difference.", GREEN)
        self.tabview.set("Waveform")

    def _on_denoise_error(self, error_msg):
        self.denoise_btn.configure(state="normal", text="⚡  DENOISE")
        messagebox.showerror("Denoise Error", f"Failed:\n{error_msg}")
        self._set_status(f"Error: {error_msg}", DANGER)

    def _play_noisy(self):
        sig = self.noisy_signal if self.noisy_signal is not None else self.original_signal
        if sig is None:
            messagebox.showwarning("No Audio", "Load an audio file first.")
            return
        self._set_status("▶ Playing original/noisy signal...", ACCENT)
        self.audio_engine.play_audio(sig, self.sample_rate,
                                      on_complete=lambda: self.after(0, lambda: self._set_status("Playback complete.", TEXT_DIM)))

    def _play_denoised(self):
        if self.denoised_signal is None:
            messagebox.showwarning("No Result", "Run denoising first.")
            return
        self._set_status("▶ Playing denoised signal...", GREEN)
        self.audio_engine.play_audio(self.denoised_signal, self.sample_rate,
                                      on_complete=lambda: self.after(0, lambda: self._set_status("Playback complete.", TEXT_DIM)))

    def _stop_playback(self):
        self.audio_engine.stop_audio()
        self._set_status("⏹ Playback stopped.", TEXT_DIM)

    def _export_audio(self):
        if self.denoised_signal is None:
            messagebox.showwarning("No Result", "Run denoising first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save Denoised Audio",
            defaultextension=".wav",
            filetypes=[("WAV File", "*.wav")])
        if not path:
            return

        try:
            AudioEngine.save_audio(path, self.denoised_signal, self.sample_rate)
            self._set_status(f"✓ Saved: {os.path.basename(path)}", GREEN)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{e}")
