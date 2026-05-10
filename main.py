"""
FourierDenoise — Audio Signal Denoising via Fourier + Linear Algebra
Entry point for the application.

Demonstrates Vector Spaces, Orthogonal Bases, Projection,
and Fourier Transform as a Linear Map to denoise audio signals.
"""

import sys
import os

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.gui import FourierDenoiseApp


def main():
    print("  FourierDenoise — Audio Signal Denoiser")
    print("  Fourier Transform + Linear Algebra\n")
    print("-"*55)
    print("  Starting application...")

    app = FourierDenoiseApp()
    app.mainloop()


if __name__ == "__main__":
    main()

