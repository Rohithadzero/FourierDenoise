"""
FourierDenoise — Math Explainer
Provides educational explanations of the linear algebra behind each method.
Uses Unicode math symbols for rendering in the GUI text panel.
"""


MATH_EXPLANATIONS = {
    "spectral_gate": {
        "title": "Spectral Gate — Threshold Projection",
        "concept": "Projection onto a Clean Subspace",
        "steps": [
            (
                "Vector Space Setup",
                "The audio signal x ∈ ℝⁿ lives in an n-dimensional\n"
                "vector space, where n = number of samples.\n"
                "Each sample x[i] is a coordinate in this space."
            ),
            (
                "Fourier Transform (Linear Map)",
                "The DFT is a linear map  F : ℝⁿ → ℂⁿ\n\n"
                "   x̂ = F · x\n\n"
                "where F is the DFT matrix with entries:\n"
                "   F[k,n] = e^(-j·2π·k·n/N)\n\n"
                "The columns of F form an ORTHONORMAL BASIS\n"
                "for ℂⁿ — every signal decomposes uniquely\n"
                "into frequency components."
            ),
            (
                "STFT: Windowed Analysis",
                "We apply the DFT to overlapping windows:\n\n"
                "   X(t,f) = Σₙ x(n)·w(n-t)·e^(-j2πfn/N)\n\n"
                "This gives us a matrix X ∈ ℂ^(F×T)\n"
                "(the spectrogram — a time-frequency map)."
            ),
            (
                "Noise Estimation",
                "From a quiet segment, estimate the noise floor:\n\n"
                "   N(f) = 𝔼[|X_noise(t,f)|²]\n\n"
                "This tells us the average noise power at\n"
                "each frequency bin."
            ),
            (
                "Projection (The Key Step!)",
                "Build a binary gate mask:\n\n"
                "   G(t,f) = 1  if |X(t,f)|² > α·N(f)\n"
                "   G(t,f) = 0  otherwise\n\n"
                "This is a PROJECTION OPERATOR — it projects\n"
                "the signal onto the subspace spanned by\n"
                "frequency components above the noise floor.\n\n"
                "   Y = G ⊙ X    (element-wise product)"
            ),
            (
                "Reconstruction",
                "Apply the INVERSE linear map  F⁻¹ : ℂⁿ → ℝⁿ\n\n"
                "   y = F⁻¹ · Ŷ\n\n"
                "Since F is unitary (orthonormal), F⁻¹ = Fᴴ\n"
                "(conjugate transpose). The denoised signal\n"
                "lives in the clean subspace of ℝⁿ."
            ),
        ],
        "key_insight": (
            "💡 KEY INSIGHT\n\n"
            "Noise spreads across ALL frequencies uniformly.\n"
            "Signal concentrates in SPECIFIC frequency bands.\n\n"
            "By zeroing frequency bins below the noise floor,\n"
            "we PROJECT the signal onto a lower-dimensional\n"
            "subspace where only the signal lives — removing\n"
            "the noise components from the vector decomposition."
        )
    },

    "svd": {
        "title": "SVD — Subspace Projection",
        "concept": "Rank-k Matrix Approximation (Eckart-Young)",
        "steps": [
            (
                "The Spectrogram as a Matrix",
                "The STFT gives us a spectrogram matrix:\n\n"
                "   S ∈ ℂ^(F×T)\n\n"
                "where F = frequency bins, T = time frames.\n"
                "Each column is a 'snapshot' of the signal's\n"
                "frequency content at one moment."
            ),
            (
                "SVD Decomposition",
                "Singular Value Decomposition factors S as:\n\n"
                "   S = U · Σ · Vᴴ\n\n"
                "   U ∈ ℂ^(F×F)  — left singular vectors\n"
                "                   (frequency patterns)\n"
                "   Σ ∈ ℝ^(F×T)  — singular values σ₁ ≥ σ₂ ≥ ...\n"
                "                   (importance weights)\n"
                "   Vᴴ ∈ ℂ^(T×T) — right singular vectors\n"
                "                   (temporal patterns)"
            ),
            (
                "The Signal-Noise Separation",
                "Key property of SVD for noisy signals:\n\n"
                "   • SIGNAL concentrates in the TOP few\n"
                "     singular values (σ₁, σ₂, ..., σₖ)\n\n"
                "   • NOISE spreads across ALL singular values\n"
                "     roughly uniformly\n\n"
                "The σᵢ values drop sharply after the signal\n"
                "components — this is the 'spectral gap'."
            ),
            (
                "Rank-k Projection (The Key Step!)",
                "Keep only the top k singular values:\n\n"
                "   S̃ = Σᵢ₌₁ᵏ σᵢ · uᵢ · vᵢᴴ\n\n"
                "This is the BEST rank-k approximation of S\n"
                "(by the Eckart-Young-Mirsky theorem).\n\n"
                "It MINIMIZES ‖S - S̃‖_F among all rank-k\n"
                "matrices — the optimal projection onto\n"
                "a k-dimensional subspace!"
            ),
            (
                "Geometric Interpretation",
                "Think of it as projecting a point in high-D\n"
                "space onto a lower-dimensional plane:\n\n"
                "   Full space: rank r (signal + noise)\n"
                "   Clean subspace: rank k (signal only)\n\n"
                "The projection discards the (r-k) dimensions\n"
                "where noise lives, keeping only the k\n"
                "dimensions where signal dominates."
            ),
            (
                "Reconstruction",
                "Inverse STFT recovers the time-domain signal:\n\n"
                "   y(n) = ISTFT(S̃)\n\n"
                "The result is the closest clean signal to\n"
                "the original — in the least-squares sense."
            ),
        ],
        "key_insight": (
            "💡 KEY INSIGHT\n\n"
            "SVD finds the 'principal components' of the\n"
            "spectrogram. The top-k components capture the\n"
            "structured, repeating patterns (= signal).\n\n"
            "The remaining components are random noise.\n\n"
            "This is the PUREST form of linear algebra\n"
            "denoising — it's literally a projection onto\n"
            "the optimal k-dimensional subspace."
        )
    },

    "wiener": {
        "title": "Wiener Filter — Optimal Linear Estimator",
        "concept": "Minimum Mean Square Error (MMSE) Filter",
        "steps": [
            (
                "The Problem Setup",
                "We observe a noisy signal:\n\n"
                "   x(n) = s(n) + n(n)\n\n"
                "where s(n) is the clean signal and n(n) is noise.\n"
                "In the frequency domain (via FFT):\n\n"
                "   X(f) = S(f) + N(f)"
            ),
            (
                "Power Spectrum Estimation",
                "Estimate noise power from a quiet segment:\n\n"
                "   P_N(f) = 𝔼[|N(f)|²]\n\n"
                "Estimate signal power by spectral subtraction:\n\n"
                "   P_S(f) = max(|X(f)|² - P_N(f), 0)"
            ),
            (
                "The Wiener Gain (The Key Step!)",
                "The optimal linear filter is:\n\n"
                "         P_S(f)\n"
                "   H(f) = ─────────────\n"
                "         P_S(f) + P_N(f)\n\n"
                "This is derived by minimizing the MSE:\n\n"
                "   min_H  𝔼[|S(f) - H(f)·X(f)|²]\n\n"
                "   • Where signal dominates: H(f) → 1  (keep!)\n"
                "   • Where noise dominates:  H(f) → 0  (suppress!)"
            ),
            (
                "Why It's Optimal",
                "The Wiener filter minimizes:\n\n"
                "   𝔼[‖s - ŝ‖²] = 𝔼[Σₙ|s(n) - ŝ(n)|²]\n\n"
                "among ALL linear filters. It's the projection\n"
                "of x onto the space of signals correlated with s.\n\n"
                "In linear algebra terms: it's the ORTHOGONAL\n"
                "PROJECTION that minimizes the residual norm."
            ),
            (
                "Application in Frequency Domain",
                "Apply the gain to each frequency bin:\n\n"
                "   Ŝ(f) = H(f) · X(f)\n\n"
                "This is a weighted projection — instead of\n"
                "binary keep/discard, each frequency is scaled\n"
                "by its estimated signal-to-noise ratio."
            ),
            (
                "Reconstruction",
                "Inverse FFT recovers the time-domain estimate:\n\n"
                "   ŝ(n) = F⁻¹[Ŝ(f)]\n\n"
                "The result is the minimum-variance linear\n"
                "estimate of the clean signal — provably\n"
                "the best any linear method can achieve."
            ),
        ],
        "key_insight": (
            "💡 KEY INSIGHT\n\n"
            "The Wiener filter is the OPTIMAL linear filter —\n"
            "no other linear operation can produce a better\n"
            "estimate of the clean signal.\n\n"
            "It works by computing an SNR-weighted projection:\n"
            "frequencies with high SNR are kept (H≈1), while\n"
            "frequencies dominated by noise are suppressed (H≈0).\n\n"
            "The gain H(f) is itself a projection weight —\n"
            "it smoothly interpolates between keeping and\n"
            "discarding each frequency component."
        )
    }
}


def get_explanation(method: str) -> dict:
    """Get the math explanation for a given method."""
    return MATH_EXPLANATIONS.get(method, MATH_EXPLANATIONS["spectral_gate"])


def format_explanation_text(method: str) -> str:
    """Format the full explanation as a single text block."""
    exp = get_explanation(method)
    lines = []

    lines.append(f"{'═' * 50}")
    lines.append(f"  {exp['title']}")
    lines.append(f"  Concept: {exp['concept']}")
    lines.append(f"{'═' * 50}")
    lines.append("")

    for i, (step_title, step_content) in enumerate(exp['steps'], 1):
        lines.append(f"  ┌─ Step {i}: {step_title}")
        lines.append(f"  │")
        for line in step_content.split('\n'):
            lines.append(f"  │  {line}")
        lines.append(f"  │")
        lines.append(f"  └{'─' * 45}")
        lines.append("")

    lines.append(f"{'─' * 50}")
    lines.append(exp['key_insight'])
    lines.append(f"{'─' * 50}")

    return '\n'.join(lines)
