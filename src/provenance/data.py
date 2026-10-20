"""Isolated HSC galaxy models used as ground truth for simulated observations.

The source catalogue is the 500-row HSC tract 9697 / patch 72 subset published in
SampsonML/HSC_galaxy_tools (fetch it with ``scripts/fetch_data.sh``). Its ``scar_*``
columns are SCARLET's deblended, deconvolved single-source models, i.e. the same kind
of image the galaxygrad prior was trained on. Caveat: the prior's training set was
drawn from three HSC tracts including 9697, so these sources are probably not held
out from the prior's training data.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "hsc_9697_72_subset.pkl"
IMAGE_SIZE = 32


@dataclass(frozen=True)
class Source:
    """A single galaxy, peak-normalised to 1 (the prior's training convention)."""

    source_id: str
    truth: np.ndarray  # (IMAGE_SIZE, IMAGE_SIZE), max == 1
    psf: np.ndarray  # odd-sized kernel, sums to 1
    sigma: float  # per-pixel noise std of the real HSC exposure, in truth units
    scale: float  # raw flux of the truth's peak pixel

    @property
    def snr(self):
        """Matched-filter S/N of the noiseless truth at the native noise level."""
        return float(np.sqrt(np.sum(self.truth**2)) / self.sigma)


def _pad_center(a, size):
    n = a.shape[0]
    lo = (size - n) // 2
    return np.pad(a, ((lo, size - n - lo), (lo, size - n - lo)))


def _crop_center(a, size):
    lo = (a.shape[0] - size) // 2
    return a[lo : lo + size, lo : lo + size]


def load_sources(path=DEFAULT_PATH, band="i", psf_size=15, min_snr=20.0):
    """Return isolated sources whose model box fits in a 32x32 frame, sorted by id."""
    df = pd.read_pickle(path)
    sources = []
    for row in df.itertuples():
        model = np.asarray(getattr(row, f"scar_{band}"), dtype=np.float64)
        if model.shape[0] > IMAGE_SIZE or model.shape[0] != model.shape[1]:
            continue
        peak = model.max()
        if not np.isfinite(peak) or peak <= 0:
            continue
        truth = _pad_center(np.clip(model, 0, None) / peak, IMAGE_SIZE)
        psf = _crop_center(np.asarray(getattr(row, f"PSF_{band}"), dtype=np.float64), psf_size)
        psf = np.clip(psf, 0, None)
        psf /= psf.sum()
        var = np.asarray(getattr(row, f"var_{band}"))
        sigma = float(np.sqrt(np.median(var[var > 0]))) / peak
        src = Source(f"{row.parent_ID}-{row.object_num}", truth, psf, sigma, float(peak))
        if src.snr >= min_snr:
            sources.append(src)
    return sorted(sources, key=lambda s: s.source_id)


def split(sources, n_dev=8):
    """Fixed dev/eval split: the first ``n_dev`` sources (by id) are used for tuning."""
    return sources[:n_dev], sources[n_dev:]
