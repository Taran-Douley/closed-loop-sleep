"""Load Sleep-EDF (Sleep Cassette) recordings with a per-sample sleep stage array.

Sleep-EDF's Fpz-Cz channel is a frontal bipolar derivation at 100 Hz, which is a
reasonable stand-in for the frontal sensors on consumer headbands like the Muse S.
"""

import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# R&K stages in Sleep-EDF -> AASM-style integer codes
STAGE_CODES = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # S3 + S4 = N3
    "Sleep stage R": 4,
}
STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM", -1: "?"}


@dataclass
class Recording:
    subject: int
    eeg: np.ndarray  # microvolts
    sfreq: float
    stages: np.ndarray  # int per sample, -1 = unscored/movement
    night: int = 1

    @property
    def hours(self) -> float:
        return len(self.eeg) / self.sfreq / 3600


MIRROR = "https://physionet-open.s3.amazonaws.com"
MIRROR_PREFIX = "sleep-edfx/1.0.0/sleep-cassette/"


def prefetch(subject: int, night: int = 1) -> None:
    """Download from PhysioNet's AWS open-data mirror, which is far faster than
    physionet.org. MNE then finds (and hash-checks) the files instead of re-downloading."""
    dest = DATA_DIR / "physionet-sleep-data"
    dest.mkdir(parents=True, exist_ok=True)
    stem = f"SC4{subject:02d}{night}"  # then E, F or G (technician code)
    listing = urllib.request.urlopen(f"{MIRROR}/?list-type=2&prefix={MIRROR_PREFIX}{stem}").read().decode()
    for key in re.findall(r"<Key>([^<]+)</Key>", listing):
        target = dest / key.rsplit("/", 1)[1]
        if not target.exists():
            tmp = target.with_suffix(".part")
            urllib.request.urlretrieve(f"{MIRROR}/{key}", tmp)
            tmp.rename(target)


# Sleep Cassette: subjects 0-82, two nights each, minus these missing recordings
MISSING = {(13, 2), (36, 1), (39, 1), (39, 2), (52, 1), (68, 1), (68, 2), (69, 1), (69, 2), (78, 1), (78, 2), (79, 1), (79, 2)}
ALL_RECORDINGS = [(s, n) for s in range(83) for n in (1, 2) if (s, n) not in MISSING]


def load_subject(subject: int, channel: str = "EEG Fpz-Cz", night: int = 1) -> Recording:
    try:
        prefetch(subject, night)
    except OSError as e:  # fall back to MNE's own download from physionet.org
        print(f"mirror prefetch failed for subject {subject} night {night}: {e}")
    [(psg_file, hyp_file)] = mne.datasets.sleep_physionet.age.fetch_data(
        subjects=[subject], recording=[night], path=DATA_DIR, on_missing="raise", verbose="error"
    )
    raw = mne.io.read_raw_edf(psg_file, include=[channel], preload=True, verbose="error")
    annot = mne.read_annotations(hyp_file)

    sfreq = raw.info["sfreq"]
    eeg = raw.get_data()[0] * 1e6
    stages = np.full(len(eeg), -1, dtype=np.int8)
    for onset, duration, desc in zip(annot.onset, annot.duration, annot.description):
        if desc in STAGE_CODES:
            a = int(round(onset * sfreq))
            b = min(int(round((onset + duration) * sfreq)), len(eeg))
            stages[a:b] = STAGE_CODES[desc]

    # Sleep-EDF cassette files run ~22 h; keep the night (first to last sleep epoch) +/- 30 min
    asleep = np.flatnonzero(stages > 0)
    pad = int(30 * 60 * sfreq)
    a, b = max(asleep[0] - pad, 0), min(asleep[-1] + pad, len(eeg))
    return Recording(subject, eeg[a:b], sfreq, stages[a:b], night)
