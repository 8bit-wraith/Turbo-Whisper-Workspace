"""
Marine-Sense Integration for Vocalis

Brings O(1) salience detection, VAD, and sound localization capabilities
to the Vocalis audio processing pipeline.
"""

from .marine_vad import MarineVAD, VadState, VadSegment
from .marine_localization import (
    MarineLocalization,
    MicrophonePosition,
    SoundSource,
    MarineAlgorithm
)

__all__ = [
    'MarineVAD',
    'VadState',
    'VadSegment',
    'MarineLocalization',
    'MicrophonePosition',
    'SoundSource',
    'MarineAlgorithm',
]
