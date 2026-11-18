import numpy as np

from samleng.audio import stray_tail

SR = 16000


def tone(seconds: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return amp * np.sin(2 * np.pi * 220 * t)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR))


def test_stray_tail_finds_a_burst_after_the_speech_ended():
    clip = np.concatenate([tone(0.7), silence(0.9), tone(0.3)])
    assert stray_tail(clip, SR) is True  # saved to JSON, so a plain bool
    # A burst that fades in softly (short5-200).
    soft = np.concatenate([tone(0.1, 0.002), tone(0.05, 0.02), tone(0.1, 0.3)])
    assert stray_tail(np.concatenate([tone(0.9), silence(1.0), soft]), SR)


def test_stray_tail_keeps_normal_clips_and_real_pauses():
    assert not stray_tail(np.concatenate([tone(1.0), silence(0.3)]), SR)
    # A real second phrase after a pause is longer than a burst.
    assert not stray_tail(np.concatenate([tone(0.7), silence(0.9), tone(1.2)]), SR)
    # A short ending after a short pause is speech, not noise.
    assert not stray_tail(np.concatenate([tone(0.7), silence(0.3), tone(0.3)]), SR)
    assert not stray_tail(silence(1.0), SR)
