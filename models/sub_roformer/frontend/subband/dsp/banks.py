import math
import numpy as np


def linear_banks(
    sr: int, 
    n_bands: int, 
) -> list[tuple[float, float]]:
    r"""Linear banks.

    Returns:
        (n_banks, 2)
    """
    freqs = np.linspace(0, sr / 2, n_bands + 1)
    banks = [[freqs[i].item(), freqs[i + 1].item()] for i in range(len(freqs) - 1)]  # (k1+k2, 2)
    return banks


def mel_linear_banks(
    sr: int, 
    n_bands: int, 
    max_bandwidth: float
) -> list[tuple[float, float]]:
    r"""Mel bank in low frequency and linear band in high frequency.

    Returns:
        (n_banks, 2)
    """
    freqs = np.linspace(0, hz_to_mel(sr / 2), n_bands + 1)
    freqs = mel_to_hz(freqs)

    if max(np.diff(freqs)) >= max_bandwidth:
        idx = np.argmax(np.diff(freqs) >= max_bandwidth)  # (k1,)
        mel_part = freqs[: idx + 1]  # (k1,)
        linear_part = np.arange(mel_part[-1] + max_bandwidth, sr//2 + max_bandwidth, max_bandwidth)  # (k2,)
        freqs = np.concatenate([mel_part, linear_part])  # (k1+k2,)
        freqs[-1] = sr // 2

    banks = [[freqs[i].item(), freqs[i + 1].item()] for i in range(len(freqs) - 1)]  # (k1+k2, 2)
    return banks


def mel_linear_banks_triangle(
    sr: int, 
    n_bands: int, 
    max_bandwidth: float
) -> list[tuple[float, float]]:
    r"""Mel bank in low frequency and linear band in high frequency.

    Returns:
        (n_banks, 2)
    """
    freqs = np.linspace(0, hz_to_mel(sr / 2), n_bands + 1)
    freqs = mel_to_hz(freqs)

    if max(np.diff(freqs)) >= max_bandwidth:
        idx = np.argmax(np.diff(freqs) >= max_bandwidth)  # (k1,)
        mel_part = freqs[: idx + 1]  # (k1,)
        linear_part = np.arange(mel_part[-1] + max_bandwidth, sr//2 + max_bandwidth, max_bandwidth)  # (k2,)
        freqs = np.concatenate([mel_part, linear_part])  # (k1+k2,)
        freqs[-1] = sr // 2

    banks = [[freqs[0].item(), freqs[1].item()]]
    banks += [[freqs[i].item(), freqs[i + 1].item(), freqs[i + 2].item()] for i in range(len(freqs) - 2)]
    banks += [[freqs[-2].item(), freqs[-1].item()]]
    return banks


def erb_linear_banks(
    sr: int, 
    n_bands: int, 
    max_bandwidth: float,
) -> list[tuple[float, float]]:
    r"""ERB bank in low frequency and linear band in high frequency.

    Returns:
        (n_banks, 2)
    """
    freqs = np.linspace(0, hz_to_erb(sr / 2), n_bands + 1)
    freqs = erb_to_hz(freqs)

    if max(np.diff(freqs)) >= max_bandwidth:
        idx = np.argmax(np.diff(freqs) >= max_bandwidth)  # (k1,)
        mel_part = freqs[: idx + 1]  # (k1,)
        linear_part = np.arange(mel_part[-1] + max_bandwidth, sr//2 + max_bandwidth, max_bandwidth)  # (k2,)
        freqs = np.concatenate([mel_part, linear_part])  # (k1+k2,)
        freqs[-1] = sr // 2

    banks = [[freqs[i].item(), freqs[i + 1].item()] for i in range(len(freqs) - 1)]  # (k1+k2, 2)
    return banks


def erb_linear_banks_overlap(
    sr: int, 
    n_bands: int, 
    max_bandwidth: float,
) -> list[tuple[float, float]]:
    r"""ERB bank in low frequency and linear band in high frequency.

    Returns:
        (n_banks, 2)
    """
    freqs = np.linspace(0, hz_to_erb(sr / 2), n_bands + 1)
    freqs = erb_to_hz(freqs)

    if max(np.diff(freqs)) >= max_bandwidth:
        idx = np.argmax(np.diff(freqs) >= max_bandwidth)  # (k1,)
        mel_part = freqs[: idx + 1]  # (k1,)
        linear_part = np.arange(mel_part[-1] + max_bandwidth, sr//2 + max_bandwidth, max_bandwidth)  # (k2,)
        freqs = np.concatenate([mel_part, linear_part])  # (k1+k2,)
        freqs[-1] = sr // 2

    banks = [[freqs[0].item(), freqs[1].item()]]
    banks += [[freqs[i].item(), freqs[i + 2].item()] for i in range(len(freqs) - 2)]  # (k1+k2, 2)
    banks += [[freqs[-2].item(), freqs[-1].item()]]
    return banks


def erb_linear_banks_triangle(
    sr: int, 
    n_bands: int, 
    max_half_bandwidth: float,
) -> list[tuple[float, float]]:
    r"""ERB bank in low frequency and linear band in high frequency.

    Returns:
        (n_banks, 2)
    """
    freqs = np.linspace(0, hz_to_erb(sr / 2), n_bands + 1)
    freqs = erb_to_hz(freqs)

    if max(np.diff(freqs)) >= max_half_bandwidth:
        idx = np.argmax(np.diff(freqs) >= max_half_bandwidth)  # (k1,)
        mel_part = freqs[: idx + 1]  # (k1,)
        linear_part = np.arange(mel_part[-1] + max_half_bandwidth, sr//2 + max_half_bandwidth, max_half_bandwidth)  # (k2,)
        freqs = np.concatenate([mel_part, linear_part])  # (k1+k2,)
        freqs[-1] = sr // 2

    banks = [[freqs[0].item(), freqs[1].item()]]
    banks += [[freqs[i].item(), freqs[i + 1].item(), freqs[i + 2].item()] for i in range(len(freqs) - 2)]
    banks += [[freqs[-2].item(), freqs[-1].item()]]
    
    return banks


def erb_linear_ex_banks_triangle(
    sr: int, 
    n_bands: int, 
    max_half_bandwidth: float,
    a: float,
    b: float,
) -> list[tuple[float, float]]:
    r"""ERB bank in low frequency and linear band in high frequency.

    Returns:
        (n_banks, 2)
    """
    freqs = np.linspace(0, hz_to_erb_ex(sr / 2, a, b), n_bands + 1)
    freqs = erb_to_hz_ex(freqs, a, b)

    if max(np.diff(freqs)) >= max_half_bandwidth:
        idx = np.argmax(np.diff(freqs) >= max_half_bandwidth)  # (k1,)
        mel_part = freqs[: idx + 1]  # (k1,)
        linear_part = np.arange(mel_part[-1] + max_half_bandwidth, sr//2 + max_half_bandwidth, max_half_bandwidth)  # (k2,)
        freqs = np.concatenate([mel_part, linear_part])  # (k1+k2,)
        freqs[-1] = sr // 2

    banks = [[freqs[0].item(), freqs[1].item()]]
    banks += [[freqs[i].item(), freqs[i + 1].item(), freqs[i + 2].item()] for i in range(len(freqs) - 2)]
    banks += [[freqs[-2].item(), freqs[-1].item()]]
    
    return banks


def exp_linear_banks(
    sr: int, 
    n_bands: int, 
) -> list[tuple[float, float]]:
    r"""ERB bank in low frequency and linear band in high frequency.

    Returns:
        (n_banks, 2)
    """

    # Low freq
    freqs = [0, 25, 30]

    # Middle freq
    r = 2
    for i in range(68):
        freqs.append(freqs[-1] + r)
        r = r ** 1.032

    # High freq
    n_linear_bands = (n_bands - len(freqs))
    bw = ((sr / 2) - freqs[-1]) / n_linear_bands

    for _ in range(n_linear_bands):
        freqs.append(freqs[-1] + bw)
    
    freqs[-1] = sr / 2

    # Banks
    banks = [[freqs[0], freqs[1]]]
    banks += [[freqs[i], freqs[i + 1], freqs[i + 2]] for i in range(len(freqs) - 2)]
    banks += [[freqs[-2], freqs[-1]]]
    
    return banks


def exp_linear_banks2(
    sr: int, 
    n_bands: int, 
    max_bandwidth: float,
    q: float
) -> list[tuple[float, float]]:
    r"""ERB bank in low frequency and linear band in high frequency.

    Returns:
        (n_banks, 2)
    """

    n_bands -= 2
    max_bw = max_bandwidth

    a1 = (max_bw * q) / (q - 1) - sr / 2 + n_bands * max_bw
    a1 = a1 / max_bw - 1
    exp = q ** a1
    f0 = max_bw / (q - 1) / exp

    n = 1 + math.log((max_bw / f0) / (q - 1)) / math.log(q)
    n = math.ceil(n)

    # Calibriate f0
    f0 = max_bw / (q ** n - q ** (n-1))

    freqs = [0]
    for i in range(n+1):
        freqs.append(f0 * q**i)

    n_linear_bands = n_bands - n
    bw = (sr / 2 - f0 * q ** n) / n_linear_bands
    for i in range(n_linear_bands):
        freqs.append(freqs[-1] + bw)

    freqs[-1] = sr / 2

    # Banks
    banks = [[freqs[0], freqs[1]]]
    banks += [[freqs[i], freqs[i + 1], freqs[i + 2]] for i in range(len(freqs) - 2)]
    banks += [[freqs[-2], freqs[-1]]]
    
    return banks


def linear_banks_triangle(
    sr: int, 
    n_bands: int, 
) -> list[tuple[float, float]]:
    r"""ERB bank in low frequency and linear band in high frequency.

    Returns:
        (n_banks, 2)
    """
    
    freqs = np.linspace(0, sr / 2, n_bands)
    banks = [[freqs[0].item(), freqs[1].item()]]
    banks += [[freqs[i].item(), freqs[i + 1].item(), freqs[i + 2].item()] for i in range(len(freqs) - 2)]
    banks += [[freqs[-2].item(), freqs[-1].item()]]
    
    return banks


def hz_to_erb(f):
    return 21.4 * np.log10(1 + 0.00437 * f)


def erb_to_hz(erb):
    return 1 / 0.00437 * (10 ** (erb / 21.4) - 1)


def hz_to_erb_ex(f, a, b):
    return a * np.log10(1 + b * f)


def erb_to_hz_ex(erb, a, b):
    return 1 / b * (10 ** (erb / a) - 1)



def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)
