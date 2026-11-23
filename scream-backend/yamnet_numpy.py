"""
yamnet_numpy.py
===============
YAMNet written with plain NumPy, so it runs WITHOUT TensorFlow.

It gives the same 1024-number embeddings as the TensorFlow YAMNet that was
used to train scream_classifier.pkl, so the .pkl keeps working unchanged.

Needs only:  numpy   +   the file yamnet_weights.npz (made once in Colab)

Usage:
    from yamnet_numpy import YamnetNumpy
    yamnet = YamnetNumpy("yamnet_weights.npz")
    embedding = yamnet.embedding(audio_16khz_float32)      # shape (1024,)
"""

import numpy as np

# ---- YAMNet settings (same as params.py of the official model) ---------------
SAMPLE_RATE = 16000
STFT_WINDOW = 400            # 0.025 s
STFT_HOP = 160               # 0.010 s
FFT_LENGTH = 512
MEL_BANDS = 64
MEL_MIN_HZ = 125.0
MEL_MAX_HZ = 7500.0
LOG_OFFSET = 0.001
PATCH_FRAMES = 96            # 0.96 s
PATCH_HOP_FRAMES = 48        # 0.48 s
BN_EPSILON = 1e-4

# (kind, stride, filters) for layer1 ... layer14
LAYER_DEFS = [
    ("conv", 2, 32),
    ("sep", 1, 64),
    ("sep", 2, 128),
    ("sep", 1, 128),
    ("sep", 2, 256),
    ("sep", 1, 256),
    ("sep", 2, 512),
    ("sep", 1, 512),
    ("sep", 1, 512),
    ("sep", 1, 512),
    ("sep", 1, 512),
    ("sep", 1, 512),
    ("sep", 2, 1024),
    ("sep", 1, 1024),
]


# ---- audio -> log-mel patches --------------------------------------------------
def _pad_waveform(wave):
    """Same padding rule as YAMNet's pad_waveform()."""
    min_samples = int(np.float32((0.96 + 0.025 - 0.010) * SAMPLE_RATE))
    hop_samples = int(np.float32(0.48 * SAMPLE_RATE))
    n = len(wave)
    pad = max(0, min_samples - n)
    n = max(n, min_samples)
    after_first = n - min_samples
    hops = int(np.ceil(np.float32(after_first) / np.float32(hop_samples)))
    pad += hop_samples * hops - after_first
    return np.pad(wave, (0, pad))


def _mel_matrix():
    """Same as tf.signal.linear_to_mel_weight_matrix (HTK mel scale)."""
    n_bins = FFT_LENGTH // 2 + 1

    def hz_to_mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    freqs = np.linspace(0.0, SAMPLE_RATE / 2.0, n_bins)[1:]      # DC bin is skipped
    bins_mel = hz_to_mel(freqs)[:, None]
    edges = np.linspace(hz_to_mel(MEL_MIN_HZ), hz_to_mel(MEL_MAX_HZ), MEL_BANDS + 2)
    lower, center, upper = edges[:-2][None, :], edges[1:-1][None, :], edges[2:][None, :]
    lower_slope = (bins_mel - lower) / (center - lower)
    upper_slope = (upper - bins_mel) / (upper - center)
    weights = np.maximum(0.0, np.minimum(lower_slope, upper_slope))
    return np.pad(weights, ((1, 0), (0, 0)))                      # shape (257, 64)


def _frames(x, length, step):
    """Cut x (axis 0) into overlapping frames; only complete frames are kept."""
    n = 1 + (x.shape[0] - length) // step
    idx = np.arange(length)[None, :] + step * np.arange(n)[:, None]
    return x[idx]


def waveform_to_patches(wave, mel_matrix, window):
    wave = _pad_waveform(wave.astype(np.float32))
    framed = _frames(wave, STFT_WINDOW, STFT_HOP) * window        # (frames, 400)
    spectrum = np.abs(np.fft.rfft(framed, n=FFT_LENGTH, axis=1))  # (frames, 257)
    mel = spectrum @ mel_matrix
    log_mel = np.log(mel + LOG_OFFSET).astype(np.float32)         # (frames, 64)
    return _frames(log_mel, PATCH_FRAMES, PATCH_HOP_FRAMES)       # (patches, 96, 64)


# ---- neural network --------------------------------------------------------------
def _same_padding(size, kernel, stride):
    out = -(-size // stride)
    total = max((out - 1) * stride + kernel - size, 0)
    return out, total // 2, total - total // 2


def _conv3x3(x, kernel, stride, depthwise):
    """3x3 convolution with TensorFlow 'same' padding.
    x: (B, H, W, C)   kernel: (3, 3, C, F) normal  |  (3, 3, C) depthwise"""
    _, h, w, _ = x.shape
    kh, kw = kernel.shape[0], kernel.shape[1]
    oh, ph0, ph1 = _same_padding(h, kh, stride)
    ow, pw0, pw1 = _same_padding(w, kw, stride)
    xp = np.pad(x, ((0, 0), (ph0, ph1), (pw0, pw1), (0, 0)))
    out = 0
    for i in range(kh):
        for j in range(kw):
            part = xp[:, i:i + stride * (oh - 1) + 1:stride,
                      j:j + stride * (ow - 1) + 1:stride, :]
            out = out + (part * kernel[i, j] if depthwise else part @ kernel[i, j])
    return out


class YamnetNumpy:
    def __init__(self, weights_path):
        data = np.load(weights_path)
        self.layers = []
        for i, (kind, stride, _) in enumerate(LAYER_DEFS, start=1):
            if kind == "conv":
                self.layers.append(("conv", stride, self._fold(data, f"layer{i}/conv", False)))
            else:
                dw = self._fold(data, f"layer{i}/depthwise_conv", True)
                pw = self._fold(data, f"layer{i}/pointwise_conv", False)
                self.layers.append(("sep", stride, (dw, pw)))
        self.mel_matrix = _mel_matrix().astype(np.float64)
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(STFT_WINDOW) / STFT_WINDOW)
                       ).astype(np.float32)

    @staticmethod
    def _fold(data, prefix, depthwise):
        """Merge the batch-norm into the convolution: returns (kernel, bias)."""
        kernel = data[prefix + "/kernel"].astype(np.float64)
        if depthwise:                                   # (3, 3, C, 1) -> (3, 3, C)
            kernel = kernel[..., 0]
        mean = data[prefix + "/bn_mean"].astype(np.float64)
        var = data[prefix + "/bn_var"].astype(np.float64)
        beta = data[prefix + "/bn_beta"].astype(np.float64)
        scale = 1.0 / np.sqrt(var + BN_EPSILON)
        kernel = kernel * scale                         # scale the output channels
        bias = beta - mean * scale
        return kernel.astype(np.float32), bias.astype(np.float32)

    def patch_embeddings(self, wave):
        """One 1024-number embedding per 0.96 s patch (hop 0.48 s)."""
        patches = waveform_to_patches(np.asarray(wave), self.mel_matrix, self.window)
        x = patches[..., None].astype(np.float32)       # (patches, 96, 64, 1)
        for kind, stride, params in self.layers:
            if kind == "conv":
                kernel, bias = params
                x = np.maximum(_conv3x3(x, kernel, stride, False) + bias, 0.0)
            else:
                (dk, db), (pk, pb) = params
                x = np.maximum(_conv3x3(x, dk, stride, True) + db, 0.0)
                x = np.maximum(x @ pk[0, 0] + pb, 0.0)
        return x.mean(axis=(1, 2))                      # (patches, 1024)

    def embedding(self, wave):
        """Mean of the patch embeddings = what the classifier was trained on."""
        return self.patch_embeddings(wave).mean(axis=0).astype(np.float32)
