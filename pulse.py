from scipy.signal import butter, filtfilt, welch
from numba import njit, prange
import numpy as np


# noinspection SpellCheckingInspection
class BPM:
    """
    This class transforms a BVP signal in a BPM signal using CPU.
    BVP signal must be a float32 numpy.ndarray with shape [num_estimators, num_frames].

    Input 'bvp_sig' is a BVP signal defined as a float32 Numpy.ndarray with shape [num_estimators, num_frames]
    """

    def __init__(self, bvp_sig, fps, minHz=0.65, maxHz=4.):

        self.nFFT = 2048 // 1  # freq. resolution for STFTs
        if len(bvp_sig.shape) == 1:
            self.bvp_sig = bvp_sig.reshape(1, -1)  # 2D array raw-wise
        else:
            self.bvp_sig = bvp_sig
        self.fps = fps  # sample rate
        self.minHz = minHz
        self.maxHz = maxHz

    def BVP_to_BPM(self):
        """
        Return the BPM signal as a float32 Numpy.ndarray with shape [num_estimators, ].
        This method use the Welch's method to estimate the spectral density of the BVP signal,
        then it chooses as BPM the maximum Amplitude frequency.
        """
        if self.bvp_sig.shape[0] == 0:
            return np.float32(0.0)
        Pfreqs, Power = Welch(self.bvp_sig, self.fps, self.minHz, self.maxHz, self.nFFT)
        #  BPM estimate
        Pmax = np.argmax(Power, axis=1)  # power max
        return Pfreqs[Pmax.squeeze()]



@njit(['float32[:,:](uint8[:,:,:], int32, int32)', ], parallel=True, fastmath=True, nogil=True)
def rgb_mean(im, RGB_LOW_TH, RGB_HIGH_TH):
    """
    RGB mean
    This method computes the RGB-Mean Signal excluding 'im' pixels
    that are outside the RGB range [RGB_LOW_TH, RGB_HIGH_TH] (extremes are included).
    the function is jited using numba for performance reasons
    Args:
        im (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels]. contains only the skin portion of the face.
        RGB_LOW_TH (numpy.int32): RGB low threshold value.
        RGB_HIGH_TH (numpy.int32): RGB high threshold value.
    Returns:
        RGB-Mean Signal as float32 ndarray with shape [1,3], where 1 is the single estimator,
        and 3 are r-mean, g-mean and b-mean.
    """
    mean = np.zeros((1, 3), dtype=np.float32)
    mean_r = np.float32(0.0)
    mean_g = np.float32(0.0)
    mean_b = np.float32(0.0)
    num_elems = np.float32(0.0)
    for x in prange(im.shape[0]):
        for y in prange(im.shape[1]):
            if not ((im[x, y, 0] <= RGB_LOW_TH and im[x, y, 1] <= RGB_LOW_TH and im[x, y, 2] <= RGB_LOW_TH)
                    or (im[x, y, 0] >= RGB_HIGH_TH and im[x, y, 1] >= RGB_HIGH_TH and im[x, y, 2] >= RGB_HIGH_TH)):
                mean_r += im[x, y, 0]
                mean_g += im[x, y, 1]
                mean_b += im[x, y, 2]
                num_elems += 1.0
    if num_elems > 1.0:
        mean[0, 0] = mean_r / num_elems
        mean[0, 1] = mean_g / num_elems
        mean[0, 2] = mean_b / num_elems
    else:
        mean[0, 0] = mean_r
        mean[0, 1] = mean_g
        mean[0, 2] = mean_b
    return mean


def BPfilter(sig, fps, minHz=0.7, maxHz=3.0, order=6):
    """
    pre-filter
    Band Pass filter for RGB signal and BVP signal.
    """
    x = np.array(np.swapaxes(sig, 1, 2))
    b, a = butter(order, Wn=[minHz, maxHz], fs=fps, btype='bandpass')
    y = filtfilt(b, a, x, axis=1)
    y = np.swapaxes(y, 1, 2)
    return y


def cpu_POS(signal, fps):
    """
    POS method on CPU using Numpy.
    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions
    on Biomedical Engineering, 64(7), 1479-1491.
    https://pure.tue.nl/ws/portalfiles/portal/31563684/TBME_00467_2016_R1_preprint.pdf
    """
    # Run the pos algorithm on the RGB color signal c with sliding window length wlen
    # Recommended value for wlen is 32 for a 20 fps camera (1.6 s)
    eps = 10 ** -9
    X = signal
    e, c, f = X.shape  # e = #estimators, c = 3 rgb ch., f = #frames
    w = int(1.6 * fps)  # window length

    # stack e times fixed mat P
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)

    # Initialize (1)
    H = np.zeros((e, f))
    for n in np.arange(w, f):
        # Start index of sliding window (4)
        m = n - w + 1
        # Temporal normalization (5)
        Cn = X[:, :, m:(n + 1)]
        M = 1.0 / (np.mean(Cn, axis=2) + eps)
        M = np.expand_dims(M, axis=2)  # shape [e, c, w]
        Cn = np.multiply(M, Cn)

        # Projection (6)
        S = np.dot(Q, Cn)
        S = S[0, :, :, :]
        S = np.swapaxes(S, 0, 1)  # remove 3-th dim

        # Tuning (7)
        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
        # Overlap-adding (8)
        H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)

    return H


# noinspection SpellCheckingInspection
def Welch(bvps, fps, minHz=0.65, maxHz=4.0, nfft=2048):
    """
    This function computes Welch'method for spectral density estimation.
    Args:
        bvps(flaot32 numpy.ndarray): BVP signal as float32 Numpy.ndarray with shape [num_estimators, num_frames].
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        nfft (int): number of DFT points, specified as a positive integer.
    Returns:
        Sample frequencies as float32 numpy.ndarray, and Power spectral density or power spectrum as float32 numpy.ndarray.
    """
    _, n = bvps.shape
    if n < 256:
        seglength = n
        overlap = int(0.8 * n)  # fixed overlapping
    else:
        seglength = 256
        overlap = 200
    # -- periodogram by Welch
    F, P = welch(bvps, nperseg=seglength, noverlap=overlap, fs=fps, nfft=nfft)
    F = F.astype(np.float32)
    P = P.astype(np.float32)
    # -- freq sub-band (0.65 Hz - 4.0 Hz)
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
    Pfreqs = 60 * F[band]
    Power = P[:, band]
    return Pfreqs, Power
