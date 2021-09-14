import collections
import numpy as np
from scipy.signal import butter


class ButterFilter(object):
    """ Implements butterworth low-pass filter.
    Based on https://github.com/google-research/motion_imitation/blob/master/motion_imitation/robots/action_filter.py
    """

    def __init__(self, sampling_rate, action_size, highcut = [4.0]):
        self.action_size = action_size
        self.sampling_rate = sampling_rate

        self.highcut = highcut
        self.lowcut = [0.0]
        self.order = 2

        a_coeffs = []
        b_coeffs = []
        for i, h in enumerate(self.highcut):
            b, a = self.butter_filter_coefficients(h, sampling_rate, self.order)
            b_coeffs.append(b)
            a_coeffs.append(a)

        if isinstance(a, list):
            self.a = a
            self.b = b
        else:
            self.a = [a]
            self.b = [b]

        # Normalize by a[0]
        for i in range(len(self.a)):
            self.b[i] /= self.a[i][0]
            self.a[i] /= self.a[i][0]

        # Convert single filter to same format as filter per joint
        if len(self.a) == 1:
            self.a *= action_size
            self.b *= action_size
        self.a = np.stack(self.a)
        self.b = np.stack(self.b)

        assert len(self.b[0]) == len(self.a[0]) == self.order + 1
        self.hist_len = self.order

        self.yhist = collections.deque(maxlen=self.hist_len)
        self.xhist = collections.deque(maxlen=self.hist_len)
        self.reset()

    def reset(self):
        self.yhist.clear()
        self.xhist.clear()
        for _ in range(self.hist_len):
            self.yhist.appendleft(np.zeros((self.action_size, 1)))
            self.xhist.appendleft(np.zeros((self.action_size, 1)))

    def filter(self, x):
        xs = np.concatenate(list(self.xhist), axis=-1)
        ys = np.concatenate(list(self.yhist), axis=-1)
        y = np.multiply(x, self.b[:, 0]) + np.sum(
            np.multiply(xs, self.b[:, 1:]), axis=-1) - np.sum(
            np.multiply(ys, self.a[:, 1:]), axis=-1)
        self.xhist.appendleft(x.reshape((self.action_size, 1)).copy())
        self.yhist.appendleft(y.reshape((self.action_size, 1)).copy())
        return y

    def init_history(self, x):
        x = np.expand_dims(x, axis=-1)
        for i in range(self.hist_len):
            self.xhist[i] = x
            self.yhist[i] = x

    def butter_filter_coefficients(self, highcut, fs, order=5):
        nyq = 0.5 * fs
        high = highcut / nyq
        b, a = butter(order, [high], btype='low')
        return b, a
