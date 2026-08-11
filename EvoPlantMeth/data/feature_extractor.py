from __future__ import division, print_function
import numpy as np

class KnnCpgFeatureExtractor(object):
    def __init__(self, k=1):
        self.k = k

    def extract(self, x, y, ys):
        n, m, k, kk = len(x), len(y), self.k, 2 * self.k
        yc = self.__larger_equal(x, y)
        knn_cpg = np.full((n, kk), np.nan, dtype=np.float16)
        knn_dist = np.full((n, kk), np.nan, dtype=np.float32)

        for i in range(n):
            yl, yr = yc[i] - k, yc[i] - 1
            if yr >= 0:
                xl, xr = 0, k - 1
                if yl < 0:
                    xl += np.abs(yl)
                    yl = 0
                knn_cpg[i, xl:xr+1] = ys[yl:yr+1]
                knn_dist[i, xl:xr+1] = np.abs(y[yl:yr+1] - x[i])

            yl = yc[i]
            if yl >= m: continue
            if x[i] == y[yl]:
                yl += 1
                if yl >= m: continue
            yr = yl + k - 1
            xl, xr = k, 2 * k - 1
            if yr >= m:
                xr -= yr - m + 1
                yr = m - 1
            knn_cpg[i, xl:xr+1] = ys[yl:yr+1]
            knn_dist[i, xl:xr+1] = np.abs(y[yl:yr+1] - x[i])

        return (knn_cpg, knn_dist)

    def __larger_equal(self, x, y):
        n, m = len(x), len(y)
        rv = np.empty(n, dtype=np.int64)
        i = j = 0
        while i < n and j < m:
            while j < m and x[i] > y[j]: j += 1
            rv[i] = j
            i += 1
        if i < n: rv[i:] = m
        return rv


class IntervalFeatureExtractor(object):
    @staticmethod
    def join_intervals(s, e):
        n = len(s)
        if n == 0: return ([], [])
        rs, re = [], []
        l, r = s[0], e[0]
        for i in range(1, n):
            if s[i] > r:
                rs.append(l); re.append(r)
                l, r = s[i], e[i]
            else:
                r = max(r, e[i])
        rs.append(l); re.append(r)
        return (rs, re)

    @staticmethod
    def index_intervals(x, ys, ye):
        n, m = len(ys), len(x)
        rv = np.full(m, -1, dtype=np.int64)
        i = j = 0
        while i < n and j < m:
            while j < m and x[j] <= ye[i]:
                if x[j] >= ys[i]: rv[j] = i
                j += 1
            i += 1
        return rv

    def extract(self, x, ys, ye):
        return self.index_intervals(x, ys, ye) >= 0


class KmersFeatureExtractor(object):
    def __init__(self, kmer_len, nb_char=4):
        self.kmer_len = kmer_len
        self.nb_char = nb_char
        self.nb_kmer = self.nb_char**self.kmer_len

    def __call__(self, seqs):
        nb_seq, seq_len = seqs.shape
        kmer_freq = np.zeros((nb_seq, self.nb_kmer), dtype=np.int32)
        vec = np.array([self.nb_char**i for i in range(self.kmer_len)], dtype=np.int32)
        for i in range(nb_seq):
            for j in range(seq_len - self.kmer_len + 1):
                kmer = seqs[i, j:(j + self.kmer_len)]
                kmer_freq[i, kmer.dot(vec)] += 1
        return kmer_freq