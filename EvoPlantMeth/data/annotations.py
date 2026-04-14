from __future__ import division, print_function
import pandas as pd
import numpy as np

def read_bed(filename, sort=False, usecols=[0, 1, 2], *args, **kwargs):
    d = pd.read_table(filename, header=None, usecols=usecols, *args, **kwargs)
    d.columns = list(range(d.shape[1]))
    d.rename(columns={0: 'chromo', 1: 'start', 2: 'end'}, inplace=True)
    if sort: d.sort_values(['chromo', 'start', 'end'], inplace=True)
    return d

def in_which(x, ys, ye):
    n, m = len(ys), len(x)
    rv = np.full(m, -1, dtype=int)
    i = j = 0
    while i < n and j < m:
        while j < m and x[j] <= ye[i]:
            if x[j] >= ys[i]: rv[j] = i
            j += 1
        i += 1
    return rv

def is_in(pos, start, end):
    return in_which(pos, start, end) >= 0

def distance(pos, start, end):
    m, n = len(start), len(pos)
    i = j = 0
    end_prev = -10**7
    dist = np.zeros(n)
    while i < m and j < n:
        while j < n and pos[j] <= end[i]:
            if pos[j] < start[i]:
                dist[j] = min(pos[j] - end_prev, start[i] - pos[j])
            j += 1
        end_prev = end[i]
        i += 1
    dist[j:] = pos[j:] - end_prev
    return dist

def join_overlapping(s, e):
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

def join_overlapping_frame(d):
    d = d.sort_values(['chromo', 'start', 'end'])
    e = []
    for chromo in d.chromo.unique():
        dc = d.loc[d.chromo == chromo]
        start, end = join_overlapping(dc.start.values, dc.end.values)
        e.append(pd.DataFrame(dict(chromo=chromo, start=start, end=end)))
    return pd.concat(e).loc[:, ['chromo', 'start', 'end']]

def group_overlapping(s, e):
    n = len(s)
    group = np.zeros(n, dtype='int32')
    if n == 0: return group
    idx, r = 0, e[0]
    for i in range(1, n):
        if s[i] > r:
            idx += 1
            r = e[i]
        else:
            r = max(r, e[i])
        group[i] = idx
    return group

def extend_len(start, end, min_len, min_pos=1):
    delta = np.maximum(0, min_len - (end - start + 1))
    ext = np.floor(0.5 * delta).astype(int)
    start_ext = np.maximum(min_pos, start - ext)
    end_ext = end + np.maximum(0, (min_len - (end - start_ext + 1)))
    return (start_ext, end_ext)

def extend_len_frame(d, min_len):
    start, end = extend_len(d.start.values, d.end.values, min_len)
    e = d.copy()
    e['start'], e['end'] = start, end
    return e