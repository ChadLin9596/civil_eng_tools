import numpy as np

def distances(s, e):
    if np.shape(s) != np.shape(e):
        raise ValueError("s and e must be (n, 3) array")
    return np.sqrt(np.sum((e - s) ** 2, axis=-1))

def normalize(v):
    d = distances(np.zeros(v.shape), v)

    if len(v.shape) <= 1:
        return v / d if d > 0 else v

    d[d==0] = 1.
    d = d[..., None]
    return v / d

def expand_line_points(s, e, step_size=0.005, start=False, end=False):
    d = distances(s, e)
    v = normalize(e - s) * step_size
    n = (d / step_size).astype(np.int64) + 1

    ss = np.repeat(s, n, axis=0)
    vs = np.repeat(v, n, axis=0)
    ns = np.hstack([np.arange(i) for i in n])

    ps = ss + vs * ns[:, None]
    if ~start:
        ps = ps[1:]
    if ~end:
        ps = ps[:-1]
    return ps

def find_min_index_in_1d_list(arr, f=None):
	# assume arr is 1d numpy array and f can return a list of bool value
    assert(len(np.shape(arr)) == 1)
    if f is None:
        return np.argmin(arr)

    l = np.arange(len(arr))
    m = f(arr)

    if np.all(~m):
        return None

    i = np.argmin(arr[m])
    return l[m][i]

def find_max_index_in_1d_list(arr, f=None):
	# assume arr is 1d numpy array and f can return a list of bool value
    assert(len(np.shape(arr)) == 1)
    if f is None:
        return np.argmax(arr)

    l = np.arange(len(arr))
    m = f(arr)

    if np.all(~m):
        return None

    i = np.argmax(arr[m])
    return l[m][i]

def intersections(starts, candidates, direction, return_details=False):

    direction = normalize(direction)  # (3,)

    # if candidates (3,)   and starts (3,)   -> V (1, 3)
    # if candidates (3,)   and starts (m, 3) -> V (m, 1, 3)
    # if candidates (n, 3) and starts (m, 3) -> V (m, n, 3)
    # if candidates (n, 3) and starts (3,)   -> V (n, 3)
    V = candidates - starts[..., None, :]

    # dot product a.k.a the distance from `start` to
    # projected points of `candidates` along with `direction`
    # D can (1,), (m, 1), (m, n) or (n,)
    D = np.sum(V * direction, axis=-1)

    # if D (1,)   and starts (3,)   -> Ps (1, 3)
    # if D (m, 1) and starts (m, 3) -> Ps (m, 1, 3)
    # if D (m, n) and starts (m, 3) -> Ps (m, n, 3)
    # if D (n,)   and starts (3,)   -> Ps (n, 3)
    Ps = starts[..., None, :] + direction * D[..., None]

    if return_details:
        return Ps, D
    return Ps