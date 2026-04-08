import numpy as np
import matplotlib.pyplot as plt

def vertical_line(ar):
    kernel_v = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 0]
    ])
    
    v_line = np.zeros_like(ar)
    h, w = ar.shape
    kh, kw = kernel_v.shape
    for i in range(h):
        for j in range(w):
            i_min = max(0, i - kh//2)
            i_max = min(h, i + kh//2 + 1)
            j_min = max(0, j - kw//2)
            j_max = min(w, j + kw//2 + 1)

            window = ar[i_min:i_max, j_min:j_max]
            kernel_cut = kernel_v[:i_max-i_min, :j_max-j_min]
            v_line[i, j] = (window * kernel_cut).sum()

    return v_line

def horizontal_line(ar):
    kernel_h = np.array([
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0]
    ])
    
    h_line = np.zeros_like(ar)
    h, w = ar.shape
    kh, kw = kernel_h.shape
    for i in range(h):
        for j in range(w):
            i_min = max(0, i - kh//2)
            i_max = min(h, i + kh//2 + 1)
            j_min = max(0, j - kw//2)
            j_max = min(w, j + kw//2 + 1)

            window = ar[i_min:i_max, j_min:j_max]
            kernel_cut = kernel_h[:i_max-i_min, :j_max-j_min]
            h_line[i, j] = (window * kernel_cut).sum()

    return h_line

def diagonal_line(ar):
    kernel_d = np.eye(3, 3)

    d_line = np.zeros_like(ar)
    h, w = ar.shape
    kh, kw = kernel_d.shape
    for i in range(h):
        for j in range(w):
            i_min = max(0, i - kh//2)
            i_max = min(h, i + kh//2 + 1)
            j_min = max(0, j - kw//2)
            j_max = min(w, j + kw//2 + 1)

            window = ar[i_min:i_max, j_min:j_max]
            kernel_cut = kernel_d[:i_max-i_min, :j_max-j_min]
            d_line[i,j] = (window * kernel_cut).sum()
    return d_line

def blurr_kernel(img, a):
    kernel_blurr = np.ones((3,3))

    blurr = np.zeros_like(img)
    h, w = img.shape
    kh, kw = kernel_blurr.shape
    for i in range(h):
        for j in range(w):
            i_min = max(0, i - kh//2)
            i_max = min(h, i + kh//2 + 1)
            j_min = max(0, j - kw//2)
            j_max = min(w, j + kw//2 + 1)

            window = img[i_min:i_max, j_min:j_max]
            kernel_cut = kernel_blurr[:i_max-i_min, :j_max-j_min]
            value = (window * kernel_cut).sum()
            blurr[i, j] = value * a

    return blurr

def sharpened(img):
    kernel_1 = ([
        [0, 0, 0],
        [0, 2, 0],
        [0, 0, 0]
    ])
    kernel_2 = np.ones((3, 3))
    kernel = kernel_1 - (1/9)*kernel_2

    out = np.zeros_like(img)
    h, w = img.shape
    kh, kw = kernel.shape

    for i in range(h):
        for j in range(w):
            i_min = max(0, i - kh//2)
            i_max = min(h, i + kh//2 + 1)
            j_min = max(0, j - kw//2)
            j_max = min(w, j + kw//2 + 1)

            window = img[i_min:i_max, j_min:j_max]
            kernel_cut = kernel[:i_max-i_min, :j_max-j_min]

            out[i, j] = (window * kernel_cut).sum()

    return out

def blurr_kernel_5x5(img, a):
    kernel_blurr = np.ones((5,5))

    blurr = np.zeros_like(img)
    h, w = img.shape
    kh, kw = kernel_blurr.shape

    for i in range(h):
        for j in range(w):
            i_min = max(0, i - kh//2)
            i_max = min(h, i + kh//2 + 1)
            j_min = max(0, j - kw//2)
            j_max = min(w, j + kw//2 + 1)

            window = img[i_min:i_max, j_min:j_max]
            kernel_cut = kernel_blurr[:i_max-i_min, :j_max-j_min]

            value = (window * kernel_cut).sum()
            blurr[i, j] = value * a

    return blurr

def sharpened_5x5(img):
    kernel_1 = np.zeros((5,5))
    kernel_1[2,2] = 2

    kernel_2 = np.ones((5,5)) / 25
    kernel = kernel_1 - kernel_2

    out = np.zeros_like(img)
    h, w = img.shape
    kh, kw = kernel.shape

    for i in range(h):
        for j in range(w):
            i_min = max(0, i - kh//2)
            i_max = min(h, i + kh//2 + 1)
            j_min = max(0, j - kw//2)
            j_max = min(w, j + kw//2 + 1)

            window = img[i_min:i_max, j_min:j_max]
            kernel_cut = kernel[:i_max-i_min, :j_max-j_min]

            out[i, j] = (window * kernel_cut).sum()

    return out
    