import numpy as np
import pandas as pd
import math as mt
import matplotlib.pyplot as plt
from PIL import Image


def shuffle_arr(A, block_size=(16, 16)):
    M, N = A.shape
    m, n = block_size
    height, width = ((M - (M % m)) * (N - (N % n))) // (m * n), m * n
    X = np.zeros((height, width))
    for i in range(height):
        s_x, s_y = (i * n) % (N - (N % n)), i // m * m
        f_x, f_y = s_x + n, s_y + m
        TMP = A[s_y:f_y, s_x:f_x]
        if TMP.shape != block_size:
            continue
        X[i] = TMP.reshape(1, -1)
    return X


def reshuffle_arr(X, block_size=(16, 16)):
    m, n = block_size
    M, N = X.shape
    height, width = M // m, N // n
    A = [None] * height
    for i in range(len(X)):
        TMP = X[i].reshape(block_size)
        if (i * n) % N == 0:
            A[(i * n) // N] = TMP.copy()
        else:
            A[(i * n) // N] = np.hstack([A[(i * n) // N], TMP])
    res = A[0]
    for i in range(1, len(A)):
        res = np.vstack([res, A[i]])
    return res


def dot_D_A(d, c):
    return np.array([c[i] * d[i] for i in range(len(d))])


def SVD(A):
    AAT = np.dot(A, A.T)
    ATA = np.dot(A.T, A)

    eigvals_ATA, eigvecs_ATA = np.linalg.eig(ATA)

    eigvecs_ATA = eigvecs_ATA.T
    eigvecs_ATA = np.array(
        [x for _, x in sorted(zip(eigvals_ATA, eigvecs_ATA), key=lambda pair: pair[0], reverse=True)])
    eigvecs_ATA = eigvecs_ATA.T

    eigvals_ATA = np.array([x for x in eigvals_ATA if x > 1e-8])
    sing_ATA = np.sqrt(eigvals_ATA)

    S = np.array(sorted(sing_ATA, reverse=True))  # HURRAY

    VT = eigvecs_ATA.T  # HURRAY

    UT = np.zeros((len(S), len(A)))
    for i in range(len(S)):
        d = np.dot((1 / S[i]), A)
        UT[i] = np.dot(d, VT[i])
    U = UT.T  # HURRAY
    return U, S, VT


def apply_rank(U, S, VT, r):
    if r is None:
        r = len(S)
    S_r = S[:r]
    U_r = U[:, :r]
    VT_r = VT[:r]
    print("SVD shape:", r, U_r.shape, S_r.shape, VT_r.shape)
    return U_r, S_r, VT_r


def SVD_to_A(U, S, VT):
    A_RECOVERED = np.dot(U, dot_D_A(S, VT))
    return A_RECOVERED


def load_img(path='lena.jpg'):
    im = Image.open(path)
    return im


def img_to_arr_colored(im):
    height, width = im.size
    arr = np.array(im.getdata())
    arr = arr.T
    r = arr[0]
    g = arr[1]
    b = arr[2]
    print(r.shape)
    r = r.reshape(height, width)
    g = g.reshape(height, width)
    b = b.reshape(height, width)
    return r, g, b


def img_to_arr_grey(im):
    height, width = im.size
    arr = np.array(im.getdata())
    arr = arr.reshape(height, width)
    return arr


def arr_to_img(arr):
    return Image.fromarray(arr)


def show_img(im):
    plt.imshow(im, cmap='gray')
    plt.show()


def SSVD(file_name, rank=None, im_type='gray'):
    if im_type.lower() == 'rgb':
        return SSVD_colored(file_name, rank)
    if im_type.lower() == 'gray' or im_type.lower() == 'grey':
        return SSVD_gray(file_name, rank)


def SSVD_colored(file_name, rank=None):
    im = load_img(file_name)
    h, w = im.size
    # arr2 = np.array(im)
    # arr2 = arr2.reshape(w, h, 3)
    # im2 = arr_to_img(np.uint8(arr2))
    # show_img(im2)
    # print(im2.shape)
    r, g, b = img_to_arr_colored(im)
    res = []
    for arr in (r, g, b):
        # arr = shuffle_arr(arr)
        U, S, VT = SVD(arr)
        U_r, S_r, VT_r = apply_rank(U, S, VT, rank)
        arr = SVD_to_A(U_r, S_r, VT_r)
        # arr = reshuffle_arr(arr)
        res.append(arr)
    new_im = np.array(res)
    new_im = new_im.reshape(w, h, 3)
    return arr_to_img(np.uint8(new_im))


def SSVD_gray(file_name, rank=None):
    im = load_img(file_name)
    arr = img_to_arr_grey(im)
    # arr = shuffle_arr(arr)
    U, S, VT = SVD(arr)
    U_r, S_r, VT_r = apply_rank(U, S, VT, rank)
    arr = SVD_to_A(U_r, S_r, VT_r)
    # arr = reshuffle_arr(arr)
    return arr_to_img(arr)


new_im = SSVD('lena.jpg', im_type='gray')
show_img(new_im)