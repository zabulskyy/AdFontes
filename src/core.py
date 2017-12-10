import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


def shuffle_arr(A, block_size=(16, 16), verbose=False):
    """
    shuffle array A
    :param A:
    :param block_size:
    :param verbose:
    :return:
    """
    M, N = A.shape
    m, n = block_size
    X = []
    for i in range(M // m):
        a = i * m
        for j in range(N // n):
            b = j * n
            cell = A[a:a + m, b:b + n]
            cell = cell.reshape((1, -1))
            X.append(cell[0])
    X = np.array(X)
    return X


def reshuffle_arr(X, old_size, block_size=(16, 16), verbose=False):
    m, n = block_size
    M, N = old_size
    M, N = M - (M % m), N - (N % n)
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


def SVD(A, verbose=False):
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


def apply_rank(U, S, VT, r, verbose=False):
    if r is None:
        r = len(S)
    S_r = S[:r]
    U_r = U[:, :r]
    VT_r = VT[:r]
    if verbose:
        print("Rank:", r, "SVD shape:", U_r.shape, S_r.shape, VT_r.shape)
    return U_r, S_r, VT_r


def SVD_to_A(U, S, VT):
    A = np.dot(U, dot_D_A(S, VT))
    return A


def rgb_to_grey(arr, size, verbose=False):

    def weighted_average(pixel):
        return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]

    h, w = size
    arr = arr.reshape(h, w, 3)
    grey = np.zeros((h, w))

    for row in range(len(arr)):
        for column in range(len(arr[row])):
            grey[row][column] = weighted_average(arr[row][column])
    grey = grey.reshape(w, h)
    return grey


def load_img(path='lena.jpg', verbose=False):
    return Image.open(path)


def colored_img_to_arr(image, verbose=False):
    height, width = image.size
    arr = np.array(image.getdata())
    arr = arr.reshape(3, height, width)
    r = arr[0]
    g = arr[1]
    b = arr[2]
    return r, g, b


def grey_img_to_arr(image, verbose=False):
    try:
        w, h = image.size
        arr = np.array(image.getdata())
        arr = rgb_to_grey(arr, (h, w), verbose=verbose)
        if verbose:
            print("Converted from RGB to grayscale")
    except:
        height, width = image.size
        arr = np.array(image.getdata())
        arr = arr.reshape(height, width)
    return arr


def arr_to_img(arr, verbose=False):
    return Image.fromarray(arr)


def show_img(image):
    plt.imshow(image)
    plt.show()


def compressor(file_name, rank=None, im_type='gray', compressor_type="SSVD", verbose=False):
    shuffled = compressor_type.lower() == "ssvd"
    block_size = (16, 16)
    if verbose:
        print("\nImage processing...\n")
    if im_type.lower() == 'rgb':
        return compressor_for_color_img(file_name, rank, block_size=block_size, shuffled=shuffled, verbose=verbose)
    if im_type.lower() == 'gray' or im_type.lower() == 'grey':
        return compressor_for_gray_img(file_name, rank, block_size=block_size, shuffled=shuffled, verbose=verbose)


def compressor_for_color_img(file_name, rank=None, block_size=None, shuffled=True, verbose=False):
    image = load_img(file_name, verbose=verbose)
    height, width = image.size
    if shuffled:
        height, width = image.size
        square_root = int((height * width) ** 0.5)
        image = image.resize((square_root, square_root))
    r, g, b = colored_img_to_arr(image, verbose=verbose)
    res = []
    for i, arr in enumerate((r, g, b)):
        if verbose:
            if i == 0:
                print("Red color processing..")
            elif i == 1:
                print("Green color processing..")
            else:
                print("Blue color processing..")
        if shuffled:
            arr = shuffle_arr(arr, block_size=block_size, verbose=verbose)
        U, S, VT = SVD(arr, verbose=verbose)
        U_r, S_r, VT_r = apply_rank(U, S, VT, rank, verbose=verbose)
        arr = SVD_to_A(U_r, S_r, VT_r)
        if shuffled:
            arr = reshuffle_arr(arr, (square_root, square_root), block_size=block_size, verbose=verbose)
        arr[(arr > 255)] = 255
        arr[(arr < 0)] = 0
        res.append(arr)
    new_h, new_w = res[0].shape
    new_im = np.array(res)
    new_im = new_im.reshape(new_w, new_h, 3)
    image = arr_to_img(np.uint8(new_im), verbose=verbose)
    if shuffled:
        image = image.resize((height, width))
    return image


def compressor_for_gray_img(file_name, rank=None, block_size=None, shuffled=True, verbose=False):
    square_root = None
    image = load_img(file_name, verbose=verbose)
    height, width = image.size
    if shuffled:
        height, width = image.size
        square_root = int((height * width) ** 0.5)
        image = image.resize((square_root, square_root))
    arr = grey_img_to_arr(image, verbose=verbose)
    if shuffled:
        arr = shuffle_arr(arr, block_size=block_size, verbose=verbose)
    U, S, VT = SVD(arr, verbose=verbose)
    U_r, S_r, VT_r = apply_rank(U, S, VT, rank, verbose=verbose)
    arr = SVD_to_A(U_r, S_r, VT_r)
    if shuffled:
        arr = reshuffle_arr(arr, (square_root, square_root), block_size=block_size, verbose=verbose)
        image = arr_to_img(arr, verbose=verbose)
        image = image.resize((height, width))
        return image
    height, width = arr.shape
    arr = arr.reshape(width, height)
    return arr_to_img(arr, verbose=verbose)


not_shuffled_im = compressor('grid.jpg', im_type='gray', rank=100, compressor_type="SVD", verbose=False)
shuffled_im = compressor('grid.jpg', im_type='gray', rank=100, compressor_type="SSVD", verbose=False)
show_img(not_shuffled_im)
show_img(shuffled_im)
