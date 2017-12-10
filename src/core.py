import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


def _shuffle_arr(A, block_size=(16, 16), verbose=False):
    """
    preprocessing shuffle array A due to SSVD algorithm
    breaking on blocks and stretch each block to row
    :param A: numpy.array
    :param block_size: tuple(int, int)
    :param verbose: boolean
    :return: numpy.array
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


def _reshuffle_arr(X, old_size, block_size=(16, 16), verbose=False):
    """
    postprocessing shuffle array A due to SSVD algorithm
    :param X: numpy.array
    :param old_size: tuple(int, int)
    :param block_size: tuple(int, int)
    :param verbose: boolean
    :return: numpy.array
    """
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


def _dot_D_A(d, A):
    """
    dot product of diagonal matrix (one-dimensional) and matrix
    :param d: numpy.array
    :param A: numpy.array
    :return: numpy.array
    """
    return np.array([A[i] * d[i] for i in range(len(d))])


def SVD(A, verbose=False):
    """
    SVD algorithm
    :param A: numpy.array
    :param verbose: boolean
    :return: numpy.array, numpy.array, numpy.array
    """

    # (A^T)A
    ATA = np.dot(A.T, A)

    # find eigenvectors and eigenvalues of (A^T)A
    eigvals_ATA, eigvecs_ATA = np.linalg.eig(ATA)

    # eigenvectors are columns of a matrix eigvecs_ATA, but it is easier to iterate rows
    eigvecs_ATA = eigvecs_ATA.T

    # sort eigenvectors with respect to sorted eigenvalues
    eigvecs_ATA = np.array(
        [x for _, x in sorted(zip(eigvals_ATA, eigvecs_ATA), key=lambda pair: pair[0], reverse=True)])

    # back to proper form
    eigvecs_ATA = eigvecs_ATA.T

    # get rid of negative eigenvalues and eigenvalues close to zero
    eigvals_ATA = np.array([x for x in eigvals_ATA if x > 1e-8])

    # find singular values
    sing_ATA = np.sqrt(eigvals_ATA)

    # find S
    S = np.array(sorted(sing_ATA, reverse=True))

    # find V^T
    VT = eigvecs_ATA.T

    # find U^T from equation A = U*S*V.T
    UT = np.zeros((len(S), len(A)))
    for i in range(len(S)):
        d = np.dot((1 / S[i]), A)
        UT[i] = np.dot(d, VT[i])

    # find U
    U = UT.T
    return U, S, VT


def _apply_rank(U, S, VT, r, verbose=False):
    """
    apply rank r due to SVD algorithm
    :param U: numpy.array
    :param S: numpy.array
    :param VT: numpy.array
    :param r: int
    :param verbose: boolean
    :return: numpy.array, numpy.array, numpy.array
    """
    if r is None:
        r = len(S)
    S_r = S[:r]
    U_r = U[:, :r]
    VT_r = VT[:r]
    if verbose:
        print("Rank:", r, "SVD shape:", U_r.shape, S_r.shape, VT_r.shape)
    return U_r, S_r, VT_r


def _SVD_to_A(U, S, VT):
    """
    multiply U, S and VT, due to last step of SVD algorithm
    :param U: numpy.array
    :param S: numpy.array
    :param VT: numpy.array
    :return: numpy.array
    """
    A = np.dot(U, _dot_D_A(S, VT))
    return A


def _rgb_to_grey(arr, size, verbose=False):
    """
    convert color RGB array to greyscale array
    :param arr: numpy.array
    :param size: tuple(int, int)
    :param verbose: boolean
    :return: numpy.array
    """

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


def _load_img(path='lena.jpg', verbose=False):
    """
    open image
    :param path: string
    :param verbose: boolean
    :return: Image
    """
    return Image.open(path)


def _colored_img_to_arr(image, verbose=False):
    """
    convert RGB Image to an array
    :param image: Image
    :param verbose: boolean
    :return:
    """
    height, width = image.size
    arr = np.array(image.getdata())
    arr = arr.reshape(3, height, width)
    r = arr[0]
    g = arr[1]
    b = arr[2]
    return r, g, b


def _grey_img_to_arr(image, verbose=False):
    """
    convert greyscale Image to an array
    :param image: Image
    :param verbose: boolean
    :return:
    """
    try:
        w, h = image.size
        arr = np.array(image.getdata())
        arr = _rgb_to_grey(arr, (h, w), verbose=verbose)
        if verbose:
            print("Converted from RGB to grayscale")
    except:
        height, width = image.size
        arr = np.array(image.getdata())
        arr = arr.reshape(height, width)
    return arr


def _arr_to_img(arr, verbose=False):
    """
    convert arr to an Image
    :param arr: Image
    :param verbose: boolean
    :return: Image
    """
    return Image.fromarray(arr)


def show_img(image):
    """
    print Image
    :param image: Image
    :return: Image
    """
    plt.imshow(image)
    plt.show()
    return image


def _compressor_for_color_img(file_name, rank=None, block_size=None, shuffled=True, verbose=False):
    """
    compressor for color images
    :param file_name: string
    :param rank: int
    :param block_size: tuple(int, int)
    :param shuffled: boolean
    :param verbose: boolean
    :return: Image
    """
    image = _load_img(file_name, verbose=verbose)
    height, width = image.size
    if shuffled:
        height, width = image.size
        square_root = int((height * width) ** 0.5)
        image = image.resize((square_root, square_root))
    r, g, b = _colored_img_to_arr(image, verbose=verbose)
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
            arr = _shuffle_arr(arr, block_size=block_size, verbose=verbose)
        U, S, VT = SVD(arr, verbose=verbose)
        U_r, S_r, VT_r = _apply_rank(U, S, VT, rank, verbose=verbose)
        arr = _SVD_to_A(U_r, S_r, VT_r)
        if shuffled:
            arr = _reshuffle_arr(arr, (square_root, square_root), block_size=block_size, verbose=verbose)
        arr[(arr > 255)] = 255
        arr[(arr < 0)] = 0
        res.append(arr)
    new_h, new_w = res[0].shape
    new_im = np.array(res)
    new_im = new_im.reshape(new_w, new_h, 3)
    image = _arr_to_img(np.uint8(new_im), verbose=verbose)
    if shuffled:
        image = image.resize((height, width))
    return image


def _compressor_for_gray_img(file_name, rank=None, block_size=None, shuffled=True, verbose=False):
    """
    compressor for grey images
    :param file_name: string
    :param rank: int
    :param block_size: tuple(int, int)
    :param shuffled: boolean
    :param verbose: boolean
    :return: Image
    """
    square_root = None
    image = _load_img(file_name, verbose=verbose)
    height, width = image.size
    if shuffled:
        height, width = image.size
        square_root = int((height * width) ** 0.5)
        image = image.resize((square_root, square_root))
    arr = _grey_img_to_arr(image, verbose=verbose)
    if shuffled:
        arr = _shuffle_arr(arr, block_size=block_size, verbose=verbose)
    U, S, VT = SVD(arr, verbose=verbose)
    U_r, S_r, VT_r = _apply_rank(U, S, VT, rank, verbose=verbose)
    arr = _SVD_to_A(U_r, S_r, VT_r)
    if shuffled:
        arr = _reshuffle_arr(arr, (square_root, square_root), block_size=block_size, verbose=verbose)
        image = _arr_to_img(arr, verbose=verbose)
        image = image.resize((height, width))
        return image
    height, width = arr.shape
    arr = arr.reshape(width, height)
    return _arr_to_img(arr, verbose=verbose)


def compressor(file_name, rank=None, im_type='gray', compressor_type="SSVD", verbose=False):
    """
    main function
    calls functions, depending on arguments
    :param file_name: string
    :param rank: int: rank, which will be applied in the SVD algorithm
    :param im_type: "rgb" | "grey"
    :param compressor_type: "SVD" | "SSVD"
    :param verbose: boolean: use if you want to see all prints
    :return: Image
    """
    shuffled = compressor_type.lower() == "ssvd"
    block_size = (16, 16)
    if verbose:
        print("\nImage processing...\n")
    if im_type.lower() == 'rgb':
        return _compressor_for_color_img(file_name, rank, block_size=block_size, shuffled=shuffled, verbose=verbose)
    if im_type.lower() == 'gray' or im_type.lower() == 'grey':
        return _compressor_for_gray_img(file_name, rank, block_size=block_size, shuffled=shuffled, verbose=verbose)
