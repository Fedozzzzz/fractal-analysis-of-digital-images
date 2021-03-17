import numpy as np
import cv2


def get_image_resolution(img):
    return img.shape[0], img.shape[1]


def get_minkowski_dimension(input_img):
    # convert input image to grayscale
    img_gs = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # applying thresholding
    img = cv2.threshold(img_gs, 127, 255, cv2.THRESH_BINARY)[1] / 255

    # deltas array (δ-array)
    deltas = [1, 2, 3]

    w, h = get_image_resolution(img)

    # cell size
    n = 100

    # crop image array by cell size
    S = np.add.reduceat(
        np.add.reduceat(img, np.arange(0, w, n), axis=0),
        np.arange(0, h, n), axis=1)

    S_w = S.shape[0]
    S_h = S.shape[1]

    u = np.zeros((len(deltas) + 1, S_w + 1, S_h + 1))
    b = np.zeros((len(deltas) + 1, S_w + 1, S_h + 1))
    Vol = np.zeros(len(deltas) + 1)

    # applying MFS method
    for delta in deltas:
        for i in range(0, S_w):
            for j in range(0, S_h):
                u[0][i][j] = S[i][j]
                b[0][i][j] = S[i][j]

                # calculate the top surface (u) and bottom surface (b)
                u[delta][i][j] = max(u[delta - 1][i][j] + 1,
                                     max(u[delta - 1][i + 1][j + 1], u[delta - 1][i - 1][j + 1],
                                         u[delta - 1][i + 1][j - 1],
                                         u[delta - 1][i - 1][j - 1]))
                b[delta][i][j] = min(b[delta - 1][i][j] - 1,
                                     min(b[delta - 1][i + 1][j + 1], b[delta - 1][i - 1][j + 1],
                                         b[delta - 1][i + 1][j - 1],
                                         b[delta - 1][i - 1][j - 1]))

        # the volume of a δ-parallel body is calculated as
        summa = 0
        for i in range(0, S_w):
            for j in range(0, S_h):
                summa += u[delta][i][j] - b[delta][i][j]

        Vol[delta] = summa

    squares_A = []
    for delta in deltas:
        squares_A.append(((Vol[delta] - Vol[delta - 1]) / 2))

    # applying least square method for calculating logarithms for each cell
    res = np.polyfit(np.log(squares_A), np.log(deltas), 1)
    return 2 - -res[0]


def run_example_by_paths(paths):
    assert len(paths) != 0
    for img_path in paths:
        img = cv2.imread(img_path)
        print("{:10}: {}".format(img_path, get_minkowski_dimension(img)))


if __name__ == '__main__':
    paths = ['./images/1.jpg', './images/2.jpg', './images/3.jpg', './images/4.jpg', './images/image.jpg',
             './images/1.png', './images/m1.png', './images/russia.png']
    run_example_by_paths(paths)

    """
     RESULT IS: 
    ./images/1.jpg: 1.3833961452228705
    ./images/2.jpg: 1.7061064915712267
    ./images/3.jpg: 1.6024032195980311
    ./images/4.jpg: 1.7895351151785681
    ./images/image.jpg: 1.6464720914522937
    ./images/1.png: 1.5721023387422974
    ./images/m1.png: 0.992093955126383
    ./images/russia.png: 1.8706248059568238
    """
