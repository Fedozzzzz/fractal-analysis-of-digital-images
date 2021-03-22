import numpy as np
import cv2


def get_image_resolution(img):
    return img.shape[0], img.shape[1]


def get_A(img, deltas):
    w, h = get_image_resolution(img)

    u = np.zeros((len(deltas) + 1, w + 1, h + 1))
    b = np.zeros((len(deltas) + 1, w + 1, h + 1))
    Vol = np.zeros((len(deltas) + 1))
    A = np.zeros((len(deltas) + 1))

    # applying MFS method
    # for each cell calculate Vol
    for i in range(0, w):
        for j in range(0, h):
            u[0][i][j] = img[i][j]
            b[0][i][j] = img[i][j]

            # calculate the top surface (u) and bottom surface (b)
            for delta in deltas:
                u[delta][i][j] = max(u[delta - 1][i][j] + 1,
                                     max(u[delta - 1][i + 1][j + 1], u[delta - 1][i - 1][j + 1],
                                         u[delta - 1][i + 1][j - 1],
                                         u[delta - 1][i - 1][j - 1]))
                b[delta][i][j] = min(b[delta - 1][i][j] - 1,
                                     min(b[delta - 1][i + 1][j + 1], b[delta - 1][i - 1][j + 1],
                                         b[delta - 1][i + 1][j - 1],
                                         b[delta - 1][i - 1][j - 1]))

    # the volume of a δ-parallel body is calculated as
    for delta in deltas:
        summa = 0
        for x in range(0, w):
            for y in range(0, h):
                summa += u[delta][x][y] - b[delta][x][y]
        Vol[delta] = summa

    for delta in deltas:
        A[delta] = ((Vol[delta] - Vol[delta - 1]) / 2)

    return A


def get_minkowski_dimension(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # deltas array (δ-array)
    deltas = [1, 2]

    w, h = get_image_resolution(img)

    cell_size = 50

    c_w = int(w / cell_size)
    c_h = int(h / cell_size)

    A = np.zeros((c_w, c_h, len(deltas) + 1))

    for x in range(0, c_w):
        for y in range(0, c_h):
            crop_img = img[y * cell_size:cell_size * (y + 1), x * cell_size:cell_size * (x + 1)]
            res = get_A(crop_img, deltas)
            for delta in deltas:
                A[x][y][delta] = res[delta]

    deltas_A = []
    for delta in deltas:
        s = 0
        for x in range(0, c_w):
            for y in range(0, c_h):
                s += A[x][y][delta]
        deltas_A.append(s)


    # applying least square method for calculating logarithms for each cell
    res = np.polyfit(np.log(deltas_A), np.log(deltas), 1)
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
    ./images/1.jpg: 1.625568806000594
    ./images/2.jpg: 1.7845237036139086
    ./images/3.jpg: 1.7646270934933632
    ./images/4.jpg: 1.8405488686991418
    ./images/image.jpg: 1.7326906497144254
    ./images/1.png: 1.8060886139445047
    ./images/m1.png: 1.5368203460200918
    ./images/russia.png: 1.831861523276633
    """
