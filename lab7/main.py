import numpy as np
import cv2
import matplotlib.pyplot as plt


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


def get_minkowski_dimension(image, cell_size=50, deltas=None):
    if deltas is None:
        deltas = [1, 2]

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    w, h = get_image_resolution(img)

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

    res = np.polyfit(np.log(deltas_A), np.log(deltas), 1)
    return 2 - -res[0]


def get_data_for_graphic(img, cell_sizes):
    res = []
    for cs in cell_sizes:
        res.append(get_minkowski_dimension(img, cell_size=cs))

    return res


def run_example_by_paths(paths):
    assert len(paths) != 0
    cell_sizes = np.flip(2 ** np.arange(8, 0, -1))
    fig, ax = plt.subplots()

    for img_path in paths:
        img = cv2.imread(img_path)

        res = get_data_for_graphic(img, cell_sizes)
        ax.plot(cell_sizes, res)

    ax.set(xlabel='размер ячейки (px)', ylabel='Размерность Минковского')
    ax.grid()
    ax.legend(paths)
    fig.savefig("res.png")
    plt.show()


if __name__ == '__main__':
    paths = ['./images/healthy_blood_cells.png', './images/leukemia.png']
    run_example_by_paths(paths)
