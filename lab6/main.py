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

    # applying least square method for calculating logarithms for each cell
    deltas_A = []
    for delta in deltas:
        s = 0
        for x in range(0, c_w):
            for y in range(0, c_h):
                s += A[x][y][delta]
        deltas_A.append(s)

    # applying least square method for calculating logarithms for each cell
    res = np.polyfit(np.log2(deltas_A), np.log2(deltas), 1)
    return -res[0]


def get_data_for_graphic(img, deltas):
    res = []
    for d in deltas:
        res.append(get_minkowski_dimension(img, deltas=np.arange(1, d + 1)))

    print(res)
    return res


def run_example_by_paths(paths):
    assert len(paths) != 0
    deltas = np.arange(2, 31)
    fig, ax = plt.subplots()

    for img_path in paths:
        img = cv2.imread(img_path)

        res = get_data_for_graphic(img, deltas)
        ax.plot(deltas, res)

    ax.set(xlabel='δ', ylabel='log(Aδ)/log(δ)')
    ax.grid()
    ax.legend(paths)
    fig.savefig("res.png")
    plt.show()


# [0.33313893962975594, 0.42876786433259056, 0.5067191851222299, 0.5738911701950545, 0.6337899537743391, 0.6885312245554028, 0.7390634217653433, 0.7862986433143457, 0.8307409332612178, 0.872928912997131, 0.9129127450744066, 0.9510553953637668, 0.9874004687593856, 1.0221108232097063, 1.0552876535726834, 1.087148691570297, 1.1177445119716383, 1.1471488746478953, 1.1754395698052134, 1.2027007930939206, 1.2289357365770157, 1.2541296887442432, 1.27836737975361, 1.3016788670742292, 1.3240975324890585, 1.3456572511862674, 1.3662754297872086, 1.3858536687273129, 1.4046705417625234]
# [0.48941905706114774, 0.618123860098284, 0.7217552720085679, 0.8086685038208042, 0.883450248588422, 0.9480087541908571, 1.0034111641510959, 1.0506475332831582, 1.0904417390820573, 1.1222573833523177, 1.147521617383064, 1.16488686328936, 1.176659989387095, 1.1845667576225551, 1.1892849994548207, 1.1918345123799556, 1.1918384166416869, 1.1892309516940702, 1.18543650719758, 1.1806485117961192, 1.1752542637388967, 1.169569416350535, 1.1636767679523452, 1.1566381066490674, 1.1497024609539797, 1.1439673672113584, 1.13809505617404, 1.1329416475865959, 1.1290682645962826]


if __name__ == '__main__':
    paths = ['./images/healthy_blood_cells.png', './images/leukemia.png']
    run_example_by_paths(paths)
