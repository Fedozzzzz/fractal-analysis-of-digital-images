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

    return A[deltas[len(deltas) - 1]]


def segment_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # deltas array (δ-array)
    deltas = [1, 2]

    w, h = get_image_resolution(img)

    cell_size = 15

    c_w = int(w / cell_size)
    c_h = int(h / cell_size)

    A = np.zeros((c_w, c_h))

    # calculate segments surface area
    for x in range(0, c_w):
        for y in range(0, c_h):
            crop_img = img[x * cell_size:cell_size * (x + 1), y * cell_size:cell_size * (y + 1)]
            A[x][y] = get_A(crop_img, deltas)

    # getting threshold for segmentation
    threshold = np.mean(A)

    # applying segmentation to the input document
    # white color - text or graphics
    # black color - background
    for x in range(0, c_w):
        for y in range(0, c_h):
            if A[x][y] > threshold:
                cv2.rectangle(img, (y * cell_size, x * cell_size), (cell_size * (y + 1), cell_size * (x + 1)), (255, 255, 255), -1)
            else:
                cv2.rectangle(img, (y * cell_size, x * cell_size), (cell_size * (y + 1), cell_size * (x + 1)),
                              (0, 0, 255), -1)

    # save result ro a file
    cv2.imwrite('./results/segmented_image.jpg', img)

def run_example_by_paths(paths):
    assert len(paths) != 0
    for img_path in paths:
        img = cv2.imread(img_path)
        print("{:10}: {}".format(img_path, segment_image(img)))


if __name__ == '__main__':
    paths = ['./images/doc.png']
    run_example_by_paths(paths)
