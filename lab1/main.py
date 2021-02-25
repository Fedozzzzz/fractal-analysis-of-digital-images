import numpy as np
import cv2


def get_image_resolution(img):
    return img.shape[0], img.shape[1]


def get_box_count(img, k):
    w, h = get_image_resolution(img)

    # crop image by box size
    S = np.add.reduceat(
        np.add.reduceat(img, np.arange(0, w, k), axis=0),
        np.arange(0, h, k), axis=1)

    # count non-empty and non-full boxes
    return len(np.where((S > 0) & (S < k * k))[0])


def get_fractal_dimension(input_img):
    # convert input image to grayscale
    img_gs = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # applying thresholding
    img = cv2.threshold(img_gs, 127, 255, cv2.THRESH_BINARY)[1] / 255

    # minimal dimension of image
    min_dim = min(get_image_resolution(img))

    # greatest power of 2 less than or equal to min_dim
    gp = 2 ** np.floor(np.log(min_dim) / np.log(2))

    # extract the exponent
    exp = int(np.log(gp) / np.log(2))

    sizes = 2 ** np.arange(exp, 1, -1)

    # call box counting method with decreasing size
    counts = []
    for size in sizes:
        counts.append(get_box_count(img, size))

    # applying least square method and returning result
    res = np.polyfit(np.log(1.0 / sizes), np.log(counts), 1)
    return res[0]


def run_example_by_paths(paths):
    assert len(paths) != 0
    for img_path in paths:
        img = cv2.imread(img_path)
        print("{:10}: {}".format(img_path, get_fractal_dimension(img)))


if __name__ == '__main__':
    paths = ['./images/1.jpg', './images/2.jpg', './images/3.jpg', './images/4.jpg', './images/image.jpg',
             './images/1.png', './images/m1.png', './images/russia.png']
    run_example_by_paths(paths)

    """
     RESULT IS: 
    ./images/1.jpg: 1.7432035170898532
    ./images/2.jpg: 1.5108463676425543
    ./images/3.jpg: 1.7910185277891446
    ./images/4.jpg: 1.4907692413072249
    ./images/image.jpg: 1.3428497984694394
    ./images/1.png: 1.1938172240598617
    ./images/m1.png: 1.6483351742881667
    ./images/russia.png: 1.312861999042033
    """
