import cv2
import numpy as np
import argparse


def shearImage(image: np.ndarray, alpha: float, bgcolor: int) -> np.ndarray:
    result_size = (int(image.shape[1] + np.ceil(abs(alpha*image.shape[0]))), image.shape[0])
    sheared_image = np.zeros(result_size)
    shiftX = max(-alpha*image.shape[0], 0)
    transform_matrix = np.array([[1, alpha, shiftX],
                                 [0,     1,      0]], dtype=np.float)
    sheared_image = cv2.warpAffine(image, transform_matrix, result_size, sheared_image,
                                   borderMode=cv2.INTER_NEAREST, borderValue=bgcolor)
    return sheared_image


def deslantImage(image: np.ndarray, bgcolor: int = 255):
    alphaVals = np.arange(-1, 1.1, 0.25)

    best_alpha = -1
    max_sum_alpha = 0

    for alpha in alphaVals:
        sum_alpha = 0
        sheared_image: np.ndarray = shearImage(image, alpha, bgcolor)
        fg = sheared_image == 0
        h_alpha = fg.sum(axis=0)
        for col in range(fg.shape[1]):
            indexes = np.nonzero(fg[:, col])[0]
            if len(indexes) != 0:
                d_y_alpha = np.max(indexes) - np.min(indexes)
                if d_y_alpha == h_alpha[col]:
                    sum_alpha += h_alpha[col]**2
        if sum_alpha > max_sum_alpha:
            max_sum_alpha = sum_alpha
            best_alpha = alpha

    result = shearImage(image, best_alpha, bgcolor)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('-o', '--output', type=str, default='out.png')
    parser.add_argument('-b', '--background', type=int, default=255)
    args = parser.parse_args()

    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    result = deslantImage(image, args.background)
    cv2.imwrite(args.output, result)