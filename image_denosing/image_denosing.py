import os

import numpy as np
import random
from matplotlib import pyplot as plt
from PIL import Image
import scipy


def simple_gibbs(s, x, prob_func):
    res = []
    x = np.array(x)
    d = x.shape[0]
    for t in range(s):
        for i in range(d):
            if np.random.uniform() < prob_func(i, x):
                x[i] = 1
            else:
                x[i] = -1
        res.append(x.copy())
    return res


def generate_nb(w,h):
    d = w*h
    nb = []

    def from2d(x, y):
        return int(x * w + y)

    def to2d(i):
        return i // w, i % w

    def valid(x, y):
        return x >= 0 and x < h and y >= 0 and y < w

    for i in range(d):
        nbi = []
        x, y = to2d(i)
        if valid(x + 1, y):
            nbi.append(from2d(x + 1, y))
        if valid(x - 1, y):
            nbi.append(from2d(x - 1, y))
        if valid(x, y + 1):
            nbi.append(from2d(x, y + 1))
        if valid(x, y - 1):
            nbi.append(from2d(x, y - 1))
        nb.append(nbi)
    return nb


def load_img(path) -> np.ndarray:
    img = Image.open(path)
    res = np.array(img.convert('L').getdata())
    res = np.where(res > 128, 1, -1)
    return res, img.size


def save_img(img, path):
    Image.fromarray(img * 255).convert('L').save(path)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def denoise_prob(i, theta_1, theta_2, x, nb):
    return sigmoid(2 * (theta_1[i] + np.sum([theta_2 * x[j] for j in nb[i]])))


def denoise_img(input_path, output_path, theta_1=0.4, theta_2=0.4, sample_step=100, show_fig=False, save_fig=True, output_intermediate=False, prob_func=denoise_prob):
    noisy_img, (width, height) = load_img(input_path)
    nbs = generate_nb(width, height)
    processed_img = np.array(simple_gibbs(sample_step, noisy_img, lambda i, x: prob_func(i, theta_1 * noisy_img, theta_2, x, nbs)))
    final_img = np.mean(processed_img, axis=0)
    final_img_show = (final_img.reshape(height, -1) + 1) / 2
    if save_fig:
        save_img(final_img_show, output_path)
    if show_fig:
        plt.imshow((final_img_show.reshape(height, -1) + 1) / 2, 'gray')
        plt.show()
    plt.close()
    res = final_img
    if output_intermediate:
        res = [final_img, processed_img]
    return res

if __name__ == '__main__':
    denoise_img('im_noisy.png', 'im_cleaned.png')