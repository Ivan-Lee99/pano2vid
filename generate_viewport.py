import numpy as np
import cv2
import matplotlib.pyplot as plt
# from skimage import io

pi = np.pi


def bicubic_interpolation(im, x_out, y_out, width, height):
    y_f = int(y_out)
    x_f = int(x_out)
    p = y_out - y_f
    q = x_out - x_f
    if y_f == 0:
        p = 0
    if y_f >= height - 1:
        y_f = height - 1
        return (1 - q) * im[y_f, np.mod(x_f - 1, width) + 1] + q * im[y_f, np.mod(x_f, width) + 1]
    else:
        return (1 - p) * (1 - q) * im[y_f, np.mod(x_f - 1, width) + 1] + (1 - p) * q * im[y_f, np.mod(x_f, width) + 1] + p * (1 - q) * im[y_f + 1, np.mod(x_f - 1, width) + 1] + p * q * im[y_f + 1, np.mod(x_f, width) + 1]


def viewport_generation(im, lon, lat, FOV):
    height = im.shape[0]
    width = im.shape[1]
    # 4:3 field of view
    F_h = FOV
    F_v = 0.75 * FOV
    viewport_width_size = np.floor(F_h / (2 * pi) * width)
    viewport_height_size = np.floor(F_v / pi * height)
    print(viewport_width_size, viewport_height_size)
    viewport = np.zeros([int(viewport_height_size), int(viewport_width_size), 3])
    R = np.array([[np.cos(lon), np.sin(lon) * np.sin(lat), np.sin(lon) * np.cos(lat)],
                  [0, np.cos(lat), - np.sin(lat)],
                  [-np.sin(lon), np.cos(lon) * np.sin(lat), np.cos(lon) * np.cos(lat)]])
    for i in range(int(viewport_height_size) - 1):
        for j in range(int(viewport_width_size) - 1):
            u = (j + 0.5) * 2 * np.tan(F_h / 2) / viewport_width_size
            v = (i + 0.5) * 2 * np.tan(F_v / 2) / viewport_height_size

            x1 = u - np.tan(F_h / 2)
            y1 = -v + np.tan(F_v / 2)
            z1 = 1.0

            r = np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2)

            sphere_coords = [x1 / r, y1 / r, z1 / r]
            rotated_sphere_coords = np.matmul(R, sphere_coords)

            x = rotated_sphere_coords[0]
            y = rotated_sphere_coords[1]
            z = rotated_sphere_coords[2]

            theta = np.arccos(y)
            phi = np.arctan2(x, z)

            x_out = width * phi / (2 * np.pi)
            y_out = height * theta / np.pi

            viewport[i, j] = bicubic_interpolation(im, x_out, y_out, width, height)

    return viewport


if __name__ == '__main__':
    img = cv2.imread('./waste_bin/245.jpg')
    viewport = viewport_generation(img, pi, 0, 0.75 * pi)
    viewport = viewport / viewport.max()
    viewport = viewport * 255
    viewport = viewport.astype(np.uint8)
    cv2.imshow('result', viewport)
    cv2.waitKey(0)
