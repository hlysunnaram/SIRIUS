#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
import numpy as np


def normalize_filter(filter_data, oversampling):
    """
    Normalize the filter with the given oversampling
    :param filter_data: filter to normalize
    :param oversampling: data oversampling
    :return:
    """
    for i in range(oversampling):
        filter_data[i::oversampling] /= (np.sum(filter_data[i::oversampling]) * oversampling)
    return filter_data


def generate_sinc_2d(width, samples_per_unit):
    """
    Generate 2D cardinal sine function from -width/2 to width/2
    :param width: width
    :param samples_per_unit: samples per unit
    :return: 2D numpy array
    """
    logger = logging.getLogger(__name__)
    logger.info('generating sinc 2D filter...')

    x = _generate_x(width, samples_per_unit)
    sincx = np.sinc(x)
    return _create_2d_separable_func(sincx)


def generate_lanczos_1d(width, samples_per_unit, a):
    """
    Generate 1D Lanczos from -width/2 to width/2
    :param width: width
    :param samples_per_unit: samples per unit
    :param a: kernel size
    :return: 1D numpy array
    """
    logger = logging.getLogger(__name__)
    logger.debug('generating lanczos 1D filter (a={})...'.format(a))

    x = _generate_x(width, samples_per_unit)

    sincx = np.sinc(x)
    sincx_a = np.sinc(x / a)
    rect_lanczos = np.zeros(sincx_a.size)
    rect_lanczos[np.where((-a < x) * (x < a))] = 1
    return sincx * sincx_a * rect_lanczos


def generate_lanczos_2d(width, samples_per_unit, a):
    """
    Generate 2D Lanczos from -width/2 to width/2
    :param width: width
    :param samples_per_unit: samples per unit
    :param a: kernel size
    :return: 2D numpy array
    """
    logger = logging.getLogger(__name__)
    logger.debug('generating lanczos 2D filter (a={})...'.format(a))

    lanczos_1d = generate_lanczos_1d(width, samples_per_unit, a)
    return _create_2d_separable_func(lanczos_1d)


def generate_bicubic_1d(width, samples_per_unit, a):
    """
    Generate 1D Bicubic from -width/2 to width/2
    :param width: width
    :param samples_per_unit: samples per unit
    :param a: a
    :return: 1D numpy array
    """
    logger = logging.getLogger(__name__)
    logger.debug('generating bicubic 1D filter (a={})...'.format(a))

    x = _generate_x(width, samples_per_unit)

    bicubic_1d = np.zeros(x.size)
    for i, xi in enumerate(x):
        abs_xi = abs(xi)
        if abs_xi <= 1:
            # (a+2) * |xi|^3 - (a+3) * |xi|^2 + 1
            bicubic_1d[i] = ((a + 2) * abs_xi * abs_xi * abs_xi) - ((a + 3) * abs_xi * abs_xi) + 1
        elif 1 < abs_xi <= 2:
            # a * |xi|^3 - 5a |xi|^2 + 8a |xi| - 4a
            bicubic_1d[i] = (a * abs_xi * abs_xi * abs_xi) - (5 * a * abs_xi * abs_xi) + (8 * a * abs_xi) - (4 * a)
        else:
            bicubic_1d[i] = 0
    return bicubic_1d


def generate_bicubic_2d(width, samples_per_unit, a):
    """
    Generate 2D Bicubic from -width/2 to width/2
    :param width: width
    :param samples_per_unit: samples per unit
    :param a: a
    :return: 2D numpy array
    """
    logger = logging.getLogger(__name__)
    logger.debug('generating bicubic 2D filter (a={})...'.format(a))

    bicubic_1d = generate_bicubic_1d(width, samples_per_unit, a)
    return _create_2d_separable_func(bicubic_1d)


def generate_cubicbspline_1d(width, samples_per_unit):
    """
    Generate 1D Cubic B Spline from -width/2 to width/2
    :param width: width
    :param samples_per_unit: samples per unit
    :return: 1D numpy array
    """
    logger = logging.getLogger(__name__)
    logger.debug('generating cubic B Spline 1D filter...')

    x = _generate_x(width, samples_per_unit)
    cubicbspline_1d = np.zeros(x.size)
    for i, xi in enumerate(x):
        abs_xi = abs(xi)
        if 0 <= abs_xi < 1:
            # 2/3 - |xi|^2 + (|xi|^3 / 2)
            cubicbspline_1d[i] = 2 / 3 - (abs_xi * abs_xi) + ((abs_xi * abs_xi * abs_xi) / 2)
        elif 1 <= abs_xi < 2:
            # (2 - |xi|^3) / 6
            cubicbspline_1d[i] = (2 - abs_xi * abs_xi * abs_xi) / 6
        else:
            cubicbspline_1d[i] = 0
    return cubicbspline_1d


def generate_cubicbspline_2d(width, samples_per_unit):
    """
    Generate 2D Cubic B Spline from -width/2 to width/2
    :param width: width
    :param samples_per_unit: samples per unit
    :return: 2D numpy array
    """
    logger = logging.getLogger(__name__)
    logger.debug('generating cubic B Spline 2D filter...')

    cubicbspline_1d = generate_cubicbspline_1d(width, samples_per_unit)
    return _create_2d_separable_func(cubicbspline_1d)


def generate_gaussian_2d(width, samples_per_unit, sigma):
    """
    Generate 2D Gaussian from -width/2 to width/2
    :param width: width
    :param samples_per_unit: samples per unit
    :param sigma: standard deviation
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.debug('generating gaussian 2D filter (sigma={})...'.format(sigma))

    x = _generate_x(width, samples_per_unit)
    gaussian_2d = np.zeros([x.size, x.size])
    sigma_squared = sigma * sigma
    factor = 1 / (2 * np.pi * sigma_squared)
    for p_x in range(x.size):
        for p_y in range(x.size):
            gaussian_2d[p_x][p_y] = factor * np.exp(-(x[p_x] * x[p_x] + x[p_y] * x[p_y]) / (2 * sigma_squared))
    return gaussian_2d


def _generate_x(width, samples_per_unit):
    """
    Generate x from -width/2 to width/2 with width * samples_per_unit + 1 total samples
    :param width: width
    :param samples_per_unit: samples per unit
    :return: 1D numpy array
    """
    logger = logging.getLogger(__name__)

    samples_count = width * samples_per_unit + 1
    start = -width / 2
    stop = width / 2
    logger.debug('generating x: {} samples from {} to {}...'.format(samples_count, start, stop))
    return np.linspace(start, stop, samples_count)


def _create_2d_separable_func(func_1d):
    func_2d = np.zeros([func_1d.size, func_1d.size])
    for p_x in range(func_1d.size):
        for p_y in range(func_1d.size):
            func_2d[p_x][p_y] = func_1d[p_x] * func_1d[p_y]
    return func_2d
