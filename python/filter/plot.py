#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np


class FilterPlot:
    def add_filter(self, filter_data, label):
        shifted_filter_data = np.fft.ifftshift(filter_data)
        shifted_filter_fft = np.fft.fft2(shifted_filter_data)

        figure = plt.figure()
        figure.suptitle(label)

        # 3D filter data
        ax = figure.add_subplot(2, 2, 1, projection='3d',
                                adjustable='box', aspect=0.3)
        ax.set_title('3D')
        FilterPlot.__plot_3d(ax, filter_data)

        # 2D filter data
        ax = figure.add_subplot(2, 2, 2)
        ax.set_title('2D')
        ax.set_axis_off()
        ax.imshow(filter_data, cmap='gist_ncar')

        # FFT
        ax = figure.add_subplot(2, 2, 3)
        ax.set_title('FFT')
        ax.set_axis_off()
        ax.imshow(shifted_filter_fft.real, cmap='gist_ncar')

        # shifted FFT
        ax = figure.add_subplot(2, 2, 4)
        ax.set_title('shifted FFT')
        ax.set_axis_off()
        ax.imshow(np.fft.fftshift(shifted_filter_fft).real, cmap='gist_ncar')

    def show(self):
        plt.show()

    @staticmethod
    def __plot_3d(ax, z_data):
        x_data = range(len(z_data))
        x, y = np.meshgrid(x_data, x_data)
        ax.plot_surface(x, y, z_data, rstride=1, cstride=1, cmap='gist_rainbow', edgecolor='none', shade=True)
