#!/usr/bin/env python
# -*- coding: utf-8 -*-


import abc
import argparse
from filter import generators
import logging


class Parser:
    def __init__(self):
        self.__parser = argparse.ArgumentParser(description='Filter generator')
        self.__parser.add_argument('-o', '--output-file', metavar='output_file', help='Filter output file')
        self.__parser.add_argument('-v', '--verbosity',
                                   choices=['debug', 'info', 'warning', 'error', 'critical', 'off'],
                                   default='info')
        self.__parser.add_argument('--normalize', action='store_true', help='normalize filter')
        self.__parser.add_argument('--show', action='store_true', help='display filter and its FFT')
        self.__parser.add_argument('width', metavar='width', type=int,
                                   help='Sampling will go from -width/2 to width/2')
        self.__parser.add_argument('samples_per_unit', metavar='samples_per_unit', type=int,
                                   help='Samples count per unit')

        self.__filter_subparsers = self.__parser.add_subparsers(help='filter generator', metavar='filter_generator',
                                                                dest='filter')
        self.__filter_subparsers.required = True

    def register_filter(self, filter_generator):
        filter_generator.add_command(self.__filter_subparsers)

    def parse(self):
        return self.__parser.parse_args()


class AFilterGenerator(abc.ABC):
    @abc.abstractmethod
    def add_command(self, subparsers):
        """
        Add filter generator command
        :param subparsers: argparse subparser
        :return: None
        """
        pass

    @abc.abstractmethod
    def generate(self, command_args):
        """
        Generate the filter with the given arguments
        :param command_args: parsed args
        :return: generated filter
        """
        pass

    def support(self, command_args):
        """
        Check that the filter generate support the command arguments
        :param command_args: parsed args
        :return: True if command supported
        """
        return False

    def label(self, command_args):
        return command_args.filter


class SincFilterGenerator(AFilterGenerator):
    def support(self, command_args):
        return command_args.filter == 'sinc'

    def description(self, command_args):
        samples_count = command_args.samples_per_unit * command_args.width + 1
        return '{} ([{},{}], {}x{})'.format('Sinc', -command_args.width / 2, command_args.width / 2, samples_count,
                                            samples_count)

    def add_command(self, subparsers):
        subparsers.add_parser('sinc', help='sinc generator')

    def generate(self, command_args):
        logger = logging.getLogger(__name__)
        logger.info('generating sinc filter...')

        sincx_2d = generators.generate_sinc_2d(command_args.width, command_args.samples_per_unit)
        if command_args.normalize:
            logger.info('normalizing sinc filter...')
            return generators.normalize_filter(sincx_2d, command_args.samples_per_unit)
        else:
            return sincx_2d


class LanczosFilterGenerator(AFilterGenerator):
    def support(self, command_args):
        return command_args.filter == 'lanczos'

    def description(self, command_args):
        samples_count = command_args.samples_per_unit * command_args.width + 1
        return '{} a={} ([{},{}], {}x{})'.format('Lanczos', command_args.kernel_size, -command_args.width / 2,
                                                 command_args.width / 2, samples_count, samples_count)

    def add_command(self, subparsers):
        lanczos_parser = subparsers.add_parser('lanczos', help='lanczos generator')
        lanczos_parser.add_argument('kernel_size', metavar='kernel_size', type=int, help='Kernel size')

    def generate(self, command_args):
        logger = logging.getLogger(__name__)
        logger.info('generating lanczos filter...')

        lanczos_2d = generators.generate_lanczos_2d(command_args.width, command_args.samples_per_unit,
                                                    command_args.kernel_size)
        if command_args.normalize:
            logger.info('normalizing lanczos filter...')
            return generators.normalize_filter(lanczos_2d, command_args.samples_per_unit)
        else:
            return lanczos_2d


class BicubicFilterGenerator(AFilterGenerator):
    def support(self, command_args):
        return command_args.filter == 'bicubic'

    def description(self, command_args):
        samples_count = command_args.samples_per_unit * command_args.width + 1
        return '{} a={} ([{},{}], {}x{})'.format('Bicubic', command_args.a, -command_args.width / 2,
                                                 command_args.width / 2, samples_count, samples_count)

    def add_command(self, subparsers):
        bicubic_parser = subparsers.add_parser('bicubic', help='bicubic generator')
        bicubic_parser.add_argument('a', metavar='a', type=float, help='bicubic a parameter')

    def generate(self, command_args):
        logger = logging.getLogger(__name__)
        logger.info('generating bicubic filter...')

        bicubic_2d = generators.generate_bicubic_2d(command_args.width, command_args.samples_per_unit,
                                                    command_args.a)
        if command_args.normalize:
            logger.info('normalizing bicubic filter...')
            return generators.normalize_filter(bicubic_2d, command_args.samples_per_unit)
        else:
            return bicubic_2d


class CubicBSplineFilterGenerator(AFilterGenerator):
    def support(self, command_args):
        return command_args.filter == 'cubicbspline'

    def description(self, command_args):
        samples_count = command_args.samples_per_unit * command_args.width + 1
        return '{} ([{},{}], {}x{})'.format('Cubic B Spline', -command_args.width / 2, command_args.width / 2,
                                            samples_count, samples_count)

    def add_command(self, subparsers):
        subparsers.add_parser('cubicbspline', help='cubicbspline generator')

    def generate(self, command_args):
        logger = logging.getLogger(__name__)
        logger.info('generating cubicbspline filter...')

        cubicbspline_2d = generators.generate_cubicbspline_2d(command_args.width, command_args.samples_per_unit)
        if command_args.normalize:
            logger.info('normalizing cubicbspline filter...')
            return generators.normalize_filter(cubicbspline_2d, command_args.samples_per_unit)
        else:
            return cubicbspline_2d


class GaussianFilterGenerator(AFilterGenerator):
    def support(self, command_args):
        return command_args.filter == 'gaussian'

    def description(self, command_args):
        samples_count = command_args.samples_per_unit * command_args.width + 1
        return '{} sigma={} ([{},{}], {}x{})'.format('Gaussian', command_args.sigma, -command_args.width / 2,
                                                     command_args.width / 2, samples_count, samples_count)

    def add_command(self, subparsers):
        gaussian_parser = subparsers.add_parser('gaussian', help='gaussian generator')
        gaussian_parser.add_argument('sigma', metavar='sigma', type=float,
                                     help='Standard deviation of gaussian distribution')

    def generate(self, command_args):
        logger = logging.getLogger(__name__)
        logger.info('generating gaussian filter...')

        gaussian_2d = generators.generate_gaussian_2d(command_args.width, command_args.samples_per_unit,
                                                      command_args.sigma)
        if command_args.normalize:
            logger.info('normalizing gaussian filter...')
            return generators.normalize_filter(gaussian_2d, command_args.samples_per_unit)
        else:
            return gaussian_2d
