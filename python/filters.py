#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from osgeo import gdal

from filter import cli, plot


def main():
    cli_parser = cli.Parser()

    filter_generators = [cli.GaussianFilterGenerator(), cli.SincFilterGenerator(), cli.LanczosFilterGenerator(),
                         cli.BicubicFilterGenerator(), cli.CubicBSplineFilterGenerator()]

    for filter_generator in filter_generators:
        cli_parser.register_filter(filter_generator)

    command_args = cli_parser.parse()

    init_logging(command_args.verbosity)
    logger = logging.getLogger(__name__)

    generator = None
    for filter_generator in filter_generators:
        if filter_generator.support(command_args):
            generator = filter_generator
            break
    else:
        logger.error("unsupported generator {}".format(command_args.filter))
        return

    logger.info('generating {}...'.format(generator.description(command_args)))

    filter_data = generator.generate(command_args)
    image_path = command_args.output_file if command_args.output_file else '{}-{}-{}.tif'.format(command_args.filter,
                                                                                                 command_args.width,
                                                                                                 command_args.samples_per_unit)
    logger.info('output file: {}'.format(image_path))
    save_image(filter_data, image_path)

    if command_args.show:
        logger.info('display filter...')
        filter_plot = plot.FilterPlot()
        filter_plot.add_filter(filter_data, generator.description(command_args))
        filter_plot.show()


def save_image(data, output_filepath):
    driver = gdal.GetDriverByName('GTiff')
    col_count = len(data[0])
    row_count = len(data)
    band_count = 1
    dst_dataset = driver.Create(output_filepath, col_count, row_count, band_count, gdal.GDT_Float64)
    dst_dataset.GetRasterBand(1).WriteArray(data)


def init_logging(global_level):
    log_format = '[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s'
    if global_level == 'info':
        level = logging.INFO
    elif global_level == 'debug':
        level = logging.DEBUG
    elif global_level == 'warning':
        level = logging.WARNING
    elif global_level == 'error':
        level = logging.ERROR
    elif global_level == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.NOTSET
    logging.basicConfig(format=log_format, level=level)


if __name__ == "__main__":
    main()
