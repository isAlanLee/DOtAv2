# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
YAML loading helpers for OpenCOOD-style configuration files.

The training configs only store high-level lidar ranges and voxel sizes. The
parser functions below fill in derived geometry fields that downstream
preprocessors, postprocessors, and models expect.
"""

import math
import os
import re

import numpy as np
import yaml


def load_yaml(file, opt=None):
    """
    Load a yaml file and optionally apply the parser named by ``yaml_parser``.

    Parameters
    ----------
    file : str
        Path to the yaml file.
    opt : argparse.Namespace, optional
        If ``opt.model_dir`` is set, load ``config.yaml`` from that checkpoint
        directory. This is the convention used by inference and resumed
        training in OpenCOOD.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if opt and getattr(opt, 'model_dir', None):
        file = os.path.join(opt.model_dir, 'config.yaml')

    with open(file, 'r', encoding='utf-8') as stream:
        loader = yaml.Loader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        param = yaml.load(stream, Loader=loader)

    parser_name = param.get('yaml_parser')
    if parser_name:
        parser = _PARSER_REGISTRY.get(parser_name)
        if parser is None:
            raise ValueError('Unsupported yaml_parser: %s' % parser_name)
        param = parser(param)

    return param


def load_voxel_params(param):
    """
    Fill derived geometry values for VoxelNet-style configs.
    """
    anchor_args = param['postprocess']['anchor_args']
    cav_lidar_range = anchor_args['cav_lidar_range']
    voxel_size = param['preprocess']['args']['voxel_size']

    vw, vh, vd = voxel_size
    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd
    anchor_args['W'] = int((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args['H'] = int((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    anchor_args['D'] = int((cav_lidar_range[5] - cav_lidar_range[2]) / vd)
    param['postprocess'].update({'anchor_args': anchor_args})

    if 'model' in param:
        param['model']['args']['W'] = anchor_args['W']
        param['model']['args']['H'] = anchor_args['H']
        param['model']['args']['D'] = anchor_args['D']

    return param


def load_point_pillar_params(param):
    """
    Fill derived geometry values for PointPillar-style configs.
    """
    cav_lidar_range = param['preprocess']['cav_lidar_range']
    voxel_size = param['preprocess']['args']['voxel_size']

    grid_size = (np.array(cav_lidar_range[3:6]) -
                 np.array(cav_lidar_range[0:3])) / np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    param['model']['args']['point_pillar_scatter']['grid_size'] = grid_size

    anchor_args = param['postprocess']['anchor_args']
    vw, vh, vd = voxel_size
    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd
    anchor_args['W'] = math.ceil((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args['H'] = math.ceil((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    anchor_args['D'] = math.ceil((cav_lidar_range[5] - cav_lidar_range[2]) / vd)
    param['postprocess'].update({'anchor_args': anchor_args})

    return param


def load_second_params(param):
    """
    Fill derived geometry values for SECOND-style configs.
    """
    cav_lidar_range = param['preprocess']['cav_lidar_range']
    voxel_size = param['preprocess']['args']['voxel_size']

    grid_size = (np.array(cav_lidar_range[3:6]) -
                 np.array(cav_lidar_range[0:3])) / np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    param['model']['args']['grid_size'] = grid_size

    anchor_args = param['postprocess']['anchor_args']
    vw, vh, vd = voxel_size
    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd
    anchor_args['W'] = int((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args['H'] = int((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    anchor_args['D'] = int((cav_lidar_range[5] - cav_lidar_range[2]) / vd)
    param['postprocess'].update({'anchor_args': anchor_args})

    return param


def load_bev_params(param):
    """
    Fill BEV geometry parameters used by PIXOR/BEV-style configs.
    """
    res = param['preprocess']['args']['res']
    l1, w1, h1, l2, w2, h2 = param['preprocess']['cav_lidar_range']
    downsample_rate = param['preprocess']['args']['downsample_rate']

    def span(low, high, resolution):
        return int((high - low) / resolution)

    input_shape = (
        int(span(l1, l2, res)),
        int(span(w1, w2, res)),
        int(span(h1, h2, res) + 1),
    )
    label_shape = (
        int(input_shape[0] / downsample_rate),
        int(input_shape[1] / downsample_rate),
        7,
    )
    geometry_param = {
        'L1': l1,
        'L2': l2,
        'W1': w1,
        'W2': w2,
        'H1': h1,
        'H2': h2,
        'downsample_rate': downsample_rate,
        'input_shape': input_shape,
        'label_shape': label_shape,
        'res': res,
    }

    param['preprocess']['geometry_param'] = geometry_param
    param['postprocess']['geometry_param'] = geometry_param
    param['model']['args']['geometry_param'] = geometry_param

    return param


def save_yaml(data, save_name):
    """
    Save a dictionary to yaml.
    """
    with open(save_name, 'w', encoding='utf-8') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def save_yaml_wo_overwriting(data, save_name):
    """
    Save yaml without overwriting existing keys in an existing file.
    """
    if os.path.exists(save_name):
        prev_data = load_yaml(save_name)
        data = {**data, **prev_data}
    save_yaml(data, save_name)


_PARSER_REGISTRY = {
    'load_voxel_params': load_voxel_params,
    'load_point_pillar_params': load_point_pillar_params,
    'load_second_params': load_second_params,
    'load_bev_params': load_bev_params,
}
