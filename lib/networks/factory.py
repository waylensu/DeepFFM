# --------------------------------------------------------
# SubCNN_TF
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import networks.deep_ffm
import pdb
import tensorflow as tf


def get_network(name = None):
    """Get a network by name."""
    return networks.deep_ffm.Deepffm()
    
def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
