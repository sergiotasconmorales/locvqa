# Project:
#   Localized Questions in VQA
# Description:
#   Dataset creation for localized questions, depending on chosen config file
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

import os
import json

from misc import io
from dataset_factory import regions

# get args and config
args = io.get_config_file_name()
config = io.read_config(args.path_config)

# invoke constructor depending on class name given in the config file
dataset_obj = getattr(regions, config['dataset'])(config)

dataset_obj.print_details()
dataset_obj.build()
dataset_obj.created_successfully()