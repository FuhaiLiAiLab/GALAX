import sys
import os
import torch
import random
import numpy as np
from texttable import Texttable
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask

def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    
def tab_printer(args, float_precision=5):
    """Function to print the logs in a nice tabular format with configurable precision.

    Note
    ----
    Package `Texttable` is required.
    Run `pip install Texttable` if was not installed.

    Parameters
    ----------
    args: Parameters used for the model.
    float_precision: Number of decimal places to display for float values (default: 5)
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.set_precision(float_precision)  # Set the precision for floating point numbers
    
    # Format the rows with proper string formatting for different types
    formatted_rows = []
    for k in keys:
        if not k.startswith('__'):
            value = args[k]
            if isinstance(value, float):
                # Format float with specified precision
                formatted_value = f"{value:.{float_precision}f}"
            else:
                formatted_value = str(value)
            formatted_rows.append([k, formatted_value])
    
    # Add header and formatted rows
    t.add_rows([["Parameter", "Value"]] + formatted_rows)
    return t.draw()
