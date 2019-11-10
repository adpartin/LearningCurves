"""
This script contains functions that help with meta-analysis of the different runs.
"""

import os
import sys
from pathlib import Path
from time import time
from collections import OrderedDict

import sklearn
import numpy as np
import pandas as pd
