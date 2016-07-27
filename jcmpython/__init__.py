
__doc__ = """
This module defines classes and functions to extend the python interface
of JCMwave.

Copyright(C) 2016 Carlo Barth.
*** This software project is controlled using git *** 
"""

# Start up by parsing the configuration file and importing jcmwave
from jcmpython.internals import _config, jcm, daemon

# Configure the logging
import log
import logging
logger = logging.getLogger('init')

from parallelization import read_resources_from_config, DaemonResource
# initialize the daemon resources and load them into the namespace
logger.debug('Initializing resources from configuration.')
resources = read_resources_from_config()

from core import JCMProject, Simulation, Results, SimulationSet
from materials import RefractiveIndexInfo
# from JCMpython.Accessory import * 
# from JCMpython.BandstructureTools import * 
# from JCMpython.startup import * 
# from JCMpython.DaemonResources import Workstation, Queue
# from JCMpython.MaterialData import MaterialData, RefractiveIndexInfo
# from JCMpython.Results import Results
# from JCMpython.Simulation import Simulation

# Clear unnecessary variables from the namespace
del logger, logging


