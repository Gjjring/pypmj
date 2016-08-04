"""The core functionality of jcmpython. Defines the classes JCMProject,
Simulation and SimulationSet.

Authors : Carlo Barth
"""

# Imports
# =============================================================================
import logging
from jcmpython.internals import jcm, daemon, _config, ConfigurationError
from jcmpython import resources, __version__, __jcm_version__
from copy import deepcopy
from datetime import date
from glob import glob
from itertools import product
import numpy as np
from shutil import copytree, rmtree
import os
import pandas as pd
import sys
import time
import utils

# Get a logger instance
logger = logging.getLogger(__name__)

# Load values from configuration
PROJECT_BASE = _config.get('Data', 'projects')
DBASE_NAME = _config.get('DEFAULTS', 'database_name')
DBASE_TAB = _config.get('DEFAULTS', 'database_tab_name')

# Global defaults
SIM_DIR_FMT = 'simulation{0:06d}'

def _default_sim_wdir(storage_dir, sim_number):
    """Returns the default working directory path for a given storage folder
    and simulation number."""
    return os.path.join(storage_dir, SIM_DIR_FMT.format(sim_number))


# =============================================================================
class JCMProject(object):
    """Class that finds a JCMsuite project using a path specifier (relative to
    the `projects` path specified in the configuration), checks its validity
    and provides functions to copy its content to a working directory, remove
    it afterwards etc.
    
    Parameters
    ----------
    specifier : str or list
        Can be
          * a path relative to the `projects` path specified in the
            configuration, given as complete str to append or sequence of
            strings which are .joined by os.path.join(),
          * or an absolute path to the project directory.
    working_dir : str or None, default None
        The path to which the files in the project directory are copied. If 
        None, a folder called `current_run` is created in the current working
        directory
    project_file_name : str or None, default None
        The name of the project file. If None, automatic detection is tried
        by looking for a .jcmp or .jcmpt file with a line that starts with
        the word `Project`. If this fails, an Exception is raised.
    job_name : str or None, default None
        Name to use for queuing system such as slurm. If None, a name is
        composed using the specifier.
    
    """
    def __init__(self, specifier, working_dir=None, project_file_name=None, 
                 job_name=None):
        self.source = self._find_path(specifier)
        self._check_project()
        self._check_working_dir(working_dir)
        if project_file_name is None:
            self.project_file_name = self._find_project_file()
        else:
            if (not isinstance(project_file_name, (str, unicode)) or
                not os.path.splitext(project_file_name)[1] in ['.jcmp',
                                                               '.jcmpt']):
                raise ValueError('`project_file_name` must be a project '+
                                 'filename or None')
                return
            self.project_file_name = project_file_name
        if job_name is None:
            job_name = 'JCMProject_{}'.format(os.path.basename(self.source))
        self.job_name = job_name
        self.was_copied = False
        
    def _find_path(self, specifier):
        """Finds a JCMsuite project using a path specifier relative to
        the `projects` path specified in the configuration or an absolute path.
        """
        # Check whether the path is absolute
        if isinstance(specifier, (str, unicode)):
            if os.path.isabs(specifier):
                if not os.path.exists(specifier):
                    raise OSError('The absolute path {} does not exist.'.format(
                                                                    specifier))
                else:
                    return specifier
        
        # Treat the relative path
        err_msg = 'Unable to find the project source folder specified' +\
                  ' by {} (using project root: {})'.format(specifier, 
                                                           PROJECT_BASE)
        try:
            if isinstance(specifier, (list,tuple)):
                source_folder = os.path.join(PROJECT_BASE, *specifier)
            else:
                source_folder = os.path.join(PROJECT_BASE, specifier)
        except:
            raise OSError(err_msg)
        if not os.path.isdir(source_folder):
            print source_folder
            raise OSError(err_msg)
        return source_folder
    
    def _find_project_file(self):
        """Tries to find the project file name in the source folder by parsing
        all .jcmp or .jcmpt files."""
        jcmpts = glob(os.path.join(self.source, '*.jcmpt'))
        jcmps = glob(os.path.join(self.source, '*.jcmp'))
        for files in [jcmpts, jcmps]:
            matches = self.__parse_jcmp_t_files(files)
            if len(matches) == 1:
                return matches[0]
            elif len(matches) > 1:
                raise Exception('Multiple valid project files found in '+
                                'source folder: {}'.format(matches) + 
                                'Please specify the project filename manually.')
        # Only arrives here if no valid project file was found
        raise Exception('No valid project file found in source folder. ' + 
                        'Please specify the project filename manually.')
    
    def __parse_jcmp_t_files(self, files):
        """Returns all valid project files in a list of jcmp(t)-files."""
        return [os.path.basename(f) for f in files if self.__is_project_file(f)]
    
    def __is_project_file(self, fname):
        """Checks if a given file contains a line starting with the
        word `Project`."""
        with open(fname, 'r') as f:
            for line in f.readlines():
                if line.strip().startswith('Project'):
                    return True
        return False
    
    def __repr__(self):
        return 'JCMProject({})'.format(self.source)
    
    def _check_project(self):
        """Checks if files of signature *.jcm* are inside the project directory.
        """
        files = glob(os.path.join(self.source, '*.jcm*'))
        if len(files) == 0:
            raise Exception('Unable to find files of signature *.jcm* in the '+
                            'specified project folder {}'.format(self.source))
            
    def _check_working_dir(self, working_dir):
        """Checks if the given working directory exists and creates it if not.
        If no `working_dir` is None, a default directory called `current_run` is
        is created in the current working directory.
        """
        if working_dir is None:
            working_dir = os.path.abspath('current_run')
            logger.debug('JCMProject: No working_dir specified, using {}'.\
                                                            format(working_dir))
        else:
            if not os.path.isdir(working_dir):
                logger.debug('JCMProject: Creating working directory {}'.\
                                                            format(working_dir))
                os.makedirs(working_dir)
        self.working_dir = working_dir
    
    def copy_to(self, path=None, overwrite=True, sys_append=True):
        """Copies all files inside the project directory to path, overwriting it
        if  overwrite=True, raising an Error otherwise if it already exists. 
        Note: Appends the path to sys.path if sys_append=True.
        """
        if path is None:
            path = self.working_dir
        if os.path.exists(path):
            if overwrite:
                logger.debug('Removing existing folder {}'.format(path))
                rmtree(path)
            else:
                raise OSError('Path {} already exists! If you '.format(path)+
                              'wish copy anyway set `overwrite` to True.')
        logger.debug('Copying project to folder: {}'.format(self.working_dir))
        copytree(self.source, path)
        
        # Append this path to the PYTHONPATH. This is necessary to allow python
        # files inside the project directory, e.g. to use them inside a JCM
        # template file
        if sys_append:
            sys.path.append(path)
        self.was_copied = True
    
    def get_project_file_path(self):
        """Returns the complete path to the project file."""
        return os.path.join(self.working_dir, self.project_file_name)
    
    def remove_working_dir(self):
        """Removes the working directory.
        """
        logger.debug('Removing working directory: {}'.format(self.working_dir))
        if os.path.exists(self.working_dir):
            rmtree(self.working_dir)
        self.was_copied = False


# =============================================================================
class Simulation(object):
    """
    Class which describes a distinct simulation and provides a method to run it
    and to remove the working directory afterwards.
    """
    def __init__(self, number, keys, stored_keys, storage_dir, projectFileName,
                 rerun_JCMgeo=False):
        self.number = number
        self.keys = keys
        self.stored_keys = stored_keys
        self.storage_dir = storage_dir
        self.projectFileName = projectFileName
        self.rerun_JCMgeo = rerun_JCMgeo
        self.status = 'Pending'
    
    def __repr__(self):
        return 'Simulation(number={}, status={})'.format(self.number, 
                                                         self.status)
    
    def working_dir(self):
        """Returns the name of the working directory, specified by the 
        storage_dir and the simulation number. It is constructed using the
        global SIM_DIR_FMT formatter."""
        return _default_sim_wdir(self.storage_dir, self.number)
    
    def solve(self, **jcm_kwargs):
        """Starts the simulation (i.e. runs jcm.solve) and returns the job ID.
        
        The jcm_kwargs are directly passed to jcm.solve, except for 
        `project_dir`, `keys` and `working_dir`, which are set automatically
        (ignored if provided).
        """
        forbidden_keys = ['project_file', 'keys', 'working_dir']
        for key in jcm_kwargs:
            if key in forbidden_keys:
                logger.warn('You cannot use {} as a keyword '.format(key)+
                             'argument for jcm.solve. It is already set by the'+
                             ' Simulation instance.')
                del jcm_kwargs[key]
        
        # Make directories if necessary
        wdir = self.working_dir()
        if not os.path.exists(wdir):
            os.makedirs(wdir)
        
        # Start to solve
        self.jobID = jcm.solve(self.projectFileName, keys=self.keys, 
                               working_dir=wdir, **jcm_kwargs)
        return self.jobID
    
    def _set_jcm_results_and_logs(self, results, logs):
        """Set the logs, error message, exit code and results as returned by
        JCMsolve. This also sets the status to `Failed` or `Finished`."""
        self.logs = logs['Log']['Out']
        self.error_message = logs['Log']['Error']
        self.exit_code = logs['ExitCode']
        self.jcm_results = results
        
        # Treat failed simulations
        if self.exit_code != 0:
            self.status = 'Failed'
            return
            
        # If the solve did not fail, the results dict must contain a dict with
        # the key 'computational_costs' in the topmost level. Otherwise, 
        # something must be wrong.
        if len(results) < 1:
            raise RuntimeError('Did not receive results from JCMsolve '+
                               'although the exit status is 0.')
            self.status = 'Failed'
            return
        if not isinstance(results[0], dict):
            raise RuntimeError('Expecting a dict as the first element of the '+
                               'results list, but the type is {}'.format(
                                                            type(results[0])))
            self.status = 'Failed'
            return
        if not 'computational_costs' in results[0] and 'file' in results[0]:
            raise RuntimeError('Could not find info on computational costs in '+
                               'the JCM results.')
            self.status = 'Failed'
            return
        
        # Everything is fine if we arrive here. We also read the fieldbag file
        # path from the results
        self.fieldbag_file = results[0]['file']
        self.status = 'Finished'
    
    def evaluate_results(self, evaluation_func=None, overwrite=False):
        """Evaluate the raw results from JCMsolve with a function 
        `evaluation_func` of one input argument. The input argument, which is
        the list of results as it was set in `_set_jcm_results_and_logs`, is
        automatically passed to this function.
        
        If `evaluation_func` is None, the JCM results are not processed and
        nothing will be saved to the HDF5 store, except for the computational
        costs.
        
        The `evaluation_func` must be a function of a single input argument. 
        A list of all results returned by post processes in JCMsolve are passed
        to this function. It must return a dict with key-value pairs that should
        be saved to the HDF5 store. Consequently, the values must be of types
        that can be stored to HDF5, otherwise Exceptions will occur in the
        saving steps. 
        """
        
        if self.status in ['Pending', 'Failed']:
            logger.warn('Unable to evaluate the results, as the status of '+
                         'the simulation is: {}'.format(self.status))
            return
        elif self.status == 'Finished and evaluated':
            if overwrite:
                self.status = 'Finished'
                del self._results_dict
            else:
                logger.warn('The simulation results are already evaluated!'+
                             ' To overwrite, set `overwrite` to True.')
                return
        
        # Now the status must be 'Finished'
        if not self.status == 'Finished':
            raise RuntimeError('Unknown status: {}'.format(self.status))
            return
        
        # Process the computational costs
        self._results_dict = utils.computational_costs_to_flat_dict(
                                    self.jcm_results[0]['computational_costs'])
#         self._results_dict['fieldbag_file'] = self.fieldbag_file
        self.status = 'Finished and evaluated'
        
        # Stop here if evaluation_func is None
        if evaluation_func is None:
            logger.debug('No result evaluation was done.')
            return
        
        # Also stop, if there are no results from post processes
        if len(self.jcm_results) <= 1:
            logger.info('No further evaluation will be performed, as there '+
                         'are no results from post processes in the JCM result'+
                         ' list.')
            return
        
        # Otherwise, evaluation_func must be a callable
        if not callable(evaluation_func):
            logger.warn('`evaluation_func` must be callable of one input '+
                         'Please consult the docs of `evaluate_results`.')
            return
        
        # We try to evaluate the evaluation_func now. If it fails or its results
        # are not of type dict, it is ignored and the user will be warned
        try:
            eres = evaluation_func(self.jcm_results[1:]) # anything might happen
        except Exception as e:
            logger.warn('Call of `evaluation_func` failed: {}'.format(e))
            return
        if not isinstance(eres, dict):
            logger.warn('The return value of `evaluation_func` must be of '+
                         'type dict, not {}'.format(type(eres)))
            return
        
        # Warn the user if she/he used a key that is already present due to the
        # stored computational costs
        for key in eres.keys():
            if key in self._results_dict:
                logger.warn('The key {} is already present due to'.format(key)+
                             ' the automatic storage of computational costs. '+
                             'It will be overwritten!')
        
        # Finally, we update the results that will be stored to the 
        # _results_dict
        self._results_dict.update(eres)
    
    def _get_DataFrame(self):
        """Returns a DataFrame containing all input parameters and all results
        with the simulation number as the index. It can readily be appended to
        the HDF5 store."""
        dfdict = {skey:self.keys[skey] for skey in self.stored_keys}
        if self.status == 'Finished and evaluated':
            dfdict.update(self._results_dict)
        else:
            logger.warn('You are trying to get a DataFrame for a non-'+
                         'evaluated simulation. Returning only the keys.')
        df = pd.DataFrame(dfdict, index=[self.number])
        df.index.name = 'number'
        return df
    
    def _get_parameter_DataFrame(self):
        """Returns a DataFrame containing only the input parameters with the
        simulation number as the index. This is mainly used for HDF5 store
        comparison."""
        dfdict = {skey:self.keys[skey] for skey in self.stored_keys}
        df = pd.DataFrame(dfdict, index=[self.number])
        df.index.name = 'number'
        return df
    
    def remove_working_directory(self):
        """Removes the working directory."""
        wdir = self.working_dir()
        if os.path.exists(wdir):
            try:
                rmtree(wdir)
            except:
                logger.warn('Failed to remove working directory {}'.format(
                             os.path.basename(wdir)) +\
                             ' for simNumber {}'.format(self.number))
        else:
            logger.warn('Working directory {} does not exist'.format(
                         os.path.basename(wdir)) +\
                         ' for simNumber {}'.format(self.number))


# =============================================================================
class SimulationSet(object):
    """Class for initializing, planning, running and evaluating multiple 
    simulations.
    
    Parameters
    ----------
    project : JCMProject, str or tuple/list of the form (specifier, working_dir)
        JCMProject to use for the simulations. If no JCMProject-instance is 
        provided, it is created using the given specifier or, if project is of 
        type tuple, using (specifier, working_dir) (i.e. JCMProject(project[0], 
        project[1])).
    keys : dict
        There are two possible use cases:
          1. The keys are the normal keys as defined by JCMsuite, containing
             all the values that need to passed to parse the JCM-template files.
             In this case, a single computation is performed using these keys.
          2. The keys-dict contains at least one of the keys [`constants`,
             `geometry`, `parameters`] and no additional keys. The values of
             each of these keys must be of type dict again and contain the keys
             necessary to parse the JCM-template files. Depending on the 
             `combination_mode`, loops are performed over any 
             parameter-sequences provided in `geometry` or `parameters`. JCMgeo
             is only called if the keys in `geometry` change between consecutive
             runs. Keys in `constants` are not stored in the HDF5 store! 
             Consequently, this information is lost, but also adds the 
             flexibility to path arbitrary data types to JCMsuite that could not
             be stored in the HDF5 format.
    duplicate_path_levels : int, default 0
        For clearly arranged data storage, the folder structure of the current
        working directory can be replicated up to the level given here. I.e., if
        the current dir is /path/to/your/jcmpython/ and duplicate_path_levels=2,
        the subfolders your/jcmpython will be created in the storage base dir
        (which is controlled using the configuration file). This is not done if
        duplicate_path_levels=0.
    storage_folder : str, default 'from_date'
        Name of the subfolder inside the storage folder in which the final data
        is stored. If 'from_date' (default), the current date (%y%m%d) is used.
    storage_base : str, default 'from_config'
        Directory to use as the base storage folder. If 'from_config', the
        folder set by the configuration option Storage->base is used. 
    combination_mode : {'product', 'list'}
        Controls the way in which sequences in the `geometry` or `parameters`
        keys are treated.
          * If `product`, all possible combinations of the provided keys are
            used.
          * If `list`, all provided sequences need to be of the same length N, 
            so that N simulations are performed, using the value of the i-th 
            element of each sequence in simulation i.
    check_version_match : bool, default True
        Controls if the versions of JCMsuite and jcmpython are compared to the
        versions that were used when the HDF5 store was used. This has no effect
        if no HDF5 is present, i.e. if you are starting with an empty working
        directory.
    """
    
    # Names of the groups in the HDF5 store which are used to store metadata
    STORE_META_GROUPS = ['parameters', 'geometry']
    STORE_VERSION_GROUP = 'version_data'
    
    def __init__(self, project, keys, duplicate_path_levels=0, 
                 storage_folder='from_date', storage_base='from_config',
                 combination_mode='product', check_version_match=True):
                
        # Save initialization arguments into namespace
        self.combination_mode = combination_mode
        
        # Analyze the provided keys
        self._check_keys(keys)
        self.keys = keys
        
        # Load the project and set up the folders
        self._load_project(project)
        self._set_up_folders(duplicate_path_levels, storage_folder, 
                             storage_base)
        
        # Initialize the HDF5 store
        self._initialize_store(check_version_match)
    
    def __repr__(self):
        return 'SimulationSet(project={}, storage={})'.format(self.project,
                                                              self.storage_dir)
    
    def _check_keys(self, keys):
        """Checks if the provided keys are valid and if they contain values for
        loops.
        
        See the description of the parameter `keys` in the SimulationSet 
        documentation for further reference.
        """
        
        # Check proper type
        if not isinstance(keys, dict):
            raise ValueError('`keys` must be of type dict.')
        
        loop_indication = ['constants', 'geometry', 'parameters']
        
        # If none of the `loop_indication` keys is in the dict, case 1 is 
        # assumed
        keys_rest = [_k for _k in keys.keys() if not _k in loop_indication]
        if len(keys_rest) > 0:
            self.constants = keys
            self.geometry = []
            self.parameters = []
            return
        
        # Otherwise, case 2 is assumed
        if set(loop_indication).isdisjoint(set(keys.keys())):
            raise ValueError('`keys` must contain at least one of the keys .'+
                             ' {} or all the keys '.format(loop_indication) +
                             'necessary to compile the JCM-template files.')
        for _k in loop_indication:
            if _k in keys.keys():
                if not isinstance(keys[_k], dict):
                    raise ValueError('The values for the keys {}'.format(
                                     loop_indication) + ' must be of type '+
                                     '`dict`')
                setattr(self, _k, keys[_k])
            else:
                setattr(self, _k, {})
        
    def get_all_keys(self):
        """Returns a list of all keys that are passed to JCMsolve."""
        return self.parameters.keys() + \
               self.geometry.keys() + \
               self.constants.keys()
    
    def _load_project(self, project):
        """Loads the specified project as a JCMProject-instance."""
        if isinstance(project, (str,unicode)):
            self.project = JCMProject(project)
        elif isinstance(project, (tuple, list)):
            if not len(project) == 2:
                raise ValueError('`project` must be of length 2 if it is a '+
                                 'sequence')
            self.project = JCMProject(*project)
        else:
            # TODO: this is an ugly hack to detect whether project is of type
            # JCMProject. Somehow the normal isinstance(project, JCMproject)
            # failed in the jupyter notebook sometimes. 
            if hasattr(project, 'project_file_name'):
                self.project = project
            else:
                raise ValueError('`project` must be int, tuple or JCMproject.')
        if not self.project.was_copied:
            self.project.copy_to()
    
    def get_project_wdir(self):
        """Returns the path to the working directory of the current project."""
        return self.project.working_dir

    def _set_up_folders(self, duplicate_path_levels, storage_folder,
                        storage_base):
        """Reads storage specific parameters from the configuration and prepares
        the folder used for storage as desired.
        
        See the description of the parameters `` and `` in the SimulationSet 
        documentation for further reference.
        """
        # Read storage base from configuration
        if storage_base == 'from_config':
            base = _config.get('Storage', 'base')
            if base == 'CWD':
                base = os.getcwd()
        else:
            base = storage_base
        if not os.path.isdir(base):
            raise OSError('The storage base folder {} does not exist.'.format(
                                                                        base))
            return
        
        if duplicate_path_levels > 0:
            # get a list folders that build the current path and use the number
            # of subdirectories as specified by duplicate_path_levels
            cfolders = os.path.normpath(os.getcwd()).split(os.sep)
            base = os.path.join(base, *cfolders[-duplicate_path_levels:])
        
        if storage_folder == 'from_date':
            # Generate a directory name from date
            storage_folder = date.today().strftime("%y%m%d")
        self.storage_dir = os.path.join(base, storage_folder)
        
        # Create the necessary directories
        if not os.path.exists(self.storage_dir):
            logger.debug('Creating non-existent storage folder {}'.format(
                                                            self.storage_dir))
            os.makedirs(self.storage_dir)
        
        logger.info('Using folder {} for '.format(self.storage_dir)+ 
                     'data storage.')
    
    def _initialize_store(self, check_version_match):
        """Initializes the HDF5 store and sets the `store` attribute. The
        file name and the name of the data section inside the file are 
        configured in the DEFAULTS section of the configuration file. 
        """
        logger.debug('Initializing the HDF5 store')
        
        self._database_file = os.path.join(self.storage_dir, DBASE_NAME)
        if not os.path.splitext(DBASE_NAME)[1] == '.h5':
            logger.warn('The HDF5 store file has an unknown extension. '+
                         'It should be `.h5`.')
        self.store = pd.HDFStore(self._database_file)
        
        # Version comparison
        if not self.is_store_empty() and check_version_match:
            logger.debug('Checking version match.')
            self._check_store_version_match()
    
    def _check_store_version_match(self):
        """Compares the currently used versions of jcmpython and JCMsuite to
        the versions that were used when the store was created."""
        version_df = self.store[self.STORE_VERSION_GROUP]
        
        # Load stored versions
        stored_jcm_version = version_df.at[0,'__jcm_version__']
        stored_jpy_version = version_df.at[0,'__version__']
        
        # Check match and handle mismatches
        if not stored_jcm_version == __jcm_version__:
            raise ConfigurationError(
                'Version mismatch! HDF5 store was created using JCMsuite '+
                'version {}, but the current '.format(stored_jcm_version)+
                'version is {}. Change the version or'.format(__jcm_version__)+
                ' set `check_version_match` to False on the SimulationSet '+
                'initialization.')
            return
        if not stored_jpy_version == __version__:
            logger.warn('Version mismatch! HDF5 store was created using '+
                'jcmpython version {}, the current '.format(stored_jcm_version)+
                'version is {}.'.format(__jcm_version__))
    
    def is_store_empty(self):
        """Checks if the HDF5 store is empty.""" 
        if not DBASE_TAB in self.store:
            return True
        
        # Check store validity
        for group in self.STORE_META_GROUPS + [self.STORE_VERSION_GROUP]:
            if not group in self.store:
                raise Exception('The HDF5 store seems to be corrupted! A data' +
                                ' section was found, but the metadata group '+
                                '`{}` is missing.'.format(group))
        return False
    
    def get_store_data(self):
        """Returns the data currently in the store"""
        if self.is_store_empty():
            return None
        return self.store[DBASE_TAB]
    
    def write_store_data_to_file(self, file_path=None, mode='CSV', **kwargs):
        """Writes the data that is currently in the store to a CSV or an Excel
        file. `mode` must be either 'CSV' or 'Excel'. If `file_path` is None, 
        the default name results.csv/xls in the storage folder is used. 
        `kwargs` are passed to the corresponding pandas functions."""
        if not mode in ['CSV', 'Excel']:
            raise ValueError('Unknown mode: {}. Use CSV or Excel.'.format(mode))
        if mode=='CSV':
            if file_path is None:
                file_path = os.path.join(self.storage_dir, 'results.csv')
            self.get_store_data().to_csv(file_path, **kwargs)
        else:
            if file_path is None:
                file_path = os.path.join(self.storage_dir, 'results.xls')
            writer = pd.ExcelWriter(file_path)
            self.get_store_data().to_excel(writer, 'data', **kwargs)
            writer.save()
        
    def close_store(self):
        """Closes the HDF5 store."""
        logger.debug('Closing the HDF5 store: {}'.format(self._database_file))
        self.store.close()
    
    def append_store(self, data):
        """Appends a new row or multiple rows to the HDF5 store."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError('Can only append pandas DataFrames to the store.')
            return
        self.store.append(DBASE_TAB, data) 
    
    def make_simulation_schedule(self):
        """Makes a schedule by getting a list of simulations that must be
        performed, reorders them to avoid unnecessary calls of JCMgeo, and
        checks the HDF5 store for simulation data which is already known."""
        self._get_simulation_list()
        self._sort_simulations()
        
        # We perform the pre-check to see the state of our HDF5 store.
        #   * If it is empty, we store the current metadata and are ready to 
        #     start the simulation
        #   * If the metadata perfectly matches the current simulation 
        #     parameters, we can assume that the indices in the store correspond
        #     to the current simulation numbers and perform only the missing
        #     ones
        #   * If the status is 'Extended Check', we will need to compare the
        #     stored data to the one we want to compute currently
        precheck = self._precheck_store()
        logger.debug('Result of the store pre-check: {}'.format(precheck))
        if precheck == 'Empty':
            self.finished_sim_numbers = []
        if precheck == 'Extended Check':
            self._extended_store_check()
            logger.info('Found matches in the extended check of the HDF5 '+
                         'store. Number of stored simulations: {}'.format(
                                                len(self.finished_sim_numbers)))
        elif precheck == 'Match':
            self.finished_sim_numbers = list(self.get_store_data().index)
            logger.info('Found a match in the pre-check of the HDF5 store. '+
                         'Number of stored simulations: {}'.format(
                                                len(self.finished_sim_numbers)))
     
    def _get_simulation_list(self):
        """Check the `parameters`- and `geometry`-dictionaries for sequences and 
        generate a list which has a keys-dictionary for each distinct
        simulation by using the `combination_mode` as specified. The simulations
        that must be performed are stored in the `self.simulations`-list.
        """
        logger.debug('Analyzing loop properties.')
        self.simulations = []
         
        # Convert lists in the parameters- and geometry-dictionaries to numpy
        # arrays and find the properties over which a loop should be performed 
        # and the
        self._loop_props = []
        loopList = []
        fixedProperties = []
        for p in self.parameters.keys():
            pSet = self.parameters[p]
            if isinstance(pSet, list):
                pSet = np.array(pSet)
                self.parameters[p] = pSet
            if isinstance(pSet, np.ndarray):
                self._loop_props.append(p)
                loopList.append([(p, item) for item in pSet])
            else:
                fixedProperties.append(p)
        for g in self.geometry.keys():
            gSet = self.geometry[g]
            if isinstance(gSet, list):
                gSet = np.array(gSet)
                self.geometry[g] = gSet
            if isinstance(gSet, np.ndarray):
                self._loop_props.append(g)
                loopList.append([(g, item) for item in gSet])
            else:
                fixedProperties.append(g)
        for c in self.constants.keys():
            fixedProperties.append(c)
         
        # Now that the keys are separated into fixed and varying properties,
        # the three dictionaries can be combined for easier lookup
        allKeys = dict( self.parameters.items() + self.geometry.items() + 
                        self.constants.items() )
        
        # For saving the results it needs to be known which properties should
        # be recorded. As a default, all parameters and all geometry-info is
        # used.
        self.stored_keys = self.parameters.keys() + self.geometry.keys()
         
        # Depending on the combination mode, a list of all key-combinations is
        # generated, so that all simulations can be executed in a single loop.
        if self.combination_mode == 'product':
            # itertools.product is used to find all combinations of parameters
            # for which a distinct simulation needs to be done
            propertyCombinations = list( product(*loopList) )
        elif self.combination_mode == 'list':
            # In `list`-mode, all sequences need to be of the same length,
            # assuming that a loop has to be done over their indices 
            Nsims = len(loopList[0])
            for l in loopList:
                if not len(l) == Nsims:
                    raise ValueError('In list-mode all parameter-lists need '+
                                     'to have the same length')
            
            propertyCombinations = []
            for iSim in range(Nsims):
                propertyCombinations.append(tuple([l[iSim] for l in loopList]))

        self.num_sims = len(propertyCombinations) # total num of simulations
        if self.num_sims == 1:
            logger.info('Performing a single simulation')
        else:
            logger.info('Loops will be done over the following parameter(s):'+
                         ' {}'.format(self._loop_props))
            logger.info('Total number of simulations: {}'.format(
                                                                self.num_sims))
         
        # Finally, a list with an individual Simulation-instance for each
        # simulation is saved, over which a simple loop can be performed
        logger.debug('Generating the simulation list.')
        pfile_path = self.project.get_project_file_path()
        for i, keySet in enumerate(propertyCombinations):
            keys = {}
            for k in keySet:
                keys[ k[0] ] = k[1]
            for p in fixedProperties:
                keys[p] = allKeys[p]
            self.simulations.append(Simulation(number = i, keys = keys,
                                               stored_keys = self.stored_keys,
                                               storage_dir = self.storage_dir,
                                               projectFileName=pfile_path))

    def _sort_simulations(self):
        """Sorts the list of simulations in a way that all simulations with 
        identical geometry are performed consecutively. That way, jcmwave.geo()
        only needs to be called if the geometry changes.
        """
        logger.debug('Sorting the simulations.')
        # Get a list of dictionaries, where each dictionary contains the keys 
        # and values which correspond to geometry information of a single 
        # simulation
        allGeoKeys = []
        geometryTypes = np.zeros((self.num_sims), dtype=int)
        for s in self.simulations:
            allGeoKeys.append({k: s.keys[k] for k in self.geometry.keys()})
         
        # Find the number of different geometries and a list where each entry
        # corresponds to the geometry-type of the simulation. The types are
        # simply numbered, so that the first simulation is of type 1, as well
        # as all simulations with the same geometry and so on...
        pos = 0
        nextPos = 0
        t = 1
        while 0 in geometryTypes:
            geometryTypes[pos] = t
            foundDiscrepancy = False
            for i in range(pos+1, self.num_sims):
                if allGeoKeys[pos] == allGeoKeys[i]:
                    if geometryTypes[i] == 0:
                        geometryTypes[i] = t
                else:
                    if not foundDiscrepancy:
                        nextPos = i
                        foundDiscrepancy = True
            pos = nextPos
            t += 1
            
        # From this list of types, a new sort order is derived, in which 
        # simulations with the same geometry are consecutive.
        NdifferentGeometries = t-1
        rerunJCMgeo = np.zeros((NdifferentGeometries), dtype=int)

        sortedGeometryTypes = np.sort(geometryTypes)
        sortIndices = np.argsort(geometryTypes)
        for i in range(NdifferentGeometries):
            rerunJCMgeo[i] = np.where(sortedGeometryTypes == (i+1))[0][0]
        
        # The list of simulations is now reordered and the simulation numbers
        # are reindexed and the rerun_JCMgeo-property is set to True for each
        # simulation in the list that starts a new series of constant geometry.
        self.simulations = [self.simulations[i] for i in sortIndices]
        for i in range(self.num_sims):
            self.simulations[i].number = i
            if i in rerunJCMgeo:
                self.simulations[i].rerun_JCMgeo = True
    
    def __get_version_dframe(self):
        """Returns a pandas DataFrame from the version info of JCMsuite and 
        jcmpython which can be stored in the HDF5 store."""
        return pd.DataFrame({'__version__':__version__, 
                             '__jcm_version__':__jcm_version__}, index=[0])
    
    def _store_version_data(self):
        """Stores metadata of the JCMsuite and jcmpython versions."""
        self.store[self.STORE_VERSION_GROUP] = self.__get_version_dframe()
    
    def __get_meta_dframe(self, which):
        """Creates a pandas DataFrame from the parameters or the geometry-dict
        which can be stored in the HDF5 store. Using the 
        __restore_from_meta_dframe-method, the dict can later be restored."""
        
        # Check if which is valid
        if not which in self.STORE_META_GROUPS:
            raise ValueError('The attribute {} is not supported'.format(which)+
                             ' by _get_meta_dframe(). Valid values are: {}.'.\
                                                format(self.STORE_META_GROUPS))
            return
        d_ = getattr(self, which)
        cols = d_.keys()
        n_rows = utils.get_len_of_parameter_dict(d_)
        df = pd.DataFrame(index=range(n_rows), columns=cols)
        for c in cols:
            val = d_[c]
            if utils.is_sequence(val):
                df.loc[:len(val)-1, c] = val
            else:
                df.loc[0, c] = val
        return df
    
    def __restore_from_meta_dframe(self, which):
        """Restores a dict from data which was stored in the HDF5 store using
        `__get_meta_dframe` to compare the keys that were used for the 
        SimulationSet in which the store was created to the current one.
        `which` can be 'parameters' or 'geometry'.
        """ 
        # Check if which is valid
        if not which in self.STORE_META_GROUPS:
            raise ValueError('The attribute {} is not supported'.format(which)+
                             ' by __restore_from_meta_dframe(). Valid values '+
                             'are: {}.'.format(self.STORE_META_GROUPS))
            return
        if not which in self.store:
            raise Exception('Could not find data for {} in '.format(which)+
                            'the HDF5 store.')
            return
        
        # Load the data
        df = self.store[which]
        dict_ = {}
        for col, series in df.iteritems():
            vals = series.dropna()
            if len(vals) == 1:
                dict_[col] = vals.iat[0]
            else:
                dict_[col] = pd.to_numeric(vals, errors='ignore').values
        return dict_
    
    def _store_metadata(self):
        """Stores metadata of the current simulation set in the HDF5 store.
        
        A SimulationSet is described by its `parameters` and `geometry`
        attributes. These are stored to the HDF5 store for comparison of the
        SimulationSet properties in a future run. 
        
        The `constants` attribute is not stored in the metadata, as these keys
        are also not stored in the data store.
        """
        for group in self.STORE_META_GROUPS:
            self.store[group] = self.__get_meta_dframe(group)
        self._store_version_data()
    
    def _precheck_store(self):
        """Compares the metadata of the current SimulationSet to the metadata
        in the HDF5 store. Returns 'Empty', 'Match', 'Extended Check' or
        'Mismatch'.
        """
        if self.is_store_empty():
            return 'Empty'
        
        # Load metadata from the store
        groups = self.STORE_META_GROUPS
        meta = {g:self.__restore_from_meta_dframe(g) for g in groups}
        
        # Check if the current keys match the keys in the store
        klist = [v.keys() for v in meta.values()]
        # all keys in store:
        meta_keys = [item for sublist in klist for item in sublist]
        if not set(self.stored_keys) == set(meta_keys):
            raise Exception('The simulation keys have changed compared'+
                             ' to the results in the store. Valid keys'+
                             ' are: {}.'.format(meta_keys))
            return 'Mismatch'
        
        # Check if all stored keys are identical to the current ones
        for g in groups:
            current = getattr(self,g)
            stored = meta[g]
            for key in current:
                valc = current[key]
                vals = stored[key]
                if utils.is_sequence(valc):
                    if not utils.is_sequence(vals):
                        return 'Extended Check'
                    elif not len(valc) == len(vals):
                        return 'Extended Check'
                    elif not np.all(valc == vals):
                        return 'Extended Check'
                else:
                    if utils.is_sequence(vals):
                        return 'Extended Check'
                    elif valc != vals:
                        return 'Extended Check'
        return 'Match'
    
    def _extended_store_check(self):
        """Runs the extended comparison of current simulations to execute to
        the results in the HDF5 store.
        """
        search = pd.concat([sim._get_parameter_DataFrame() \
                                            for sim in self.simulations])
        matches, unmatched = self._compare_to_store(search)
        
        # Treat the different cases        
        # If unmatched rows have been found, raise an Error
        if len(unmatched) > 0:
            self.close_store()
            raise NotImplementedError('Found data rows in the store that do'+
                    ' not match simulations that are currently planned. '+
                    'Treating this case will be implemented in a future '+
                    'version of jcmpython. The HDF5 store is now closed.')
        
        # If indices match exactly, set the finished_sim_numbers list
        if all([t[0]==t[1] for t in matches]):
            self.finished_sim_numbers = [t[0] for t in matches]
            return
        
        # Otherwise, we need to reindex the store
        data = self.get_store_data().copy(deep=True)
        look_up_dict = {t[1]:t[0] for t in matches}
        old_index = list(data.index)
        new_index = [look_up_dict[oi] for oi in old_index]
        data.index = pd.Index(new_index)
        
        # Replace the data in the store with the new reindexed data
        self.store.remove(DBASE_TAB)
        self.append_store(data)
        self.store.flush()
        
        # If there are any working directories from the previous run with
        # non-matching simulation numbers, these directories must be renamed.
        dir_rename_dict = {}
        for idx in old_index:
            dwdir = _default_sim_wdir(self.storage_dir, idx)
            dir_rename_dict[dwdir] = _default_sim_wdir(self.storage_dir, 
                                                       look_up_dict[idx])
        logger.debug('Renaming directories.')
        utils.rename_directories(dir_rename_dict)
        self._wdirs_to_clean = dir_rename_dict.values()
        
        # Set the finished_sim_numbers list 
        self.finished_sim_numbers = list(self.get_store_data().index)
    
    def _compare_to_store(self, search):
        """Looks for simulations that are already inside the HDF5 store by
        comparing the values of the columns given by all keys of the current 
        simulations to the values of rows in the store.
        
        Returns a tuple of two lists: (matched_rows, unmatched_rows). Each can
        be None. `matched_rows` is a list of tuples of the form 
        (search_row, store_row) identifying rows in the search DataFrame with
        rows in the stored DataFrame. 'unmatched_rows' is a list of row indices
        in the store that don't have a match in the search DataFrame.
        """
        ckeys = self.get_all_keys()
        if len(ckeys) > 255:
            raise ValueError('Cannot treat more parameters than 255 in the '+
                             'current implementation.')
            return
        
        # Load the DataFrame from the store
        data = self.get_store_data()
        if data is None:
            return None, None
        
        # Check if the ckeys are among the columns of the store 
        # DataFrame
        if not all([key_ in data.columns for key_ in ckeys]):
            raise ValueError('The simulation keys have changed compared'+
                             ' to the results in the store. Valid keys'+
                             ' are: {}.'.format(list(data.columns)))
            return
        
        # Reduce the DataFrame size to the columns that need to be compared
        df_ = data.ix[:,ckeys]
        n_in_store = len(df_) # number of rows in the stored data
        if n_in_store == 0:
            return None, None
        
        # Do the comparison
        matches = []
        for srow in search.itertuples():
            # If all rows in store have matched, we're done
            if n_in_store == len(matches):
                return matches, None
            # Compare this row
            idx = utils.walk_df(df_, srow._asdict(), keys=ckeys)
            if isinstance(idx, int):
                matches.append((srow[0], idx))
            elif not idx is None:
                raise RuntimeError('Fatal error in HDF5 store comparison. '+
                                   'Found multiple matching rows.')
        
        # Return the matches plus a ist of unmatched results indices in the 
        # store
        unmatched = [i for i in list(df_.index) if not i in zip(*matches)[1]]
        return matches, unmatched
    
    def get_current_resources(self):
        """Returns a list of the currently configured resources, i.e. the 
        ones that will be added using `add_resources`."""
        if hasattr(self, 'resource_list'):
            return [resources[r] for r in self.resource_list]
        else:
            return [resources[r] for r in resources.get_resource_names()]
    
    def use_only_resources(self, names):
        """Restrict the daemon resources to `names`. Only makes sense if the
        resources have not already been added. 
        
        Names that are unknown are ignored. If no valid name is present, the
        default configuration will remain untouched.
        """
        if isinstance(names, (str, unicode)):
            names = [names]
        valid = []
        for n in names:
            if not n in resources:
                logger.warn('{} is not in the configured resources'.format(n))
            else:
                valid.append(n)
        if len(valid) == 0:
            logger.warn('No valid resources found, using all instead.')
            return
        logger.info('Restricting resources to: {}'.format(valid))
        self.resource_list = valid
    
    def add_resources(self, n_shots=10, wait_seconds=5, ignore_fail=False):
        """Tries to add all resources configured in the configuration using
        the JCMdaemon."""
        if hasattr(self, 'resource_list'):
            for r in self.resource_list:
                resources[r].add_repeatedly(n_shots, wait_seconds, ignore_fail)
        else:
            resources.add_all_repeatedly(n_shots, wait_seconds, ignore_fail)
    
    def _resources_ready(self):
        """Returns whether the resources are already added."""
        return daemon.daemonCheck(warn=False)
 
    def compute_geometry(self, simulation, **jcm_kwargs):
        """Computes the geometry (i.e. runs jcm.geo) for a specific simulation
        of the simulation set.
        
        Parameters
        ----------
        simulation : Simulation or int
            The `Simulation`-instance for which the geometry should be
            computed. If the type is `int`, it is treated as the index of the
            simulation in the simulation list.
        
        The jcm_kwargs are directly passed to jcm.geo, except for `project_dir`,
        `keys` and `working_dir`, which are set automatically (ignored if 
        provided).
        """
        logger.debug('Computing geometry.')
        
        if isinstance(simulation, int):
            simulation = self.simulations[simulation]
        if not simulation in self.simulations:
            raise ValueError('`simulation` must be a Simulation of the current'+
                             ' SimulationSet or a simulation index (int).')
            return
        
        # Check the keyword arguments
        forbidden_keys = ['project_file', 'keys', 'working_dir']
        for key in jcm_kwargs:
            if key in forbidden_keys:
                logger.warn('You cannot use {} as a keyword '.format(key)+
                             'argument for jcm.geo. It is already set by the'+
                             ' SimulationSet instance.')
                del jcm_kwargs[key]
        
        # Run jcm.geo. The cd-fix is necessary because the project_dir/working_dir
        # functionality seems to be broken in the current python interface!
        _thisdir = os.getcwd()
        os.chdir(self.get_project_wdir())
        with utils.Capturing() as output:
            jcm.geo(project_dir=self.project.working_dir,
                    keys=simulation.keys, 
                    working_dir=self.project.working_dir,
                    **jcm_kwargs)
        for line in output:
            logger.debug('[JCMgeo] '+line)
        os.chdir(_thisdir)
    
    def solve_single_simulation(self, simulation, compute_geometry=True, 
                                jcm_geo_kwargs=None, jcm_solve_kwargs=None):
        """Solves a specific simulation and returns the results and logs
        without any further evaluation and without saving of data to the HDF5
        store. Recomputes the geometry before if compute_geometry is True.
        
        Parameters
        ----------
        simulation : Simulation or int
            The `Simulation`-instance for which the geometry should be
            computed. If the type is `int`, it is treated as the index of the
            simulation in the simulation list.
        compute_geometry : bool, default True
            Runs jcm.geo before the simulation if True.
        jcm_geo_kwargs : dict or None, default None
            These keyword arguments are directly passed to jcm.geo, except for 
            `project_dir`, `keys` and `working_dir`, which are set automatically 
            (ignored if provided).
        jcm_solve_kwargs : dict or None, default None
            These keyword arguments are directly passed to jcm.solve, except for 
            `project_dir`, `keys` and `working_dir`, which are set automatically 
            (ignored if provided).
        """
        if jcm_geo_kwargs is None:
            jcm_geo_kwargs = {}
        if jcm_solve_kwargs is None:
            jcm_solve_kwargs = {}
        
        if isinstance(simulation, int):
            simulation = self.simulations[simulation]
        if not simulation in self.simulations:
            raise ValueError('`simulation` must be a Simulation of the current'+
                             ' SimulationSet or a simulation index (int).')
            return
        
        # Geometry computation
        if compute_geometry:
            self.compute_geometry(simulation, **jcm_geo_kwargs)
        
        # Add the resources if they are not ready yet
        if not self._resources_ready():
            self.add_resources()
        
        # Solve the simulation and wait for it to finish. Output is captured
        # and passed to the logger
        with utils.Capturing() as output:
            simulation.solve(**jcm_solve_kwargs)
            results, logs = daemon.wait()
        for line in output:
            logger.debug(line)
        
        # Set the results and logs in the Simulation-instance and return them
        simulation._set_jcm_results_and_logs(results[0], logs[0])
        return results[0], logs[0]
    
    def _start_simulations(self, N='all', evaluation_func=None,
                           jcm_geo_kwargs={}, jcm_solve_kwargs={}):
        """
        Starts all simulations, `N` at a time, waits for them to finish using
        `_wait_for_simulations` and evaluates the results using the
        `evaluation_func`. `jcm_geo_kwargs` and `jcm_solve_kwargs` are dicts
        of keyword arguments which are directly passed to jcm.geo and jcm.solve,
        respectively. Please consult the docs of the 
        Simulation.evaluate_results-method for info on how to use the
        evaluation_func.
        """
        logger.info('Starting to solve.')
        
        jobIDs = []
        ID2simNumber = {} # dict to find the simulation number from the job ID
        self.failed_simulations = []
        self.evaluation_func = evaluation_func
        
        if N == 'all':
            N = self.num_sims
        if not isinstance(N, int):
            raise ValueError('`N` must be an integer or "all"')
            
        for sim in self.simulations:
            i = sim.number
            
            # TODO: Find out how to reduce the calls of JCMgeo
#             if sim.rerun_JCMgeo:
#                 self.compute_geometry(sim, **jcm_geo_kwargs)
            
            # Start the simulation if it is not already finished
            if not sim.number in self.finished_sim_numbers:
                # Start to solve the simulation and receive a job ID
                jobID = sim.solve(**jcm_solve_kwargs)
                logger.debug(
                        'Queued simulation {0} of {1} with jobID {2}'.\
                                format(i+1, self.num_sims, sim.jobID))
                jobIDs.append(jobID)
                ID2simNumber[jobID] = sim.number
                
            # wait for N simulations to finish
            if len(jobIDs) != 0:
                if (divmod(i+1, N)[1] == 0) or ((i+1) == self.num_sims):
                    logger.info('Waiting for {} '.format(len(jobIDs)) +
                                  'simulation(s) to finish...')
                    self._wait_for_simulations(jobIDs, ID2simNumber)
                    jobIDs = []
                    ID2simNumber = {}
 
    def _wait_for_simulations(self, ids2waitFor, ID2simNumber):
        """Waits for the job IDS in the list `ids2waitFor` to finish using
        daemon.wait. Failed simulations are appended to the list 
        `self.failed_simulations`, while successful simulations are evaluated
        and stored.
        """
        # Wait for all simulations using daemon.wait with break_condition='any'.
        # In each loop, the results are directly evaluated and saved
        nFinished = 0
        nTotal = len(ids2waitFor)
        logger.debug('Waiting for jobIDs: {}'.format(ids2waitFor))
        while nFinished < nTotal:
            # wait till any simulations are finished
            # deepcopy is needed to protect ids2waitFor from being modified
            # by daemon.wait
            with utils.Capturing() as output:
                indices, thisResults, logs = daemon.wait(deepcopy(ids2waitFor), 
                                                     break_condition = 'any')
            for line in output:
                logger.debug(line)
              
            # Get lists for the IDs of the finished jobs and the corresponding
            # simulation numbers
            finishedIDs = []
            finishedSimNumbers = []
            for ind in indices:
                ID = ids2waitFor[ind]
                iSim = ID2simNumber[ ID ]
                finishedIDs.append(ID)
                finishedSimNumbers.append( iSim )
                
                sim = self.simulations[iSim]
                # Check whether the simulation failed
                if sim.status == 'Failed':
                    self.failed_simulations.append(sim)
                else:
                    # Add the computed results to the Simulation-instance, ...
                    sim._set_jcm_results_and_logs(thisResults[ind], logs[ind])
                    # evaluate them, ...
                    sim.evaluate_results(self.evaluation_func)
                    # and append them to the HDF5 store
                    self.append_store(sim._get_DataFrame())
              
            # Remove/zip all working directories of the finished simulations if
            # wdir_mode is 'zip'/'delete'
            if self._wdir_mode in ['zip', 'delete']:
                for n in finishedSimNumbers:
                    if self._wdir_mode == 'zip':
                        utils.append_dir_to_zip(
                                self.simulations[n].working_dir(), 
                                self._zip_file_path)
                    self.simulations[n].remove_working_directory()
              
            # Update the number of finished jobs and the list with ids2waitFor
            nFinished += len(indices)
            ids2waitFor = [ID for ID in ids2waitFor if ID not in finishedIDs]
    
    def _is_scheduled(self):
        """Checks if make_simulation_schedule was executed"""
        return hasattr(self, 'simulations')
    
    def all_done(self):
        """Checks if all simulations are done, i.e. already in the HDF5 store.
        """
        if (not hasattr(self, 'finished_sim_numbers') or 
            not hasattr(self, 'num_sims')):
            logger.info('Cannot check if all simulations are done before '+
                         '`make_simulation_schedule` was executed.')
            return False
        return set(range(self.num_sims)) == set(self.finished_sim_numbers)

    def run(self, evaluation_func=None, N='all', wdir_mode='keep', 
            zip_file_path = None, jcm_geo_kwargs={}, jcm_solve_kwargs={}):
        """Convenient function to add the resources, run all necessary
        simulations and save the results to the HDF5 store.
        
        Parameters
        ----------
        evaluation_func : callable or None, default None
            Function for result evaluation. If None, only a standard evaluation
            will be executed. See the docs of the 
            Simulation.evaluate_results-method for more info on how to use this
            parameter.
        N : int or 'all', default 'all'
            Number of simulations that will be pushed to the jcm.daemon at a
            time. If 'all', all simulations will be pushed at once. If many 
            simulations are pushed to the daemon, the number of files and the
            size on disk can grow dramatically. This can be avoided by using
            this parameter, while deleting or zipping the working directories
            at the same time using the `wdir_mode` parameter.
        wdir_mode : {'keep', 'zip', 'delete'}, default 'keep'
            The way in which the working directories of the simulations are
            treated. If 'keep', they are left on disk. If 'zip', they are
            appended to the zip-archive controled by `zip_file_path`. If 
            'delete', they are deleted. Caution: if you zip the directories and
            extend your data later in a way that the simulation numbers change,
            problems may occur.
        zip_file_path : str (file path) or None
            Path to the zip file if `wdir_mode` is 'zip'. The file is created
            if it does not exist. If None, the default file name 
            'working_directories.zip' in the current `storage_dir` is used.
        
        `jcm_geo_kwargs` and `jcm_solve_kwargs` are dicts of keyword arguments
        which are directly passed to jcm.geo and jcm.solve, respectively. 
        """
        if self.all_done():
            logger.info('Nothing to run: all simulations finished.')
            return
        
        if not self._is_scheduled():
            logger.info('Please run `make_simulation_schedule` first.')
            return
        
        if zip_file_path is None:
            zip_file_path = os.path.join(self.storage_dir, 
                                         'working_directories.zip')
        if not os.path.isdir(os.path.dirname(zip_file_path)):
            raise OSError('The zip file cannot be created, as the containing'+
                          ' folder does not exist.')
        
        if not wdir_mode in ['keep', 'zip', 'delete']:
            raise ValueError('Unknown wdir_mode: {}'.format(wdir_mode))
            return
        
        # Add class attributes for `_wait_for_simulations`
        self._wdir_mode = wdir_mode
        self._zip_file_path = zip_file_path
        
        # Start the timer
        t0 = time.time()
        
        # Store the metadata of this run
        self._store_metadata()
        
        # Try to add the resources
        if not self._resources_ready():
            self.add_resources()
        self._start_simulations(N=N, evaluation_func=evaluation_func)
        if len(self.failed_simulations) > 0:
            logger.warn('The following simulations failed: {}'.format(
                            [sim.number for sim in self.failed_simulations]))
        
        # Delete/zip working directories from previous runs if needed
        if wdir_mode in ['zip', 'delete'] and hasattr(self, '_wdirs_to_clean'):
            logger.info('Treating old working directories with mode: {}'.format(
                                                                    wdir_mode))
            for dir_ in self._wdirs_to_clean:
                if wdir_mode == 'zip':
                    utils.append_dir_to_zip(dir_, zip_file_path)
                if os.path.isdir(dir_):
                    rmtree(dir_)
        
        logger.info('Total time for all simulations: {}'.format(
                                                utils.tForm(time.time()-t0)))



# Call of the main function
if __name__ == "__main__":
    pass
    
    
    

