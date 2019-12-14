"""Defines functions and classes which are internally used in all parts of
pypmj, but may also be relevant to the user. Most importantly, the
functions `run_simusets_in_save_mode` and `send_status_email` are defined here.

Authors : Carlo Barth

"""

from pypmj import _config, daemon
from pypmj.internals import _IS_PYTHON3
import collections
if _IS_PYTHON3:
    from io import StringIO
else:
    from cStringIO import StringIO
from datetime import timedelta
import errno
import inspect
from itertools import combinations
import logging
from numbers import Number
import numpy as np
import os
import pandas as pd
from six import string_types
import sys
from tempfile import mktemp
import time
import traceback
import zipfile
logger = logging.getLogger(__name__)

# Load values from configuration
CMAP = _config.get('Preferences', 'colormap')
SEND_MAIL = _config.get('Logging', 'send_mail')
MAIL_SERVER = _config.get('Logging', 'mail_server')
EMAIL_ADDRESS = _config.get('User', 'email')


def is_callable(obj):
    """Return whether the object is callable (i.e., some kind of function).

    Note that classes are callable, as are instances with a __call__()
    method.

    """
    if _IS_PYTHON3:
        return isinstance(obj, collections.Callable)
    return callable(obj)

def file_content(file_path):
    """Returns the content of an existing file."""
    with open(file_path, 'r') as f:
        content = f.read()
    return content

def tForm(t1):
    """Returns a well formated time string."""
    return str(timedelta(seconds=t1))


def walk_df(df, col_vals, keys=None):
    """Recursively finds a row in a pandas DataFrame where all values match the
    values given in col_vals for the keys (i.e. column specifiers) in `keys`.

    If no matching rows exist, None is returned. If multiple matching rows
    exist, a list of indices of the matching rows is returned.

    Parameters
    ----------
    df : pandas.DataFrame
        This is the DataFrame in which a matching row should be found. For
        efficiency, it is not checked if the keys are present in the columns of
        df, so this should be checked by the user.
    col_vals : dict or OrderedDict
        A dict that holds the (single) values the matching row of the DataFrame
        should have, so that df.loc[match_index, key) ==  col_vals[key] for all
        keys in the row with index `match_index`. If keys is only a subset of
        the keys in the dict, remaining key-value pairs are ignored.
    keys : sequence (list/tuple/numpy.ndarray/etc.), default None
        keys (i.e. columns in df) to use for the comparison. The keys must be
        present in col_vals. If keys is None, all keys of col_vals are used.

    """
    if keys is None:
        keys = list(col_vals.keys())
    if len(keys) == 0:
        # No keys are left for comparison, but len(df)>1. This means, all
        # remaining rows are matches
        return list(df.index)

    # Reduce DataFrame to those rows that match the current key
    df_sub = df[df[keys[0]] == col_vals[keys[0]]]

    if len(df_sub) > 1:
        # Still multiple rows
        keys.pop(0)
        return walk_df(df_sub, col_vals, keys)
    elif len(df_sub) == 1:
        # Single row
        idx = df_sub.index[0]
        if all([df_sub.at[idx, k] == col_vals[k] for k in keys]):
            # Single row matches
            return idx
        else:
            # Single row does not match
            return None
    else:
        # No row left
        return None


def is_sequence(obj):
    """Checks if a given object is a sequence by checking if it is not a string
    or dict, but has a __len__-method.

    This might fail!

    """
    return not isinstance(obj, (string_types, dict)) and hasattr(obj,
                                                                 '__len__')


def lists_overlap(list_1, list_2):
    """Checks if two lists have no common elements."""
    return not set(list_1).isdisjoint(list_2)


def assign_kwargs_to_functions(functions, kwargs, ignore_unmatched=True):
    """Uses `inspect` to assign which argument in `kwargs` belongs to which of
    the functions in the `functions` list.

    If functions have any common argument names, an Error is raised. If
    `ignore_unmatched` is True, unassigned arguments are ignored.
    Returns a list of kwargs-dictionaries, one for each function.

    """
    # Check proper types
    if not isinstance(functions, list):
        raise TypeError('`functions` must be a list of callables.')
        return
    elif not all([is_callable(func) for func in functions]):
        raise TypeError('`functions` must be a list of callables.')
        return

    # Try to inspect the input arguments of the functions
    try:
        input_args = [inspect.getargspec(func)[0] for func in functions]
    except Exception as e:
        raise RuntimeError('Unable to inspect the input arguments of' +
                           ' the provided functions. Esception: {}'.format(e))
        return
    for combo in combinations(input_args, 2):
        if lists_overlap(combo[0], combo[1]):
            raise RuntimeError('Cannot assign kwargs to functions, as ' +
                               'functions have common input arguments.')
            return
    assigned_kwargs = [{} for i in range(len(functions))]
    for kw in kwargs:
        match = False
        for i, ia in enumerate(input_args):
            if kw in ia:
                assigned_kwargs[i][kw] = kwargs[kw]
                match = True
        if (not match) and (not ignore_unmatched):
            raise Exception('Could not assign `{}` to any of the '.format(kw) +
                            'functions. Set `ignore_unmatched=True` to' +
                            ' ignor such arguments.')
    return assigned_kwargs


def get_len_of_parameter_dict(d):
    """Given a dict, returns the length of the longest sequence in its
    values."""
    if not isinstance(d, dict):
        raise ValueError('Need a dict object for get_len_of_parameter_dict.')
    cols = list(d.keys())
    length = 0
    for c in cols:
        val = d[c]
        if is_sequence(val):
            l = len(val)
        else:
            l = 1
        if l > length:
            length = l
    return length


def check_type_consistency_in_sequence(sequence):
    """Checks if all elements of a sequence have the same type."""
    if len(sequence) < 2:
        return True
    type_ = type(sequence[0])
    return all([isinstance(s, type_) for s in sequence])


def infer_dtype(obj):
    """Tries to infer the numpy.dtype (or equivalent) of the elements of a
    sequence, or the numpy.dtype (or equivalent) of the object intelf if it is
    no sequence."""
    if is_sequence(obj):
        if len(obj) == 0:
            return None
        if hasattr(obj, 'dtype'):
            return obj.dtype
        else:
            if not check_type_consistency_in_sequence(obj):
                raise ValueError('Sequence of type {} has unconsistent types.'.
                                 format(type(obj)))
            return np.dtype(type(obj[0]))
    return np.dtype(type(obj))


def obj_to_fixed_length_Series(obj, length):
    """Generates a pandas Series with a fixed len of `length` with the best
    matching dtype for the object.

    If the object is sequence, the rows of the Series are filled with
    its elements. Otherwise it will be the value of the first row.

    """
    dtype = infer_dtype(obj)
    if not is_sequence(obj):
        obj = [obj]
    if len(obj) > length:
        raise ValueError('Cannot create a fixed_length_Series of ' +
                         'length {} for a sequence of length {}.'.format(
                             length, len(obj)))
        return
    obj = np.array(obj, dtype=dtype)
    series = pd.Series(index=list(range(length)), dtype=dtype)
    series[:] = np.NaN # Fixes bool-case
    series.loc[:len(obj) - 1] = obj
    return series


def computational_costs_to_flat_dict(ccosts, _sub=False):
    """Converts the computational costs dict as returned by JCMsolve to a flat
    dict with only scalar values (i.e. numbers or strings).

    This is useful to store the computational costs in a pandas
    DataFrame. Keys which have sequence values with a length other than
    1 are converted to single values, while appending an underscore plus
    index to the key.

    """

    # Check validity if this is not a sub dict
    if not _sub:
        verrormsg = 'ccosts must be a dict as returned by JCMsolve.'
        if not isinstance(ccosts, dict):
            raise ValueError(verrormsg)
            return
        if 'title' not in ccosts:
            raise ValueError(verrormsg + ' the key `title` is missing.')
            return
        if not ccosts['title'] == 'ComputationalCosts':
            raise ValueError(verrormsg + ' the value of `title` must be ' +
                             '`ComputationalCosts`.')
            return

    converted = {}
    for key in ccosts:
        if not key == 'title':
            val = ccosts[key]
            if is_sequence(val):
                if len(val) == 1:
                    if isinstance(val[0], (string_types, Number)):
                        converted[key] = val[0]
                else:
                    # Sequences longer than 1 occur if a refinement loop was
                    # used
                    for i,v in enumerate(val):
                        digits = int(np.log10(len(val)))+1
                        strfmt = '_{0:0'+str(digits)+'d}'
                        # We do not allow nested sequences
                        if not is_sequence(v):
                            if isinstance(v, (string_types, Number)):
                                converted[key+strfmt.format(i)] = v
            else:
                if isinstance(val, (string_types, Number)):
                    converted[key] = val
                elif isinstance(val, dict):
                    subdict = computational_costs_to_flat_dict(val, _sub=True)
                    for skey in subdict:
                        converted[skey] = subdict[skey]
    return converted


def rename_directories(renaming_dict):
    """Safely renames directories given as old_name:new_name pairs as keys and
    values in the renaming_dict.

    It first renames all old names to unique temporary names, and
    renames these to the new_names in a second step. This produces some
    overhead, but circumvents the problem of overlapping names in the
    old and new names. Safely ignores missing directories.

    """
    # Use only folders that exist
    valid_dict = {key: renaming_dict[key] for key in renaming_dict
                  if os.path.isdir(key)}

    # Step 1: renaming to temporary files
    tmp_dict = {}
    for dir_ in valid_dict:
        parent_dir = os.path.dirname(dir_)
        tmp_name = mktemp(dir=parent_dir)
        os.rename(dir_, tmp_name)
        tmp_dict[dir_] = tmp_name

    # Step 2: rename these folders to the target names
    for dir_ in valid_dict:
        os.rename(tmp_dict[dir_], valid_dict[dir_])


def rm_empty_directory_tail(path, stop_at=None):
    """Removes all empty directories of a path recursively, starting at the
    tail, until a non empty directory is found or `path` is the same directory
    given in `stop_at`."""
    if stop_at is not None:
        if not isinstance(stop_at, string_types):
            raise TypeError('`stop_at` must be a path.')
            return
        if not os.path.isdir(stop_at):
            raise TypeError('`stop_at` must be a path.')
            return
        if (os.path.normcase(os.path.normpath(path)) ==
                os.path.normcase(os.path.normpath(stop_at))):
            return
    if not os.path.isdir(path):
        return
    if not os.listdir(path):
        try:
            os.rmdir(path)
            rm_empty_directory_tail(os.path.dirname(path), stop_at=stop_at)
        except:
            return


def split_path_to_parts(path):
    """Splits a path to its parts, so that `os.path.join(*parts)` gives the
    input path again."""
    parts = []
    head, tail = os.path.split(path)
    while head and head != os.path.sep:
        parts.append(tail)
        head, tail = os.path.split(head)
    if head == os.path.sep:
        tail = os.path.join(head, tail)
    parts.append(tail)
    parts.reverse()
    return parts


def get_folders_in_zip(zipf):
    """Returns a list of all folders and files in the root level of an open
    ZipFile."""
    if not isinstance(zipf, zipfile.ZipFile):
        raise ValueError('`zipfile` must be an open zipfile.ZipFile.')
    folders = []
    for name in zipf.namelist():
        split = split_path_to_parts(name)
        if len(split) <= 2:
            fold = split[0]
            if fold not in folders:
                folders.append(fold)
    return folders


def append_dir_to_zip(directory, zip_file_path):
    """Appends a directory to a zip-archive.

    Raises an exception if the directory is already inside the archive.

    """
    if not os.path.isdir(directory):
        raise ValueError('{} is not a valid directory.'.format(directory))
        return
    rel_to_path = os.path.dirname(directory)
    ziph = zipfile.ZipFile(zip_file_path, 'a')
    folders = get_folders_in_zip(ziph)
    for root, _, files in os.walk(directory):
        relDir = os.path.relpath(root, rel_to_path)
        # Check if the directory is already inside
        if relDir in folders:
            raise Exception('Folder {} is already in the zip-archive'.format(
                relDir))
            ziph.close()
            return
        # Write contents to the archive
        for file_ in files:
            ziph.write(os.path.join(root, file_),
                       os.path.join(relDir, file_))
    ziph.close()


def relative_deviation(sample, reference):
    """Returns the relative deviation d=|A/B-1| of sample A and reference B.

    A can be a (complex) number or a list/numpy.ndarray of (complex)
    numbers. In case of complex numbers, the average relative deviation
    of real and imaginary part (d_real+d_imag)/2 is returned.

    """
    def rel_dev_real(A, B):
        return np.abs(A / B - 1.)
    if isinstance(sample, list):
        sample = np.array(sample)
    if np.any(np.iscomplex(sample)):
        assert np.iscomplex(reference), \
            'relative_deviation for complex numbers is only possible ' +\
            'if the reference is complex as-well.'
        return (rel_dev_real(sample.real, reference.real) +
                rel_dev_real(sample.imag, reference.imag)) / 2.
    else:
        return rel_dev_real(sample, reference.real)


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user. "default" is
    the presumed answer if the user just hits <Enter>.     It must be
    "yes" (the default), "no" or None (meaning     an answer is required
    of the user).

    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        if _IS_PYTHON3:
            choice = input().lower()
        else:
            choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


class DisableLogger(object):
    """Context manager to disable all logging events below specific level."""

    def __init__(self, level=logging.INFO):
        self.level = level

    def __enter__(self):
        logging.disable(self.level)

    def __exit__(self, a, b, c):
        logging.disable(logging.NOTSET)


class Capturing(list):
    """Context manager to capture any output printed to stdout.

    based on: http://stackoverflow.com/questions/16571150/
    how-to-capture-stdout-output-from-a-python-function-call

    """

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout


def send_status_email(text, subject='JCMwave Simulation Information',
                      subject_prefix='', subject_suffix=''):
    """Tries to send a status e-mail with the given `text` using the configured
    e-mail server and address."""
    try:
        import smtplib
        from email.mime.text import MIMEText

        # Create a text/plain message
        msg = MIMEText(text)

        # me == the sender's email address
        # you == the recipient's email address
        me = 'noreply@jcmsuite.automail.de'
        msg['Subject'] = subject_prefix + subject + subject_suffix
        msg['From'] = me
        msg['To'] = EMAIL_ADDRESS  # <- config.py

        # Send the message via our own SMTP server, but don't include the
        # envelope header.
        s = smtplib.SMTP(MAIL_SERVER)
        s.sendmail(me, [EMAIL_ADDRESS], msg.as_string())
        s.quit()
    except Exception as e:
        logger.warn('Sending of status e-mail failed: {}'.format(e))
        return


def __prepare_SimulationSet_after_fail(simuset):
    """Prepares a SimulationSet instance to be run again after termination by
    an Error.

    It reschedules the simulations and shuts down the JCMdaemon.

    """
    if hasattr(simuset, '_wdirs_to_clean'):
        del simuset._wdirs_to_clean
    simuset.make_simulation_schedule()
    try:
        daemon.shutdown()
    except:
        pass


def run_simusets_in_save_mode(simusets, Ntrials=5, **kwargs):
    """Given a list of SimulationSets, tries to run each SimulationSet
    `Ntrials` times, starting at the point where it was terminated by an
    unwanted error.

    The `kwargs` are passed to the run-method of each set or to the
    `send_status_email` utility function. They are automatically
    assigned. Status e-mails are sent if configured in the configuration
    file.

    """

    if not is_sequence(simusets):
        simusets = [simusets]

    # Check simuset validity
    for sset in simusets:
        # dirty check if this is a SimulationSet instance
        if not hasattr(sset, 'STORE_META_GROUPS'):
            raise ValueError('All simusets must be of type SimulationSet.')
            return
        if not sset._is_scheduled():
            raise RuntimeError('{} is not scheduled yet.'.format(sset))
            return

    # Assign the kwargs
    try:
        run_kw, mail_kw = assign_kwargs_to_functions([simusets[0].run,
                                                      send_status_email],
                                                     kwargs)
    except Exception as e:
        logger.warn('An error occured when trying to assign the kwargs to ' +
                    'the `run` and `send_status_email` functions: {}'.
                    format(e) + ' Assuming all kwargs are for the ' +
                    '`run`-method.')
        run_kw = kwargs
        mail_kw = {}

    # Start
    Nsets = len(simusets)
    ti0 = time.time()
    for i, sset in enumerate(simusets):
        trials = 0
        msg = 'Starting SimulationSet {0} of {1}'.format(i + 1, Nsets)
        if SEND_MAIL:
            send_status_email(msg, **mail_kw)

        # Run the simulations
        while trials < Ntrials:
            tt0 = time.time()
            try:
                if trials > 0:
                    # Set up this simuset again after the error
                    __prepare_SimulationSet_after_fail(sset)
                sset.run(**run_kw)
            except KeyboardInterrupt:
                # Allow keyboard interrupts
                return
            except:
                trials += 1
                msg = 'SimulationSet {0} failed at trial {1} of {2}'.\
                      format(i + 1, trials, Ntrials)
                msg += '\n\n***Error Message:\n' + \
                       traceback.format_exc() + '\n***'
                if SEND_MAIL:
                    send_status_email(msg, **mail_kw)
                continue
            break
        ttend = tForm(time.time() - tt0)
        msg = 'Finished SimulationSet {0} of {1}. Runtime: {2}'.\
              format(i + 1, Nsets, ttend)
        if SEND_MAIL:
            send_status_email(msg, **mail_kw)

    tend = tForm(time.time() - ti0)
    msg = 'All simulations finished after {0}'.format(tend)

    if SEND_MAIL:
        send_status_email(msg, **mail_kw)


# -----------------------------------------------------------------------------
#
# The following is based on the code of Giampaolo Rodola that can be found
# here:
#    http://code.activestate.com/recipes/
#                578022-wait-for-pid-and-check-for-pid-existance-posix/
#
# It implements the function `wait_for_all_other_daemons` that can be used
# to wait for all other JCMdaemon processes on the current system to finish.
#
# -----------------------------------------------------------------------------

class _TimeoutExpired(Exception):
    pass


def _pid_exists(pid):
    """Check whether pid exists in the current process table."""
    if pid < 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno == errno.EPERM
    else:
        return True

def _wait_pid(pid, timeout=None):
    """Wait for process with pid 'pid' to terminate and return its
    exit status code as an integer.

    If pid is not a children of os.getpid() (current process) just
    waits until the process disappears and return None.

    If pid does not exist at all return None immediately.

    Raise _TimeoutExpired on timeout expired (if specified).
    """
    def check_timeout(delay):
        if timeout is not None:
            if time.time() >= stop_at:
                raise _TimeoutExpired
        time.sleep(delay)
        return min(delay * 2, 0.04)

    if timeout is not None:
        waitcall = lambda: os.waitpid(pid, os.WNOHANG)
        stop_at = time.time() + timeout
    else:
        waitcall = lambda: os.waitpid(pid, 0)

    delay = 0.0001
    while 1:
        try:
            retpid, status = waitcall()
        except OSError as err:
            if err.errno == errno.EINTR:
                delay = check_timeout(delay)
                continue
            elif err.errno == errno.ECHILD:
                # This has two meanings:
                # - pid is not a child of os.getpid() in which case
                #   we keep polling until it's gone
                # - pid never existed in the first place
                # In both cases we'll eventually return None as we
                # can't determine its exit status code.
                while 1:
                    if _pid_exists(pid):
                        delay = check_timeout(delay)
                    else:
                        return
            else:
                raise
        else:
            if retpid == 0:
                # WNOHANG was used, pid is still running
                delay = check_timeout(delay)
                continue
            # process exited due to a signal; return the integer of
            # that signal
            if os.WIFSIGNALED(status):
                return os.WTERMSIG(status)
            # process exited using exit(2) system call; return the
            # integer exit(2) system call has been called with
            elif os.WIFEXITED(status):
                return os.WEXITSTATUS(status)
            else:
                # should never happen
                raise RuntimeError("unknown process exit status")

def _get_daemon_pids():
    """Returns the PIDs for all currently running JCMdaemon processes on UNIX
    systems."""
    import psutil
    procs = []
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['pid', 'name'])
        except psutil.NoSuchProcess:
            pass
        else:
            if 'JCMdaemon' in pinfo['name']:
                procs.append(pinfo['pid'])
    return procs

def wait_for_all_other_daemons():
    """Waits for all other currently active JCMdaemon processes to finish on
    UNIX systems.""" 
    _daemon_pids = _get_daemon_pids()
    while len(_daemon_pids) > 0:
        _wait_pid(_daemon_pids[0])
        _daemon_pids = _get_daemon_pids()

def convert_keys(keys):
    """takes a dictionary containing some or all of the three keys: constants,
    parameters and geometry which are dictionaries containing values which may
    be numpy arrays and returns a dictionary containing the keys contained in
    the previous subdictionaries where each np.ndarray has been replaced with
    the first entry in the array."""
    correct_fields = ['constants', 'geometry', 'parameters']
    loop_fields = ['geometry','parameters']
    if set(correct_fields).isdisjoint(set(keys.keys())):
        raise ValueError('`keys` must contain at least one of the keys .' +
                         ' {} or all the keys '.format(loop_indication) +
                         'necessary to compile the JCM-template files.')
    new_keys = {}
    for _k in correct_fields:
        if _k in keys:
            if not isinstance(keys[_k], dict):
                raise ValueError('The values for the keys {}'.format(
                                 loop_indication) + ' must be of type ' +
                                 '`dict`')
            subdict = keys[_k]
            for _k2 in subdict:
                if (isinstance(subdict[_k2],np.ndarray) or isinstance(subdict[_k2],list) ) and _k in loop_fields    :
                    new_keys[_k2] = subdict[_k2][0]
                else:
                    new_keys[_k2] = subdict[_k2]
            #new_keys[_k, keys[_k]]
    return new_keys
        

if __name__ == "__main__":
    pass
