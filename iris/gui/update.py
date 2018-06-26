# -*- coding: utf-8 -*-
"""
Update checks management
========================

Based on the ``outdated`` package
"""
import json
from subprocess import Popen, run, PIPE
from urllib.error import URLError
from urllib.request import urlopen

from pkg_resources import parse_version

from .. import __version__

try:
    from subprocess import CREATE_NEW_PROCESS_GROUP
    WINDOWS = True
except ImportError:
    WINDOWS = False 

DETACHED_PROCESS = 0x00000008          # 0x8 | 0x200 == 0x208

def update_available():
    """
    Checks whether the currently-installed iris-ued is outdated.
    
    Returns
    -------
    is_outdated : bool
        Whether or not a new version is available
    latest : str
        Latest available version, currently installed or not.
    
    Raises
    ------
    ConnectionError : if connection to PyPI could not be made.
    """
    url = 'https://pypi.org/pypi/iris-ued/json'

    try:
        response = urlopen(url).read().decode('utf-8')
    except URLError:
        raise ConnectionError('No connection available.')

    latest_version = parse_version(
        json.loads(response)['info']['version'])
    
    is_outdated = latest_version > parse_version(__version__)
    return is_outdated, str(latest_version)

def update_in_background():
    """ Update iris in the background. If iris-ued was installed with conda, it will be updated through conda as well;
    otherwise, pip is used. """
    # Determine if conda is installed
    try:
        conda_installed = run(['conda', '--version']).exitcode == 0
    except:
        conda_installed = False

    # Determine if iris-ued was installed with conda
    # If so, we update it with conda
    if conda_installed:
        conda_list = json.loads(run(['conda', 'list', '--json', '--no-pip'], stdout = PIPE).stdout)
        update_with_conda = 'iris-ued' in {item['name'] for item in conda_list}

    flags = DETACHED_PROCESS
    if WINDOWS:
        flags = flags | CREATE_NEW_PROCESS_GROUP
    
    if conda:
        Popen(['conda', 'update', 'iris-ued', '--yes'], creationflags = flags)
    else:
        Popen(['pip', 'install', '--upgrade', 'iris-ued', '-y'], creationflags = flags)
