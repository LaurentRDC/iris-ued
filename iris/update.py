# -*- coding: utf-8 -*-
"""
Update checks management
========================

Based on the ``outdated`` package
"""
import json
from urllib.error import URLError
from urllib.request import urlopen

from pkg_resources import parse_version

from . import __version__

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
