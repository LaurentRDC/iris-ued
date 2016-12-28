
from . import pyqtgraph as pg
from .pyqtgraph import QtCore

class WorkThread(QtCore.QThread):
    """
    Object taking care of threading computations.
    
    Signals
    -------
    done_signal
        Emitted when the function evaluation is over.
    in_progress_signal
        Emitted when the function evaluation starts.
    results_signal
        Emitted when the function evaluation is over. Carries the results
        of the computation.
        
    Attributes
    ----------
    function : callable
        Function to be called
    args : tuple
        Positional arguments of function
    kwargs : dict
        Keyword arguments of function
    results : object
        Results of the computation
    
    Examples
    --------
    >>> function = lambda x : x ** 10
    >>> result_function = lambda x: print(x)
    >>> worker = WorkThread(function, 2)  # 2 ** 10
    >>> worker.results_signal.connect(result_function)
    >>> worker.start()      # Computation starts only when this method is called
    """

    results_signal = QtCore.pyqtSignal(str, name = 'results_signal')
    done_signal = QtCore.pyqtSignal(bool, name = 'done_signal')
    in_progress_signal = QtCore.pyqtSignal(bool, name = 'in_progress_signal')

    def __init__(self, function, args = (), kwargs = {}):
        
        QtCore.QThread.__init__(self)
        self.function = function
        self.args = args
        self.kwargs = kwargs
    
    def __del__(self):
        self.wait()
    
    def run(self):
        self.in_progress_signal.emit(True)
        result = self.function(*self.args, **self.kwargs)   
        self.done_signal.emit(True)  
        self.results_signal.emit(result)

def spectrum_colors(num_colors):
    """
    Generates a set of QColor 0bjects corresponding to the visible spectrum.
    
    Parameters
    ----------
    num_colors : int
        number of colors to return
    
    Yields
    ------
    colors : QColor 
        Can be used with PyQt and PyQtGraph
    """
    # Hue values from 0 to 1
    hue_values = [i/num_colors for i in range(num_colors)]
    
    # Scale so that the maximum is 'purple':
    hue_values = [0.8*hue for hue in hue_values]
    
    colors = list()
    for h in reversed(hue_values):
        yield pg.hsvColor(hue = h, sat = 0.7, val = 0.9)