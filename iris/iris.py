# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:18:35 2016

@author: Laurent
"""

#Core functions
from dataset import DiffractionDataset, PowderDiffractionDataset, read, cast_to_16_bits
from progress_widget import InProgressWidget
import os

#GUI backends
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
import numpy as n

image_folder = os.path.join(os.path.dirname(__file__), 'images')

def spectrum_colors(num_colors):
    """
    Generates a set of QColor 0bjects corresponding to the visible spectrum.
    
    Parameters
    ----------
    num_colors : int
        number of colors to return
    
    Returns
    -------
    colors : list of QColors. Can be used with PyQt and PyQtGraph
    """
    # Hue values from 0 to 1
    hue_values = [i/num_colors for i in range(num_colors)]
    
    # Scale so that the maximum is 'purple':
    hue_values = [0.8*hue for hue in hue_values]
    
    colors = list()
    for h in reversed(hue_values):
        colors.append(pg.hsvColor(hue = h, sat = 0.7, val = 0.9))
    return colors

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
    >>> worker = WorkThread(function, 2)  # Will calculate 2 ** 10
    >>> worker.done_signal.connect(result_function)
    >>> worker.start()      # Computation starts only when this method is called
    """
    done_signal = QtCore.pyqtSignal(bool, name = 'done_signal')
    in_progress_signal = QtCore.pyqtSignal(bool, name = 'in_progress_signal')
    results_signal = QtCore.pyqtSignal(object, name = 'results_signal')
    
    def __init__(self, function, *args, **kwargs):
        
        QtCore.QThread.__init__(self)
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.result = None
    
    def __del__(self):
        self.wait()
    
    def run(self):
        self.in_progress_signal.emit(True)
        self.result = self.function(*self.args, **self.kwargs)  # Computation is here      
        self.done_signal.emit(True)        
        self.results_signal.emit(self.result)

class ImageViewer(pg.ImageView):
    """
    QWidget that displays images or entire datasets from dataset.DiffractionDataset.
    
    Attributes
    ----------
    mask_shown : bool
        If False, the beam-block mask is not currently displayed.
    center_finder_shown : bool
        If False, the center-finder widget is not currently displayed.
    peak_dynamics_region_shown : bool
        If False, the peak_dynamics_region widget is not currently displayed.
    center_position : tuple of ints, shape (2,)
        Center of the ring Region-of-interest
    mask_position : tuple of ints, shape (4,)
        Returns the x,y limits of the rectangular beam block mask.
    dataset
        DiffractionDataset instance from the parent widget.
        
    Methods
    -------
    display_data
        Display entire datasets or ndarrays
    
    toggle_radav_tools
        Toggle the appearance of the radial-averaging tools 'mask' and 'center_finder'
        
    toggle_peak_dynamics_region
        Toggle the appearance of the peak_dynamics_region widget
    """
    
    def __init__(self, parent):        
        super(ImageViewer, self).__init__()
        self.parent = parent
        
        # Radial-averaging tools
        self.mask = pg.ROI(pos = [800,800], size = [200,200], pen = pg.mkPen('r'))
        self.center_finder = pg.CircleROI(pos = [1000,1000], size = [200,200], pen = pg.mkPen('r'))
        
        # Single crystal peak dynamics tool
        self.peak_dynamics_region = pg.CircleROI(pos = [800,800], size = [200,200], pen = pg.mkPen('b'))
        
        # Miscellaneous
        self.in_progress_widget = InProgressWidget(parent = self)
        
        # Build widget
        self._init_ui()
        self._connect_signals()
        
    def _init_ui(self):
        
        # Add handles to the beam block mask
        self.mask.addScaleHandle([1, 1], [0, 0])
        self.mask.addScaleHandle([0, 0], [1, 1])
        
        # Hide the in-progress widget for now.
        self.in_progress_widget.hide()
    
    def _connect_signals(self):
        pass
    
    def resizeEvent(self, event):
        # Resize the in_progress_widget when the widget is resized
        self.in_progress_widget.resize(event.size())
        event.accept()
    
    # Attribute ---------------------------------------------------------------
    
    @property
    def dataset(self):
        return self.parent.dataset
        
    @property
    def peak_dynamics_region_shown(self):
        return self.peak_dynamics_region.getViewBox() is not None
        
    @property
    def mask_shown(self):
        return self.mask.getViewBox() is not None
    
    @property
    def center_finder_shown(self):
        return self.center_finder.getViewBox() is not None
            
    @property
    def mask_position(self):
        if not self.mask_shown: return None
            
        rect = self.mask.parentBounds().toRect()
        #If coordinate is negative, return 0
        x1 = max(0, rect.topLeft().x() )
        x2 = max(0, rect.x() + rect.width() )
        y1 = max(0, rect.topLeft().y() )
        y2 = max(0, rect.y() + rect.height() )
        return y1, y2, x1, x2       #Flip output since image viewer plots transpose
    
    @property
    def center_position(self):
        if not self.center_finder_shown: return None
            
        corner_x, corner_y = self.center_finder.pos().x(), self.center_finder.pos().y()
        radius = self.center_finder.size().x()/2.0
        #Flip output since image viewer plots transpose...
        return corner_y + radius, corner_x + radius
    
    # Methods -----------------------------------------------------------------
    
    @QtCore.pyqtSlot(object)
    def display_data(self, image = None):
        """
        Displays images or entire dataset.
        
        Parameters
        ----------
        image : ndarray, optional
            ndarray of a diffraction image.
        dataset : DiffractionDataset object, optional
            Will plot a time-series of images, with time slider below. 
            The computation of a time-series takes a few tens of seconds.
        """            
        if image is not None:
            self.setImage(image)
        elif isinstance(self.dataset, DiffractionDataset):
            # Create a thread to compute the image series array
            # This prevents the GUI from freezing.
            self.worker = WorkThread(self.dataset.image_series, True)
            self.worker.results_signal.connect(self._display_image_series)
            
            # Show a progress widget
            self.worker.in_progress_signal.connect(self.in_progress_widget.show)
            self.worker.done_signal.connect(self.in_progress_widget.hide)
            
            self.worker.start()
        else:
            #Default image is Gaussian noise
            self.setImage(n.random.rand(2048, 2048))
    
    @QtCore.pyqtSlot(object)
    def _display_image_series(self, results):
        """ Display a dataset.DiffractionDataset image series as a 3D array """
        self.time_points, self.data = results
        self.setImage(self.data, xvals = self.time_points, axes = {'t':0, 'x':1, 'y':2})
    
    @QtCore.pyqtSlot(bool)
    def toggle_peak_dynamics_region(self, show):
        if show:
            self.addItem(self.peak_dynamics_region)
        else:
            self.removeItem(self.peak_dynamics_region)
        
    @QtCore.pyqtSlot(bool)
    def toggle_radav_tools(self, show):
        """ Hides the radial-averaging tools if currently displayed, and vice-versa. """
        if show:
            self.addItem(self.mask)
            self.addItem(self.center_finder)
        else:
            self.removeItem(self.mask)
            self.removeItem(self.center_finder)


class SingleCrystalToolsWidget(QtGui.QWidget):
    """ 
    Widget displaying tools for single-crystal diffraction analysis.
    
    Attributes
    ----------
    parent : QWidget
        Iris QWidget in which this instance is contained. While technically this
        instance could be in a QSplitter, 'parent' refers to the Iris QWidget 
        specifically.
    dataset : DiffractionDataset
        Object from the parent widget.
        
    Methods
    -------
    """
    def __init__(self, parent):
        super(SingleCrystalToolsWidget, self).__init__()
        self.parent = parent
        self.peak_dynamics_viewer = pg.PlotWidget(title = 'Peak dynamics measurement', 
                                                  labels = {'left': 'Intensity (a. u.)', 'bottom': ('time', 'ps')})
    
    def _init_ui(self):
        pass
    
    def _connect_signals(self):
        pass

    def update_peak_dynamics(self):
        self.peak_dynamics_viewer.clear()
        
        #Get region
        min_x, max_x = self.peak_dynamics_region.getRegion()
        time, intensity = self.dataset.peak_dynamics(min_x, max_x)
        colors = spectrum_colors(len(time))
        self.peak_dynamics_viewer.plot(time, intensity, pen = None, symbol = 'o', 
                                       symbolPen = [pg.mkPen(c) for c in colors], symbolBrush = [pg.mkBrush(c) for c in colors], symbolSize = 3)
        
        # If the use has zoomed on the previous frame, auto range might be disabled.
        self.peak_dynamics_viewer.enableAutoRange()

    def showEvent(self, event):
        self.parent.toggle_sc_viewer.setChecked(True)
        event.accept()
    
    def hideEvent(self, event):
        self.parent.toggle_plot_viewer.setChecked(False)
        event.accept()
    
    @property
    def dataset(self):
        return self.parent.dataset
    
    
class PowderToolsWidget(QtGui.QWidget):
    """
    Widgets displaying two pyqtgraph.PlotWidget widgets. The first one is for radially-averaged
    diffraction patterns, while the other is for time-dynamics of specific regions of
    the radially-averaged diffractiona patterns.
    
    Attributes
    ----------
    parent : QWidget
        Iris QWidget in which this instance is contained. While technically this
        instance could be in a QSplitter, 'parent' refers to the Iris QWidget 
        specifically.
    
    Components
    ----------
    radial_pattern_viewer : PlotWidget
    
    peak_dynamics_viewer : PlotWidget
    
    Methods
    -------
    """
    def __init__(self, parent):
        
        super(PowderToolsWidget, self).__init__()
        
        self.parent = parent
                
        self.radial_pattern_viewer = pg.PlotWidget(title = 'Radially-averaged pattern(s)', 
                                                   labels = {'left': 'Intensity (counts)', 'bottom': 'Scattering length (1/A)'})
        self.peak_dynamics_viewer = pg.PlotWidget(title = 'Peak dynamics measurement', 
                                                  labels = {'left': 'Intensity (a. u.)', 'bottom': ('time', 'ps')})
        self.peak_dynamics_region = pg.LinearRegionItem()
        
        # Background curve is an item so that it can be easily added and removed to a plot.
        # Otherwise, using plot(...) does not permit removing the curve without
        # clearing the entire radial_plot_viewer
        self.background_curve_item = None
        
        # Need 7 lines because biexponential fit has 7 free parameters
        self.inelastic_background_lines = [pg.InfiniteLine(angle = 90, movable = True, pen = pg.mkPen('b')) for i in range(10)]
        
        # In progress widgets
        self.progress_widget_radial_patterns = InProgressWidget(parent = self.radial_pattern_viewer)
        
        self._init_ui()
        self._connect_signals()
    
    def _init_ui(self):
        
        # Display buttons
        self.show_inelastic_background_btn = QtGui.QPushButton(QtGui.QIcon(os.path.join(image_folder, 'analysis.png')), 'Show inelastic background', parent = self)
        self.subtract_inelastic_background_btn = QtGui.QPushButton(QtGui.QIcon(os.path.join(image_folder, 'analysis.png')), 'Subtract inelastic background', parent = self)
        
        self.show_inelastic_background_btn.setCheckable(True)
        self.subtract_inelastic_background_btn.setCheckable(True)
        
        # Hide progress widgets
        self.progress_widget_radial_patterns.hide()
        
        # Plot scales
        self.peak_dynamics_viewer.getPlotItem().enableAutoRange()
        
        # Create horizontal splitter
        self.splitter = QtGui.QSplitter(QtCore.Qt.Vertical)
        self.splitter.addWidget(self.radial_pattern_viewer)
        self.splitter.addWidget(self.peak_dynamics_viewer)
        
        # Buttons
        self.plot_buttons = QtGui.QHBoxLayout()
        self.plot_buttons.addWidget(self.show_inelastic_background_btn)
        self.plot_buttons.addWidget(self.subtract_inelastic_background_btn)
        
        # TODO: insert buttons
        self.layout = QtGui.QVBoxLayout()
        self.layout.addLayout(self.plot_buttons)
        self.layout.addWidget(self.splitter)
        self.setLayout(self.layout)
        
        self._hide_peak_dynamics_setup()
    
    def _connect_signals(self):
        # Changing peak dynamics in real-time is too slow for now
        self.peak_dynamics_region.sigRegionChangeFinished.connect(self.update_peak_dynamics_plot)
        #If the splitter handle is moved, resize the progress widget
        self.splitter.splitterMoved.connect(lambda: self.progress_widget_radial_patterns.resize(self.radial_pattern_viewer.size()))
        
        # Update plots if buttons are pressed
        self.show_inelastic_background_btn.toggled.connect(self.set_background_curve)
        self.subtract_inelastic_background_btn.toggled.connect(self.display_radial_averages)
        self.subtract_inelastic_background_btn.toggled.connect(self.update_peak_dynamics_plot)
    
    def resizeEvent(self, event):
        # Resize the in_progress_widget when the widget is resized
        self.progress_widget_radial_patterns.resize(self.radial_pattern_viewer.size())
        event.accept()
    
    def showEvent(self, event):
        self.parent.toggle_plot_viewer.setChecked(True)
        event.accept()
    
    def hideEvent(self, event):
        self.parent.toggle_plot_viewer.setChecked(False)
        event.accept()
    
    @property
    def dataset(self):
        return self.parent.dataset
        
    @property
    def peak_dynamics_region_shown(self):
        return self.peak_dynamics_region.getViewBox() is not None
    
    @property
    def inelastic_lines_shown(self):
        return self.inelastic_background_lines[0].getViewBox() is not None
    
    @property
    def inelastic_background_lines_positions(self):
        intersects = list()
        for line in self.inelastic_background_lines:
            intersects.append( line.value() )
        return intersects
    
    @property
    def show_inelastic_background(self):
        return self.show_inelastic_background_btn.isChecked()
    
    @property
    def subtract_inelastic_background(self):
        return self.subtract_inelastic_background_btn.isChecked()
    
    def update_peak_dynamics_plot(self):
        self.peak_dynamics_viewer.clear()
        
        #Get region
        min_x, max_x = self.peak_dynamics_region.getRegion()
        time, intensity = self.dataset.radial_peak_dynamics(min_x, max_x, subtract_background = self.subtract_inelastic_background)
        colors = spectrum_colors(len(time))
        self.peak_dynamics_viewer.plot(time, intensity, pen = None, symbol = 'o', 
                                       symbolPen = [pg.mkPen(c) for c in colors], symbolBrush = [pg.mkBrush(c) for c in colors], symbolSize = 4)
        
        # If the use has zoomed on the previous frame, auto range might be disabled.
        self.peak_dynamics_viewer.enableAutoRange()
    
    @QtCore.pyqtSlot(bool)
    def set_background_curve(self, set_curve):
        if set_curve:
            try:
                background_curve = self.dataset.inelastic_background(time = 0.0)
            except KeyError:  #Background curve hasn't been determined
                background_curve = None
        else:
            background_curve = None
        
        #Update plot
        if background_curve is not None:
            self.background_curve_item = pg.PlotDataItem(background_curve.xdata, background_curve.ydata, pen = None, symbol = 'o',
                                            symbolPen = pg.mkPen('r'), symbolBrush = pg.mkBrush('r'), symbolSize = 2)
            self.radial_pattern_viewer.addItem(self.background_curve_item)
        else:
            # Redraw the plot. This will clear all curves first, including the last
            # background curve
            self.radial_pattern_viewer.removeItem(self.background_curve_item)
    
    @QtCore.pyqtSlot(bool)
    def untoggle_peak_dynamics_setup(self, show):
        if not show:
            self._display_peak_dynamics_setup()
        else:
            self._hide_peak_dynamics_setup()
            
    @QtCore.pyqtSlot(bool)
    def toggle_peak_dynamics_setup(self, show):
        if show:
            self._display_peak_dynamics_setup()
        else:
            self._hide_peak_dynamics_setup()
    
    def toggle_inelastic_background_setup(self):
        if self.inelastic_lines_shown:
            self._hide_inelastic_background_setup()
        else:
            self._display_inelastic_background_setup()
        
    def display_radial_averages(self):
        """ 
        Display the radial averages of a dataset, if available. 
        If the radial averages are not available, the function returns without 
        plotting anything.
        """
        if self.isHidden:
            self.show()
        
        self.radial_pattern_viewer.clear()
        self.radial_pattern_viewer.enableAutoRange()
        self.peak_dynamics_viewer.clear()
        self.peak_dynamics_viewer.enableAutoRange()
        
        try:
            curves = self.dataset.radial_pattern_series(self.subtract_inelastic_background)
        except:     # HDF5 file with radial averages is not found
            return
        
        # Get timedelay colors and plot
        colors = spectrum_colors(len(curves))
        for color, curve in zip(colors, curves):
            self.radial_pattern_viewer.plot(curve.xdata, curve.ydata, pen = None, symbol = 'o',
                                            symbolPen = pg.mkPen(color), symbolBrush = pg.mkBrush(color), symbolSize = 3)
        
        self._display_peak_dynamics_setup()
        self.update_peak_dynamics_plot()

    def _display_peak_dynamics_setup(self):
        self.radial_pattern_viewer.addItem(self.peak_dynamics_region, ignoreBounds = False)
    
    def _display_inelastic_background_setup(self):
        #Determine curve range
        try:
            xmin, xmax = self.radial_pattern_viewer.getPlotItem().listDataItems()[0].dataBounds(ax = 0)
        except:
            return
        
        # If no lines are plotted, xmin and xmax will be None
        if xmin is None or xmax is None:
            return

        #Distribute lines equidistantly            
        dist_between_lines = float(xmax - xmin)/len(self.inelastic_background_lines)
        pos = xmin
        for line in self.inelastic_background_lines:
            line.setValue(pos)
            self.radial_pattern_viewer.addItem(line)
            pos += dist_between_lines
        
    def _hide_peak_dynamics_setup(self):
        self.radial_pattern_viewer.removeItem(self.peak_dynamics_region)
    
    def _hide_inelastic_background_setup(self):
        for line in self.inelastic_background_lines:
            self.radial_pattern_viewer.removeItem(line)
        
class Iris(QtGui.QMainWindow):
    """
    Time-delay data viewer for averaged data.
    
    Components
    ----------
    image_viewer : ImageViewer widget
        Main dataset interface, through which is shown a time-series of averaged
        TIFFs.
    plot_viewer : PowderToolsWidget
        Plots and tools to analyze polycrystalline diffraction dataset
    sc_viewer : SingleCrystalToolsWidget
        Plots and tools to analyze single-crystal (sc) diffraction dataset
    
    Attributes
    ----------
    dataset : DiffractionDataset
        Dataset object
    """
    dataset_to_plot = QtCore.pyqtSignal(object, name = 'dataset_to_plot')
    
    def __init__(self):
        
        super(Iris, self).__init__()
        
        self.dataset = None
        
        self.image_viewer = ImageViewer(parent = self)
        self.plot_viewer = PowderToolsWidget(parent = self)
        self.sc_viewer = SingleCrystalToolsWidget(parent = self)
        
        self._init_actions()
        self._init_ui()
        self._connect_signals()
    
    def _init_ui(self):
        
        # UI components
        self.file_dialog = QtGui.QFileDialog(parent = self)      
        self.menubar = self.menuBar()
        self._create_menu()
        self.splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        
        #Taskbar icon
        self.setWindowIcon(QtGui.QIcon(os.path.join(image_folder, 'eye.png')))
        
        #Create window
        self.splitter.addWidget(self.sc_viewer)
        self.splitter.addWidget(self.image_viewer)
        self.splitter.addWidget(self.plot_viewer)
        self.layout = QtGui.QGridLayout()
        self.layout.addWidget(self.splitter, 0, 0)
        
        self.sc_viewer.hide()
        self.plot_viewer.hide()
        
        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)
        
        #Window settings ------------------------------------------------------
        self.setGeometry(500, 500, 800, 800)
        self.setWindowTitle('Iris - UED data exploration')
        self.center_window()
        self.show()
    
    def _init_actions(self):
        
        # General -------------------------------------------------------------
        self.powder_directory_action = QtGui.QAction(QtGui.QIcon(os.path.join(image_folder, 'locator.png')), '&Powder dataset', self)
        self.powder_directory_action.triggered.connect(self.powder_directory_locator)
        
        self.single_crystal_directory_action = QtGui.QAction(QtGui.QIcon(os.path.join(image_folder, 'locator.png')), '&(beta) Single-crystal dataset', self)
        self.single_crystal_directory_action.triggered.connect(self.single_crystal_directory_locator)
        
        self.picture_action = QtGui.QAction(QtGui.QIcon(os.path.join(image_folder, 'diffraction.png')), '&Picture', self)
        self.picture_action.triggered.connect(self.picture_locator)
        
        # Single-crystal ------------------------------------------------------
        self.toggle_sc_viewer = QtGui.QAction(QtGui.QIcon(os.path.join(image_folder, 'wand.png')), '&(beta) Show/hide single-crystal tools', self)
        self.toggle_sc_viewer.setCheckable(True)
        self.toggle_sc_viewer.toggled.connect(self.image_viewer.toggle_peak_dynamics_region)
        self.toggle_sc_viewer.toggled.connect(self.show_sc_viewer)
        
        # Polycrystaline ------------------------------------------------------
        self.set_radav_tools = QtGui.QAction(QtGui.QIcon(os.path.join(image_folder, 'analysis.png')), '&Compute radial averages with beamblock mask and center finder', self)
        self.set_radav_tools.triggered.connect(self.compute_radial_average)    
        self.set_radav_tools.setEnabled(False)
        
        self.toggle_radav_tools = QtGui.QAction(QtGui.QIcon(os.path.join(image_folder, 'toggle.png')), '&Show/hide radial-averaging tools', self)
        self.toggle_radav_tools.setCheckable(True)
        self.toggle_radav_tools.toggled.connect(self.image_viewer.toggle_radav_tools)
        self.toggle_radav_tools.toggled.connect(self.set_radav_tools.setEnabled)
        
        self.toggle_plot_viewer = QtGui.QAction(QtGui.QIcon(os.path.join(image_folder, 'toggle.png')), '&Show/hide radial patterns', self)
        self.toggle_plot_viewer.setCheckable(True)
        self.toggle_plot_viewer.toggled.connect(self.show_plot_viewer)
                
        self.set_inelastic_background = QtGui.QAction(QtGui.QIcon(os.path.join(image_folder, 'analysis.png')), '&Compute the inelastic scattering background', self)
        self.set_inelastic_background.triggered.connect(self.compute_inelastic_background)
        self.set_inelastic_background.setEnabled(False)
        
        self.set_auto_inelastic_background = QtGui.QAction(QtGui.QIcon(os.path.join(image_folder, 'wand.png')), '&(beta) Automatically compute the inelastic scattering background', self)
        self.set_auto_inelastic_background.triggered.connect(lambda: self.compute_inelastic_background(mode = 'auto'))
        
        self.toggle_inelastic_background_tools = QtGui.QAction(QtGui.QIcon(os.path.join(image_folder, 'toggle.png')), '&Show/hide inelastic background fit tools', self)
        self.toggle_inelastic_background_tools.setCheckable(True)
        self.toggle_inelastic_background_tools.toggled.connect(self.plot_viewer.toggle_inelastic_background_setup)
        self.toggle_inelastic_background_tools.toggled.connect(self.set_inelastic_background.setEnabled)
        self.toggle_inelastic_background_tools.toggled.connect(self.plot_viewer.untoggle_peak_dynamics_setup)
    
    def _create_menu(self):
        """ Menu and actions for single crystal tools. """
        
        file_menu = self.menubar.addMenu('&File')
        file_menu.addAction(self.powder_directory_action)
        file_menu.addAction(self.single_crystal_directory_action)
        file_menu.addSeparator()
        file_menu.addAction(self.picture_action)
        
        powder_menu = self.menubar.addMenu('&Polycrystalline tools')
        powder_menu.addAction(self.toggle_plot_viewer)
        powder_menu.addSeparator()
        powder_menu.addAction(self.toggle_radav_tools)
        powder_menu.addAction(self.set_radav_tools)
        powder_menu.addSeparator()
        powder_menu.addAction(self.toggle_inelastic_background_tools)
        powder_menu.addAction(self.set_inelastic_background)
        powder_menu.addAction(self.set_auto_inelastic_background)
        
        sc_menu = self.menubar.addMenu('&Single-crystal tools')
        sc_menu.addAction(self.toggle_sc_viewer)

    def _connect_signals(self):
        # Nothing to see here yet
        pass
    
    # File handling -----------------------------------------------------------
        
    def powder_directory_locator(self):
        """ 
        Activates a file dialog that selects the data directory to be processed. If the folder
        selected is one with processed images (then the directory name is C:\\...\\processed\\),
        return data 'root' directory.
        """
        
        directory = self.file_dialog.getExistingDirectory(self, 'Open diffraction dataset', 'C:\\')
        directory = os.path.abspath(directory)
        self.dataset = PowderDiffractionDataset(directory)        
        self.image_viewer.display_data()
        
        # Plot radial averages if they exist. If error, no plot. This is handled
        # by the plot viewer
        self.plot_viewer.display_radial_averages()
    
    def single_crystal_directory_locator(self):
        """ 
        Activates a file dialog that selects the data directory to be processed. If the folder
        selected is one with processed images (then the directory name is C:\\...\\processed\\),
        return data 'root' directory.
        """
        pass

    def picture_locator(self):
        """ Open a file dialog to select an image to view """
        filename = self.file_dialog.getOpenFileName(self, 'Open diffraction picture', 'C:\\')
        filename = os.path.abspath(filename)
        self.toggle_plot_viewer.setChecked(False)
        self.image_viewer.display_data(image = cast_to_16_bits(read(filename)))
    
    # Display/show slots  -----------------------------------------------------
    
    @QtCore.pyqtSlot(bool)
    def show_plot_viewer(self, show):
        if show:
            self.plot_viewer.show()
        else:
            self.plot_viewer.hide()
    
    @QtCore.pyqtSlot(bool)
    def show_sc_viewer(self, show):
        if show:
            self.sc_viewer.show()
        else:
            self.sc_viewer.hide()
    
    # Computations ------------------------------------------------------------
    
    def compute_radial_average(self):
        if self.dataset is not None:
            self.plot_viewer.show()
            
            #Thread the computation
            self.worker = WorkThread(self.dataset.radial_average_series, self.image_viewer.center_position, self.image_viewer.mask_position)
            self.worker.in_progress_signal.connect(self.plot_viewer.progress_widget_radial_patterns.show)
            self.worker.done_signal.connect(self.plot_viewer.progress_widget_radial_patterns.hide)
            self.worker.done_signal.connect(self.plot_viewer.display_radial_averages)
            self.worker.start()
            # Only hide the tools after computation has started
            self.toggle_radav_tools.setChecked(False)
    
    def compute_inelastic_background(self, mode = None):
        """
        Compute inelastic scattering background. if mode == 'auto', positions are
        automatically determined using the continuous wavelet transform
        """
        if self.dataset is not None:
            self.toggle_inelastic_background_tools.setChecked(False)
            
            #Thread the computation
            if mode == 'auto':
                self.worker = WorkThread(self.dataset.inelastic_background_fit, None)
            else:
                self.worker = WorkThread(self.dataset.inelastic_background_fit, self.plot_viewer.inelastic_background_lines_positions)
                
            self.worker.in_progress_signal.connect(self.plot_viewer.progress_widget_radial_patterns.show)
            self.worker.done_signal.connect(self.plot_viewer.progress_widget_radial_patterns.hide)
            self.worker.done_signal.connect(self.plot_viewer.display_radial_averages)
            # Show the inelastic background
            self.worker.done_signal.connect(lambda: self.plot_viewer.show_inelastic_background_btn.setChecked(True))
            self.worker.done_signal.connect(lambda: self.plot_viewer.subtract_inelastic_background_btn.setChecked(False))
            self.worker.start()
            
    # Misc --------------------------------------------------------------------
            
    def center_window(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

def run():
    import sys
    
    app = QtGui.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(os.path.join(image_folder, 'eye.png')))
    gui = Iris()
    gui.showMaximized()
    
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    run()