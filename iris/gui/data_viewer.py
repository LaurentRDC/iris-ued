# -*- coding: utf-8 -*-
"""
Viewer widgets for DiffractionDatasets
"""
from multiprocessing.sharedctypes import Value
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
import numpy as np

from .time_series_widget import TimeSeriesWidget


class ProcessedDataViewer(QtWidgets.QWidget):
    """
    Widget displaying DiffractionDatasets
    """

    timeseries_rect_signal = QtCore.pyqtSignal(tuple)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_viewer = pg.ImageView(parent=self)
        self.image_viewer.setPredefinedGradient(
            "inferno"
        )  # to distinguish between raw and processed

        self.time_series_widget = TimeSeriesWidget(parent=self)

        self.cursor_info_widget = QtWidgets.QLabel(parent=self)
        self.cursor_info_widget.setAlignment(QtCore.Qt.AlignHCenter)

        self.timeseries_rect_region = pg.ROI(
            pos=[0, 0], size=[200, 200], pen=pg.mkPen("r")
        )
        self.timeseries_rect_region.addScaleHandle([1, 1], [0, 0])
        self.timeseries_rect_region.addScaleHandle([0, 0], [1, 1])

        self.roi_topleft_text = pg.TextItem("", anchor=(1, 1))
        self.roi_bottomright_text = pg.TextItem("", anchor=(0, 0))
        self.image_viewer.addItem(self.roi_topleft_text)
        self.image_viewer.addItem(self.roi_bottomright_text)

        # Signal proxies allow to rate-limit them
        self.__cursor_proxy = pg.SignalProxy(
            self.image_viewer.scene.sigMouseMoved,
            rateLimit=60,
            slot=self.update_cursor_info,
        )
        self.__dynamics_roi_proxy = pg.SignalProxy(
            self.timeseries_rect_region.sigRegionChanged,
            rateLimit=30,
            slot=self.update_timeseries_rect,
        )

        self.__bragg_peak_items = list()
        self.__bz_items = list()

        self.image_viewer.getView().addItem(self.timeseries_rect_region)
        self.timeseries_rect_region.hide()
        self.time_series_widget.hide()

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.image_viewer)
        self.layout.addWidget(self.cursor_info_widget)
        self.layout.addWidget(self.time_series_widget)
        self.setLayout(self.layout)

    @QtCore.pyqtSlot()
    def update_timeseries_rect(self):
        rect = self.timeseries_rect_region.parentBounds().toRect()

        # Add text to show rect bounds
        # If coordinate is negative, return 0
        # Note that in case pg.getConfigOption('imageAxisOrder') is not row-major,
        # The following bounds will not work
        x1 = round(max(0, rect.topLeft().y()))
        x2 = round(max(0, rect.y() + rect.height()))
        y1 = round(max(0, rect.topLeft().x()))
        y2 = round(max(0, rect.x() + rect.width()))

        self.timeseries_rect_signal.emit((x1, x2, y1, y2))

    @QtCore.pyqtSlot(object)
    def update_cursor_info(self, event):
        """ Provide information about the cursor (position, value, etc) """
        mouse_point = self.image_viewer.getView().mapSceneToView(event[0])
        i, j = int(mouse_point.x()), int(mouse_point.y())
        try:
            val = self.image_viewer.getImageItem().image[j, i]
        except IndexError:
            val = 0
        self.cursor_info_widget.setText(
            f"Position: ({i},{j}) | Pixel value: {val:.2f} cnts"
        )

    @QtCore.pyqtSlot(bool)
    def toggle_peak_dynamics(self, toggle):
        """ Toggle interactive peak dynamics region-of-interest"""
        if toggle:
            self.timeseries_rect_region.show()
        else:
            self.timeseries_rect_region.hide()
            self.roi_topleft_text.setText("")
            self.roi_bottomright_text.setText("")

        self.time_series_widget.setVisible(toggle)

    @QtCore.pyqtSlot(bool)
    def toggle_roi_bounds_text(self, enable):
        """ Toggle showing array indices around the peak dynamics region-of-interest """
        if enable:
            self.timeseries_rect_signal.connect(self._update_roi_bounds_text)
            self.update_timeseries_rect()
        else:
            self.roi_topleft_text.setText("")
            self.roi_bottomright_text.setText("")
            self.timeseries_rect_signal.disconnect(self._update_roi_bounds_text)

    @QtCore.pyqtSlot(tuple)
    def _update_roi_bounds_text(self, rect):
        """ Update the ROI bounds text based on the bounds in ``rect`` """
        x1, x2, y1, y2 = rect
        self.roi_topleft_text.setPos(y1, x1)
        self.roi_topleft_text.setText(f"({y1},{x1})")

        self.roi_bottomright_text.setPos(y2, x2)
        self.roi_bottomright_text.setText(f"({y2},{x2})")

    @QtCore.pyqtSlot(object, bool)
    def display(self, image, autocontrast=False):
        """
        Display an image in the form of an ndarray.

        Parameters
        ----------
        image : ndarray or None
            If None, the display is cleared.
        autocontrast: bool, optional
            If True, the image contrast will be adjusted
        """
        if image is None:
            self.image_viewer.clear()
            return

        self.image_viewer.setImage(
            image, autoLevels=autocontrast, autoRange=autocontrast
        )
        self.update_timeseries_rect()

    @QtCore.pyqtSlot(object, object)
    def display_peak_dynamics(self, times, intensities):
        """
        Display the time series associated with the integral between the bounds
        of the ROI
        """
        self.time_series_widget.plot(times, intensities)

    @QtCore.pyqtSlot(dict)
    def display_bragg_peaks(self, params):
        """
        Highlight the location of Bragg peaks.

        Parameters
        ----------
        peaks : list of 2-tuple
            List of tuples [row, col] where Bragg peaks are located.
        """
        enabled_peaks = params["enable_peaks"]
        peaks = params["peaks"]
        enabled_bzs = params['enable_bz']
        voronoi_regions = params['bz']
        n_vertices = params['n_vertices']
        self.bbox_size = int(0.025*self.image_viewer.getImageItem().image.shape[0])
        for item in self.__bragg_peak_items:
            self.image_viewer.removeItem(item)
        self.__bragg_peak_items.clear()
        if enabled_peaks:
            for idx, peak in enumerate(peaks):
                r, c = peak
                generator = generate_connections(4)
                nodes = list()
                for _ in range(4):
                    nodes.append(next(generator))
                nodes = np.array(nodes).reshape(-1,2)
                self.__bragg_peak_items.append(
                    # pg.RectROI(
                    #     pos=(c-25, r-25), size=(50,50), movable=False, resizable=False, removable=False
                    # )
                    pg.GraphItem(
                        pos = np.array(
                            [
                                [c-self.bbox_size//2, r-self.bbox_size//2],
                                [c-self.bbox_size//2, r+self.bbox_size//2],
                                [c+self.bbox_size//2, r+self.bbox_size//2],
                                [c+self.bbox_size//2,r-self.bbox_size//2]
                            ]
                        ),
                        adj = nodes,
                        pen=pg.mkPen("r"),
                        size=0
                    )
                )
            for item in self.__bragg_peak_items:
                self.image_viewer.addItem(item)

        for item in self.__bz_items:
            self.image_viewer.removeItem(item)
        self.__bz_items.clear()
        if enabled_bzs:
            for r in voronoi_regions:
                if r.is_visible:
                    generator = generate_connections(n_vertices)
                    N = np.array(r.vertices).reshape(-1, 2).shape[0]
                    nodes = list()
                    for _ in range(N):
                        nodes.append(next(generator))
                    nodes = np.array(nodes).reshape(-1,2)
                    self.__bz_items.append(
                        pg.GraphItem(
                            pos = np.array(r.vertices).reshape(-1, 2),
                            adj = nodes
                        )
                    )
            for item in self.__bz_items:
                self.image_viewer.addItem(item)

def generate_connections(sym):
    """
    Generator of the connections between points to graph closed polygons in pyqtgraph
    """
    num = 0
    while True:
        yield (num, (num+1)%(sym))
        num += 1
    
