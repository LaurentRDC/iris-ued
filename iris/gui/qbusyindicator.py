# -*- coding: utf-8 -*-
"""
    Modified from:
        Author: Jared P. Sutton <jpsutton@gmail.com>
        License: LGPL
        Note: I've licensed this code as LGPL because it was a complete translation of the code found here...
        https://github.com/mojocorp/QProgressIndicator
"""

from PyQt5 import QtWidgets, QtCore, QtGui


class QBusyIndicator(QtWidgets.QWidget):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.delay = property(int, self.animationDelay, self.setAnimationDelay)
        self.color = property(QtGui.QColor, self.getColor, self.setColor)

        self.m_angle = 0
        self.m_timerId = -1
        self.m_delay = 40
        self.m_displayedWhenStopped = True
        self.m_color = QtCore.Qt.white

        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.show()

    @QtCore.pyqtSlot(bool)
    def toggle_animation(self, busy):
        if busy:
            self.m_angle = 0

            if self.m_timerId == -1:
                self.m_timerId = self.startTimer(self.m_delay)
        else:
            if self.m_timerId != -1:
                self.killTimer(self.m_timerId)

            self.m_timerId = -1
            self.update()

    def sizeHint(self):
        return QtCore.QSize(20, 20)

    def animationDelay(self):
        return self.delay

    def isAnimated(self):
        return self.m_timerId != -1

    def getColor(self):
        return self.color

    def setAnimationDelay(self, delay):
        if self.m_timerId != -1:
            self.killTimer(self.m_timerId)

        self.m_delay = delay

        if self.m_timerId != -1:
            self.m_timerId = self.startTimer(self.m_delay)

    def setColor(self, color):
        self.m_color = color
        self.update()

    def timerEvent(self, event):
        self.m_angle = (self.m_angle + 30) % 360
        self.update()

    def paintEvent(self, event):

        width = min(self.width(), self.height())

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        outerRadius = (width - 1) * 0.5
        innerRadius = (width - 1) * 0.5 * 0.38

        capsuleHeight = outerRadius - innerRadius
        capsuleWidth = capsuleHeight * 0.23 if (width > 32) else capsuleHeight * 0.35
        capsuleRadius = capsuleWidth / 2

        for i in range(0, 12):
            color = QtGui.QColor(self.m_color)

            if self.isAnimated():
                color.setAlphaF(1.0 - (i / 12.0))
            else:
                color.setAlphaF(0.2)

            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(color)
            painter.save()
            painter.translate(self.rect().center())
            painter.rotate(self.m_angle - (i * 30.0))
            painter.drawRoundedRect(
                int(capsuleWidth * -0.5),
                int((innerRadius + capsuleHeight) * -1),
                int(capsuleWidth),
                int(capsuleHeight),
                capsuleRadius,
                capsuleRadius,
            )
            painter.restore()
