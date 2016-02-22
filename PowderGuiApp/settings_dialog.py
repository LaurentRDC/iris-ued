# -*- coding: utf-8 -*-
import sys
import os
import configparser as ConfigParser
from pyqtgraph.Qt import QtCore, QtGui

settings_filename = os.path.join( os.path.dirname(__file__), 'settings.cfg' )

class SettingsDialog(QtGui.QDialog):
    
    def __init__(self):
        
        super(SettingsDialog, self).__init__()
                
        #Settings
        self.show_crosshair = True
        self.require_background = False
        
        #Initialize settings values with settings file
        self.loadSettingsFile()
        
        #UI stuff
        self.initUI()
        self.initLayout()
        self.connectSignals()
    
    def initUI(self):
        
        #Window property
        self.setWindowModality(True)
        
        #Default buttons
        self.accept_btn = QtGui.QPushButton('Save changes', parent = self)
        self.accept_btn.setDefault(True)
        self.reject_btn = QtGui.QPushButton('Cancel changes', parent = self)        
        
        #Settings
        self.show_crosshair_checkbox = QtGui.QCheckBox(text = 'Show crosshair on image viewer', parent = self)
        self.require_background_checkbox = QtGui.QCheckBox(text = 'Require background fit', parent = self)
    
    def initLayout(self):
        self.layout = QtGui.QVBoxLayout()
        
        checkboxes = QtGui.QVBoxLayout()
        checkboxes.addWidget(self.show_crosshair_checkbox)
        checkboxes.addWidget(self.require_background_checkbox)
        
        default_buttons = QtGui.QHBoxLayout()
        default_buttons.addWidget(self.accept_btn)
        default_buttons.addWidget(self.reject_btn)
        
        self.layout.addLayout(checkboxes)
        self.layout.addLayout(default_buttons)
        
        self.setLayout(self.layout)
    
    def connectSignals(self):
        self.accept_btn.clicked.connect(self.accept)
        self.reject_btn.clicked.connect(self.reject)
        
        self.accepted.connect(self.saveSettingsFile)        #This signal is called with self.accept() is called.
    
    def loadSettingsFile(self):
        
        parser = ConfigParser.SafeConfigParser()
        parser.read(settings_filename)
        
        # load settings
        self.show_crosshair = parser.getboolean('Settings', 'show_crosshair')
        self.require_background = parser.getboolean('Settings', 'require_background')
    
    def saveSettingsFile(self):
        
        parser = ConfigParser.SafeConfigParser()
        parser.add_section('Settings')
        parser.set('Settings', 'show_crosshair', str(self.show_crosshair))
        parser.set('Settings', 'require_background', str(self.require_background))
        
        #overwrite previous file
        with open(settings_filename, 'w') as configfile:
            parser.write(configfile)

if __name__ == '__main__':
    #test
    app = QtGui.QApplication(sys.argv)
    test = SettingsDialog()
    test.show()
    sys.exit(app.exec_())