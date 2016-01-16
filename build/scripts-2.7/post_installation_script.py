# -*- coding: utf-8 -*-
# This script should be run after the application has been instaled.
# Mainly to create a shortcut to the application
import PowderGuiApp

desktop = get_special_folder_path("CSIDL_DESKTOPDIRECTORY")
start_menu = get_special_folder_path("CSIDL_STARTMENU")

create_shortcut(PowderGuiApp.__file__, 'Powder Diffraction Processing', 'PowderGui.lnk')