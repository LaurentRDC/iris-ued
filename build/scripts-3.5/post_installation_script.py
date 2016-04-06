# -*- coding: utf-8 -*-
# This script should be run after the application has been installed.
# Mainly to create a shortcut to the application
import Iris

desktop = get_special_folder_path("CSIDL_DESKTOPDIRECTORY")
start_menu = get_special_folder_path("CSIDL_STARTMENU")

create_shortcut(PowderGuiApp.__file__, 'Iris', 'iris.lnk')