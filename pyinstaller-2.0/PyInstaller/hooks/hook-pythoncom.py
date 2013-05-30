# Copyright (C) 2005, Giovanni Bajo
# Based on previous work under copyright (c) 2001, 2002 McMillan Enterprises, Inc.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA

import PyInstaller.depend.modules

hiddenimports = ['win32com.server.policy']


def hook(mod):
    import sys
    newname = 'pythoncom%d%d' % sys.version_info[:2]
    if mod.typ == 'EXTENSION':
        mod.__name__ = newname
    else:
        import win32api
        h = win32api.LoadLibrary(newname + '.dll')
        pth = win32api.GetModuleFileName(h)
        #win32api.FreeLibrary(h)
        mod = PyInstaller.depend.modules.ExtensionModule(newname, pth)
    return mod
