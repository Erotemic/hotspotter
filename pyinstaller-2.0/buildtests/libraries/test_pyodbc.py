#
# Copyright (C) 2012, Martin Zibricky
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
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA


# pyodbc is a binary Python module. On Windows when installed with easy_install
# it is installed as zipped Python egg. This binary module is extracted
# to PYTHON_EGG_CACHE directory. PyInstaller should find the binary there and
# include it with frozen executable.


import pyodbc
