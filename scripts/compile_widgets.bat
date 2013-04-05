
set widget_dir=%HOTSPOTTER%\gui
::%~dp0

rm %widget_dir%\EditPrefSkel.py
rm %widget_dir%\MainSkel.py
call pyuic4 -x %widget_dir%\EditPrefSkel.ui -o %widget_dir%\EditPrefSkel.py
call pyuic4 -x %widget_dir%\MainSkel.ui -o %widget_dir%\MainSkel.py

:: actionOpen_Internal_Directory
::sed -i "s/matplotlibwidget/tpl.other.matplotlibwidget/g" MainWindowSkel.py
:: http://stackoverflow.com/questions/2398800/linking-a-qtdesigner-ui-file-to-python-pyqt

