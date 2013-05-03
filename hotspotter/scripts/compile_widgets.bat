
set widget_dir=%HOTSPOTTER%\hotspotter\front
::%~dp0

set pyuic4=C:\Python27\Lib\site-packages\PyQt4\pyuic4

rm %widget_dir%\EditPrefSkel.py
rm %widget_dir%\MainSkel.py
call %pyuic4% -x %widget_dir%\ChangeNameDialog.ui -o %widget_dir%\ChangeNameDialog.py
call %pyuic4% -x %widget_dir%\ResultDialog.ui -o %widget_dir%\ResultDialog.py
call %pyuic4% -x %widget_dir%\EditPrefSkel.ui -o %widget_dir%\EditPrefSkel.py
call %pyuic4% -x %widget_dir%\MainSkel.ui -o %widget_dir%\MainSkel.py

:: actionOpen_Internal_Directory
::sed -i "s/matplotlibwidget/tpl.other.matplotlibwidget/g" MainWindowSkel.py
:: http://stackoverflow.com/questions/2398800/linking-a-qtdesigner-ui-file-to-python-pyqt

