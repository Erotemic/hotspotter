set HS_SCRIPTS=%code%\hotspotter\hs_scripts

set CODE=%USERPROFILE%\code
set HESAFF_SRC=%CODE%\hesaff
set OPENCV_SRC=%CODE%\opencv
set FLANN_SRC=%CODE%\flann


:: Update all from git
cd %FLANN_SRC%
git pull
cd %OPENCV_SRC%
git pull
cd %HESAFF_SRC%
git pull

:: Build using win32 script
cd %FLANN_SRC%
build_flann_mingw.bat
cd %OPENCV_SRC%
build_opencv_mingw.bat
cd %HESAFF_SRC%
build_hessaff_mingw.bat

:: Install opencv and flann
cd %FLANN_SRC%
make install
cd %OPENCV_SRC%
make install

color
cd %HS_SCRIPTS%
