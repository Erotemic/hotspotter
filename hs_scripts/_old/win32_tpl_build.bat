set HS_SCRIPTS=%code%\hotspotter\hs_scripts

:: Build using win32 script
cd %code%\flann
call build_flann_mingw.bat

cd %code%\opencv
call build_opencv_mingw.bat

cd %code%\hesaff
call build_hessaff_mingw.bat

:: Install opencv and flann
cd %code%\flann
make install
cd %code%\opencv
make install

color
cd %HS_SCRIPTS%
