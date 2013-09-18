set HS_SCRIPTS=%code%\hotspotter\hs_scripts
cd %HS_SCRIPTS%

set MINGW_BIN=C:\MinGW\bin
set INSTALL32=C:\Program Files (x86)
set CODE=%USERPROFILE%\code

set HS_TPL=%CODE%\hotspotter\tpl
set EXTERN_FEAT=%HS_TPL%\extern_feat

set HESAFF_BIN=%CODE%\hesaff\build
set OPENCV_BIN=%INSTALL32%\OpenCV\bin
::set FLANN_BIN=%INSTALL32%\Flann\bin


cd %EXTERN_FEAT%

:: MinGW Dependencies
REM for %%x in (libstdc++-6.dll, libgcc_s_dw2-1.dll) do ^
copy "%MINGW_BIN%\libstdc++-6.dll" "libstdc++-6.dll"
copy "%MINGW_BIN%\libgcc_s_dw2-1.dll" "libgcc_s_dw2-1.dll"


:: OpenCV Dependencies
REM for %%x in (libopencv_core249.dll, libopencv_highgui249.dll, libopencv_imgproc249.dll) do ^
copy "%OPENCV_BIN%\libopencv_core249.dll" "libopencv_core249.dll"
copy "%OPENCV_BIN%\libopencv_highgui249.dll" "libopencv_highgui249.dll"
copy "%OPENCV_BIN%\libopencv_imgproc249.dll" "libopencv_imgproc249.dll"

:: HessAff Executable 
copy %HESAFF_BIN%\hesaff.exe hesaff.exe

:: Download the others from featurespace.org
