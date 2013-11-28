cd %hs%\hs_scripts
set INSTALL32=C:\Program Files (x86)
set CODE=%USERPROFILE%\code

cd %HS%\tpl\extern_feat

:: MinGW Dependencies
REM for %%x in (libstdc++-6.dll, libgcc_s_dw2-1.dll) do ^
copy C:\MinGW\bin\libstdc++-6.dll "libstdc++-6.dll"
copy C:\MinGW\bin\libgcc_s_dw2-1.dll "libgcc_s_dw2-1.dll"


:: OpenCV Dependencies
REM for %%x in (libopencv_core249.dll, libopencv_highgui249.dll, libopencv_imgproc249.dll) do ^
copy "%INSTALL32%\OpenCV\bin\libopencv_core249.dll" "libopencv_core249.dll"
copy "%INSTALL32%\OpenCV\bin\libopencv_highgui249.dll" "libopencv_highgui249.dll"
copy "%INSTALL32%\OpenCV\bin\libopencv_imgproc249.dll" "libopencv_imgproc249.dll"

:: HessAff Executable 
copy %CODE%\hesaff\build\hesaff.exe hesaff.exe

:: Download the others from featurespace.org
cd %hs%\hs_scripts
