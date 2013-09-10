set PATH=C:\Program Files (x86)\OpenCV\lib;C:\Program Files (x86)\OpenCV\bin;C:\Program Files (x86)\OpenCV\include\opencv;%PATH%
echo %PATH%

g++ ^
 -O2 ^
 -o main.exe main.cpp ^
 pkg-config opencv --cflags --libs

 REM -I "C:/Program Files (x86)/OpenCV/" ^
 REM -I "C:/Program Files (x86)/OpenCV/include" ^
 REM -I "C:/Program Files (x86)/OpenCV/include/opencv" ^
 REM -I "C:/Program Files (x86)/OpenCV/include/opencv2" ^
