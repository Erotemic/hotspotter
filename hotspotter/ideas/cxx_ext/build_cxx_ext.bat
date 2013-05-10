
mkdir build
cd build
cmake -G "MSYS Makefiles" -Wno-dev -DBOOST_ROOT:PATH=C:\boost_1_53_0 -DBOOST_LIBRARYDIR:PATH=C:\boost_1_53_0\stage\lib ..

mingw32-make

REM vs2010
REM cmake -G "NMake Makefiles" ..
 

cd ..
