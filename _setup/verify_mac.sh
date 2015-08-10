cd ~/code/hotspotter
# Check to make sure all dylibs have been otooled correctly
#otool -L dist/HotSpotter.app/Contents/MacOS/libopencv_highgui.2.4.dylib
#otool -L dist/HotSpotter.app/Contents/MacOS/libflann.dylib


otool -L dist/HotSpotter.app/Contents/MacOS/hstpl/extern_feat/libhesaff.dylib

# Check to make sure it runs
./dist/HotSpotter.app/Contents/MacOS/HotSpotterApp

export HESDYLIB=~/code/hotspotter/dist/HotSpotter.app/Contents/MacOS/hstpl/extern_feat/libhesaff.dylib

install_name_tool -change @loader_path/../../libopencv_videostab.2.4.dylib @loader_path/libopencv_videostab.2.4.dylib $HESDYLIB

install_name_tool -change @loader_path/libopencv_videostab.2.4.dylib @loader_path/../libopencv_videostab.2.4.dylib $HESDYLIB

install_name_tool -change @loader_path/libopencv_videostab.2.4.dylib @loader_path/../libopencv_videostab.2.4.dylib $HESDYLIB


./_setup/fix_lib_otool.py check_depends $HESDYLIB

/opt/local/lib/libopencv_videostab.2.4.dylib


-add_rpath

cp dist/HotSpotter.dmg ~/Desktop/HotSpotter.dmg
#cd dist 
#open HotSpotter.app
