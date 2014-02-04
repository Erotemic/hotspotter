cd ~/code/hotspotter
# Check to make sure all dylibs have been otooled correctly
otool -L dist/HotSpotter.app/Contents/MacOS/hstpl/extern_feat/libhesaff.dylib
otool -L dist/HotSpotter.app/Contents/MacOS/libopencv_highgui.2.4.dylib
otool -L dist/HotSpotter.app/Contents/MacOS/libflann.dylib

# Check to make sure it runs
./dist/HotSpotter.app/Contents/MacOS/HotSpotterApp


install_name_tool -change @loader_path/../../libopencv_videostab.2.4.dylib @loader_path/libopencv_videostab.2.4.dylib dist/HotSpotter.app/Contents/MacOS/hstpl/extern_feat/libhesaff.dylib

-add_rpath

cd dist 
open HotSpotter.app
