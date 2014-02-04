cd ~/code/hotspotter
# Check to make sure all dylibs have been otooled correctly
otool -L dist/HotSpotter.app/Contents/MacOS/hstpl/extern_feat/libhesaff.dylib
otool -L dist/HotSpotter.app/Contents/MacOS/libopencv_highgui.2.4.dylib
otool -L dist/HotSpotter.app/Contents/MacOS/libflann.dylib

# Check to make sure it runs
./dist/HotSpotter.app/Contents/MacOS/HotSpotterApp

cd dist 
open HotSpotter.app

