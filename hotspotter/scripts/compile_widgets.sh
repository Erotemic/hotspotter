export gui_dir=$HOTSPOTTER/gui

rm $gui_dir/EditPrefSkel.py
rm $gui_dir/MainSkel.py
pyuic4 -x $gui_dir/EditPrefSkel.ui -o $gui_dir/EditPrefSkel.py
pyuic4 -x $gui_dir/MainSkel.ui -o $gui_dir/MainSkel.py
