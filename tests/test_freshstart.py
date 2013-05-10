import os

ROOT = '%USERPROFILE%' if sys.platform == 'win32' else '~'
test_database     = ROOT+'/data/work/testset'
test_new_database = ROOT+'/data/work/newdatabase'
test_hotspotter   = ROOT+'/tests/test-hotspotter'

testbranch = 'next'
# Checkout hotspotter and move into it
os.system('git clone git@hyrule.cs.rpi.edu:hotspotter.git '+test_hotspotter)
os.chdir(test_hotspotter)
if not testbranch ['master',None]:
    os.system('git checkout '+testbranch)

# Checkout 3rd party libs, and make sure exes work
os.system('python setup.py')

def test_existing():
    # Delete the current computed directory
    import hotspotter
    hs = hotspotter.HotSpotterAPI(test_dataset)
    hd.delete_computed_directory()
    del hs

    hs = hotspotter.HotSpotterAPI(test_dataset)
    hs.fac.selc(1)
    hs.fac.selg(1)
    hs.em.quick_experiment()

def test_new():
    import hotspotter
    hs = hotspotter.HotSpotterAPI(test_new_database)
    hs.add_image_list()
    image_dir = '/tests/test-images'
    hs.add_images_from_dir(image_dir)
    for gid in iter(hs.image_ids()):
        hs.add_chip(gid=gid, roi='all')
    #hs.remove_chip
    #hs.change_name
    #hs.change_roi
    #hs.query(cid=4)
