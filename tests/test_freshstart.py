import sys
from __future__ import print_function

def print(msg):
    import sys
    sys.stdout.write(msg+'\n')
    sys.stdout.flush()

DATA = 'D:/data' if sys.platform == 'win32' else '/data'
test_map = {
    'new_database_dir'   : DATA+'/test/new_database',\
    'new_hotspotter_dir' : DATA+'/test/hotspotter',\
    'branch'             : 'master',\
    #'REPO'               : 'git@hyrule.cs.rpi.edu:'
    'REPO'               : 'https://github.com:Erotemic/'
}
# Turn all map keys into variables for testing
for (key,val) in test_map.iteritems():
    exec('%s=%r' % (key,val))
database_dir = new_database_dir

def _remove_dir(dpath):
    import shutil
    import os.path
    print('Cleaning: '+dpath)
    if os.path.exists(dpath):
        shutil.rmtree(dpath)
        print('  * Was cleaned')
    else: 
        print('  * Was already clean')

def test_cleanup(new_hotspotter_dir=None, new_database_dir=None, **kwargs):
    'Removes directories from old tests'
    _remove_dir(new_hotspotter_dir)
    _remove_dir(new_database_dir)

def system_test():
    'Put the system through the ropes'
    test_cleanup(**test_map)
    checkout_hotspotter_test(**test_map)
    create_new_database_test(**test_map)
    quick_experiment_test(test_map['new_database_dir'])

def _run_with_progress(cmd):
    import subprocess
    import sys
    print('Executing: '+cmd)
    sys.stdout.flush()
    proc = subprocess.Popen(cmd.split(' '), shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        print(line)
        sys.stdout.flush()
    print("Execution Finished.")

def checkout_hotspotter_test(new_hotspotter_dir='hotspotter', branch='next', **kwargs):
    'Checkout hotspotter, setup, and move into it'
    import os.path
    setup_fpath = os.path.normpath(os.path.join(new_hotspotter_dir, 'setup.py'))
    setup_cmd = 'python '+setup_fpath+' fix_issues'
    clone_cmd = 'git clone %s:hotspotter.git --branch %s %s' %\
              (REPO, branch, new_hotspotter_dir)
    if os.path.exists(new_hotspotter_dir):
        raise Exception('New hotspotter dir already exists!')
    _run_with_progress(clone_cmd)
    os.chdir(new_hotspotter_dir)
    _run_with_progress(setup_cmd)
    return new_hotspotter_dir

def create_new_database_test(new_database_dir=None, **kwargs):
    'Creates a new database, adds images, and adds ROIs' 
    import os
    os.mkdir(new_database_dir)
    from hotspotter.HotSpotterAPI import HotSpotterAPI
    hs = HotSpotterAPI(new_database_dir, autoload=True)
    new_images_dir = DATA+'/test/test-images'
    new_image_list = [os.path.normpath(os.path.join(new_images_dir, newimg))\
                      for newimg in os.listdir(new_images_dir)]
    hs.add_image_list(new_image_list)
    hs.add_roi_to_all_images()
    hs.save_database()
    return hs.db_dpath

def quick_experiment_test(database_dir=None, **kwargs):
    'Runs a quick experiment, which tests many things.'
    from hotspotter.HotSpotterAPI import HotSpotterAPI
    hs = HotSpotterAPI(database_dir, autoload=True)
    hs.em.quick_experiment()

# HotSpotterAPI should have...
#import hotspotter
#hs = hotspotter.HotSpotterAPI(test_new_database)
#for gid in iter(hs.image_ids()):
    #hs.add_chip(gid=gid, roi='all')
#hs.remove_chip
#hs.change_name
#hs.change_roi
#hs.query(cid=4)

