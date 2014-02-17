from hotspotter import HotSpotterAPI as api
from hscom import argparse2
from hscom import helpers
from hsgui import guiback
from hsgui import guitools
from os.path import join
import multiprocessing


multiprocessing.freeze_support()
# Parse dummy args (TODO argparse needs to be optional to make a hsobject)
args = argparse2.parse_arguments()
args.db = None
args.dbdir = None
app, is_root = guitools.init_qtapp()
hs = api.HotSpotter(args)
back = guiback.make_main_window(hs, app)

# Build the test db name
work_dir = back.get_work_directory()
new_dbname = 'scripted_test_db'
new_dbdir = join(work_dir, new_dbname)

# Remove it if it exists
helpers.delete(new_dbdir)

back.new_database(new_dbdir)

back.import_images_from_file()

guitools.run_main_loop(app, is_root, back, frequency=100)
