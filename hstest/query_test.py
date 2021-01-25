#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG

from hsdev import test_api
from hsgui import guitools
import multiprocessing

if __name__ == '__main__':
    # For windows
    multiprocessing.freeze_support()
    # Initialize a qt app (or get parent's)
    app, is_root = guitools.init_qtapp()
    # Create a HotSpotter API (hs) and GUI backend (back)
    hs, back = test_api.main(defaultdb='NAUTS', preload=False, app=app)
    # The test api returns a list of interesting chip indexes
    cx = test_api.get_test_cxs(hs, 1)[0]
    # Convert chip-index in to chip-id
    cid = hs.cx2_cid(cx)
    # Query the chip-id
    res = back.query(cid)
    # LAAEFTR: use res to do stuff
    # Run Qt Loop to use the GUI
    guitools.run_main_loop(app, is_root, back, frequency=20)
