from __init__ import *
from experiments import get_db_names_info

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    db_dir = ld2.DEFAULT
    hs = ld2.HotSpotter()
    hs.load_all(db_dir)
    name_info = get_db_names_info(hs)
