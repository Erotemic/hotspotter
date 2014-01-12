from __init__ import *
import load_data2 as ld2
import match_chips2 as mc2
import report_results2 as rr2
import db_info
import params

if __name__ == '__main__':
    from multiprocessing import freeze_support
    import draw_func2 as df2
    import vizualizations as viz
    freeze_support()
    #db_dir = params.PZ_DanExt_Test
    #db_dir = params.PZ_DanExt_All
    #db_dir = params.Wildebeast
    db_dir = params.DEFAULT
    if not db_info.has_internal_tables(db_dir):
        print('initial load shows no tables. Creating them')
        from convert_db import init_database_from_images
        init_database_from_images(db_dir)
    hs = ld2.HotSpotter()
    hs.load_all(db_dir, matcher=False)
    #
    #info_locals = db_info.db_info(hs)
    #print info_locals['info_str']
    #
    #mc2.run_matching2(hs)
    viz.DUMP = True
    viz.BROWSE = False
    print('Dumping all queries')
    rr2.dump_all_queries2(hs)
    print('Finished dump')

