'''
import multiprocessing
# Test load csv tables
if __name__ == '__main__':
    multiprocessing.freeze_support()
    from hsviz import draw_func2 as df2
    import params
    db_dir = params.DEFAULT
    hs_dirs, hs_tables, version = load_csv_tables(db_dir)
    exec(df2.present())
'''
