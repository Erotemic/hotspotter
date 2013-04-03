from Facade import Facade

def run_query_test():
    'Test if you can run a simple query'
    data_fpath = 'D:/data/work/NAUT_Dan'
    fac = Facade(use_gui=False)
    fac.open_db(data_fpath)
    fac.selc(1)
    res = fac.query()

if __name__ == '__main__':
    run_query_test()
