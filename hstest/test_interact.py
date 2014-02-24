'''
if __name__ == '__main__':
    from hscom import helpers
from hscom import helpers as util
    multiprocessing.freeze_support()
    import main
    hs = main.main()
    cx = helpers.get_arg('--cx', type_=int)
    qcx = hs.get_valid_cxs()[0]
    if cx is not None:
        qcx = cx

    res = hs.query(qcx)
    interact_chip(hs, qcx, fnum=1)
    interact_chipres(hs, res, fnum=2)
    df2.update()
    exec(df2.present())
'''
