from __future__ import division, print_function
import DataStructures as ds
import dev
import match_chips3 as mc3
import numpy as np
import helpers
from Printable import DynStruct
import sys
import params
import helpers
import extract_patch as extract_patch
import pyflann

def interact1(hs, qon_list, fnum=1):
    print('this is a demo interactive session')
    cx, ocxs, notes = qon_list[0]
    start_interaction(hs, cx, notes)

def start_interaction(hs, cx, notes):
    chip_interaction(hs, cx, notes)
    while True:
      try:
        print('>>>')
        ans = raw_input('enter a interaction command (q to exit, h for help)\n>>>')
        print('>>>')
        ans = ans.split(' ')
        if len(ans) == 0: continue
        cmd = ans[0]
        if cmd == 'q': break;
        if cmd == 'n':
            fx_ptr[0] += 1
            cx+=1
            chip_interaction(hs, cx, '')
        if cmd == 'cid':
            cx = hs.cid2_cx(int(ans[1]))
            chip_interaction(hs, cx, '')
        print('>>>')
      except Exception as ex:
          print(repr(ex))
          raise


def chip_interaction(hs, cx, notes, fnum=1, **kwargs):
    chip_info_locals = dev.chip_info(hs, cx)
    chip_title = chip_info_locals['cidstr']+' '+chip_info_locals['name']
    chip_xlabel = chip_info_locals['gname']
    class State(DynStruct):
        def __init__(state):
            super(State, state).__init__()
            state.reset()
        def reset(state):
            state.res = None
            state.scale_min = None
            state.scale_max = None
            state.fnum = 1
            state.fnum_offset = 1

    state = State()
    state.fnum = fnum
    fx_ptr = [0]
    hprint = helpers.horiz_print
    kpts = hs.get_kpts(cx)
    scale = np.sqrt(kpts.T[2]*kpts.T[4])
    desc = hs.get_desc(cx)
    rchip = hs.get_chip(cx)
    # Start off keypoints with no filters
    is_valid = np.ones(len(kpts), dtype=bool)

    def update_valid(reset=False):
        print('[interact] updating valid')
        if reset is True:
            state.reset()
            is_valid[:] = True
        if state.scale_min:
            is_valid[:] = np.bitwise_and(scale >= state.scale_min, is_valid)
        if state.scale_max:
            is_valid[:] = np.bitwise_and(scale <= state.scale_max, is_valid)
        print(state)
        print('%d valid keypoints' % sum(is_valid))
        print('kpts scale ' + helpers.printable_mystats(scale[is_valid]))
        select_ith_keypoint(fx_ptr[0])

    def keypoint_info(fx):
        kp = kpts[fx]
        print(kp)
        x,y,a,c,d = kp
        A = np.array(([a,0],[c,d]))
        print('--kp info--')
        invA = np.linalg.inv(A)

    def select_ith_keypoint(fx):
        print('-------------------------------------------')
        print('[interact] viewing ith=%r keypoint' % fx)
        kp = kpts[fx]
        sift = desc[fx]
        np.set_printoptions(precision=5)
        df2.cla()
        fig1 = df2.figure(state.fnum, **kwargs)
        df2.imshow(rchip, pnum=(2,1,1))
        #df2.imshow(rchip, pnum=(1,2,1), title='inv(sqrtm(invE*)')
        #df2.imshow(rchip, pnum=(1,2,2), title='inv(A)')
        ell_args = {'ell_alpha':.4, 'ell_linewidth':1.8, 'rect':False}
        df2.draw_kpts2(kpts[is_valid], ell_color=df2.ORANGE, **ell_args)
        df2.draw_kpts2(kpts[fx:fx+1], ell_color=df2.BLUE, **ell_args)
        ax = df2.gca()
        #ax.set_title(str(fx)+' old=b(inv(sqrtm(invE*)) and new=o(A=invA)')
        scale = np.sqrt(kp[2]*kp[4])
        printops = np.get_printoptions()
        np.set_printoptions(precision=1)
        ax.set_title(chip_title)
        ax.set_xlabel(chip_xlabel)

        extract_patch.draw_keypoint_patch(rchip, kp, sift, pnum=(2,2,3))
        ax = df2.gca()
        ax.set_title('affine feature\nfx=%r scale=%.1f' % (fx, scale))
        extract_patch.draw_keypoint_patch(rchip, kp, sift, warped=True, pnum=(2,2,4))
        ax = df2.gca()
        ax.set_title('warped feature\ninvA=%r ' % str(kp))
        golden_wh = lambda x:map(int,map(round,(x*.618 , x*.312)))
        Ooo_50_50 = {'num_rc':(1,1), 'wh':golden_wh(1400*2)}
        np.set_printoptions(**printops)
        #df2.present(**Ooo_50_50)
        #df2.update()
        fig1.show()
        fig1.canvas.draw()
        #df2.show()

    fig = df2.figure(state.fnum )
    xy = kpts.T[0:2].T
    # Flann doesn't help here at all
    use_flann = False
    flann_ptr = [None]
    def on_click(event):
        if event.xdata is None: return
        print('[interact] button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        event.button, event.x, event.y, event.xdata, event.ydata))
        x,y = event.xdata, event.ydata
        if not use_flann:
            dist = (kpts.T[0] - x)**2 + (kpts.T[1] - y)**2
            fx_ptr[0] = dist.argsort()[0]
            select_ith_keypoint(fx_ptr[0])
        else:
            flann, = flann_ptr
            if flann is None:
                flann = pyflann.FLANN()
                flann.build_index(xy, algorithm='kdtree', trees=1)
            query = np.array(((x,y)))
            knnx, kdist = flann.nn_index(query, 1, checks=8)
            fx_ptr[0]=knnx[0]
            select_ith_keypoint(fx_ptr[0])
        print('>>>')
    callback_id = fig.canvas.mpl_connect('button_press_event', on_click)

    select_ith_keypoint(fx_ptr[0])
    query_cfg = ds.QueryConfig(hs, **kwargs)
    while True:
      try:
        print('>>>')
        raw = raw_input('enter a chip-interaction command (q to exit, h for help)\n>>>')
        print('>>>')
        ans = raw.split(' ')
        if len(ans) == 0: continue
        cmd = ans[0]
        if cmd in ['e', 'exit']: break;
        elif cmd == 'n':
            fx_ptr[0] += 1
            select_ith_keypoint(fx_ptr[0])
        elif cmd in ['q', 'query']:
            print(query_cfg)
            print(query_cfg.get_uid())
            res = hs.query(cx, query_cfg=query_cfg, use_cache=False)
            state.res = res
            resfnum = state.fnum + state.fnum_offset
            res.show_topN(hs, fnum=resfnum)
            df2.update()
            #fig_res = df2.figure(fnum=resfnum)
            #fig_res.show()
            #fig_res.canvas.draw()
        elif cmd == 'K':
            query_cfg.update_cfg(K=int(ans[1]))
        elif cmd == 'svoff':
            query_cfg.update_cfg(sv_on=False)
        elif cmd == 'svon':
            query_cfg.update_cfg(sv_on=True)
        elif cmd == 'test':
            query_cfg.update_cfg(sv_on=True, K=20, use_chip_extent=True)
        elif cmd in ['m', 'mytest']:
            mycmd = open('mytest.py').read();
            print(mycmd)
            exec mycmd in locals(), globals()
            print(query_cfg)
            res = hs.query(cx, query_cfg=query_cfg, use_cache=False)
            state.res = res
            resfnum = state.fnum + state.fnum_offset
            res.show_topN(hs, fnum=resfnum)
            df2.update()
        elif cmd == 'test2':
            query_cfg.update_cfg(sv_on=True, K=20, use_chip_extent=True, xy_thresh=.1)
            #query_cfg.update_cfg(sv_on=True, K=20, use_chip_extent=False)
        elif cmd == 'reset':
            update_valid(reset=True)
        elif cmd in ['fig']:
            state.fnum_offset += 1
        elif cmd in ['smin', 'scale_min']:
            state.scale_min = int(ans[1])
            update_valid()
        elif cmd in ['smax', 'scale_max']:
            state.scale_max = int(ans[1])
            update_valid()
        else:
            print('I dont understand the answer. I hope you know what you are doing');
            print(raw)
            exec raw in globals(), locals()
        print('>>>')
      except Exception as ex:
          print(repr(ex))
          if 'doraise' in vars():
            raise

if __name__ == '__main__':
    from multiprocessing import freeze_support
    import draw_func2 as df2
    freeze_support()
    print('[interact] __main__ ')
    main_locals = dev.dev_main()
    exec(helpers.execstr_dict(main_locals, 'main_locals'))
    fnum = 1
    interact1(hs, qon_list, fnum)
    #df2.update()
    exec(df2.present()) #**df2.OooScreen2()

