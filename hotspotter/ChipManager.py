import types
import re
import numpy as np
import os.path
from hotspotter.other.AbstractPrintable import AbstractDataManager
from hotspotter.other.helpers   import filecheck
from hotspotter.other.logger    import logmsg, logdbg, logerr, logio, logwarn
from pylab           import find
from PIL             import Image 

# Chip Manager handle the chips
# this entails managing:
#   chips directory
#   image chips
#   feature representation
class ChipManager(AbstractDataManager):
    '''The chip manager maintains chip information including
    feature representations, name, image, extent, and property
    information.'''

    # --- CID CONVINENCE FUNCTIONS ---
    def is_valid(cm, cid):
        #if np.iterable(cid):
            #return all([cm.is_valid(cid_) for cid_ in cid])
        return cid < len(cm.cid2_cx) and cid >= 0 and cm.cid2_cx[cid] > 0
    def iscx_valid(cm, cx):
       return cx < len(cm.cx2_cid) and cx >= 0 and cm.cx2_cid[cx] > 0
    def cid(cm, cx):
        'maps cx to cid with error checks'
        if not cm.iscx_valid(cx):
            logerr('CX=%s is invalid' % str(cx))
        return cm.cx2_cid[cx]
    def cx(cm, cid):
        'maps cid to cx with error checks'
        if not cm.is_valid(cid):
            logerr('CID=%s is invalid' % str(cid))
        return cm.cid2_cx[cid]
    def gid(cm, cid):
        return cm.cx2_gid(cm.cx(cid))
    def info(cm, cid_list=None, lbls=None):
        return cm.cx2_info(cm.cx(cid_list), lbls)
    # TODO: info and cx2_ and cx2_dynget should merge. A standard should be chosen
    def cx2_dynget(cm, cx_list, *dynargs):
        return cm.cx2_(cx_list, *dynargs)

    def cx2_(cm, cx_list, *dynargs):
        'request chip data'
        'conviencience function to get many properties'
        #logdbg('Requested Data: %s of CX= %s' % (str(dynargs), str(cx_list)))
        to_return = []
        cid = cm.cx2_cid[cx_list]
        invalid_x = find(cid == 0)
        if len(invalid_x) > 0:
            logerr('Requested invalid cxs: '+str(cx_list[invalid_x]))
        for arg in dynargs: 
            if arg == 'cx':
                to_return.append(cx_list)
            elif arg == 'cid':
                to_return.append(cm.cx2_cid[cx_list])
            elif arg == 'nid':
                to_return.append(cm.cx2_nid(cx_list))
            elif arg == 'gid':
                to_return.append(cm.cx2_gid(cx_list))
            elif arg == 'chip':
                to_return.append(cm.cx2_chip(cx_list))
            elif arg == 'name':
                to_return.append(cm.cx2_name(cx_list))
            elif arg == 'gname':
                to_return.append(cm.cx2_gname(cx_list))
            else:
                to_return.append('__UNFILLED__') # mark unfilled requests
        return to_return
    def cid2_(cm, cid_list, *dynargs):
        'convienence for cids instead of cxs'
        cx_list = cm.cx(cid_list)
        return cm.cx2_(cx_list, *dynargs)

    # --- ACTUAL WORK FUNCTIONS
    def __init__(cm,hs):
        super( ChipManager, cm ).__init__( hs )
        # --- Table Info ---
        cm.cx2_cid       = np.empty(0, dtype=np.uint32) # index to Chip id
        cm.cx2_nx        = np.empty(0, dtype=np.uint32) # index to Name id
        cm.cx2_gx        = np.empty(0, dtype=np.uint32) # index to imaGe id
        cm.cx2_roi       = np.empty((0,4), dtype=object) #  (x,y,w,h)
        cm.cx2_theta     = np.empty(0, dtype=np.float32) # roi orientation
        # --- Feature Representation of Chip ---
        cm.cx2_fpts      = np.empty(0, dtype=object) # heshes keypoints
        cm.cx2_fdsc      = np.empty(0, dtype=object) # Root SIFT fdscriptors
        cm.cx2_transChip = np.empty(0, dtype=object)
        cm.cx2_dirty_bit = np.empty(0, dtype=np.bool) # Dirty bit flag (need to recompute)
        # --- Reverse Index --
        cm.cid2_cx       = np.array([], dtype=np.uint32)
        # --- Book Keeping --
        cm.next_cx   =  1 # the next indeX we are going to use
        cm.next_cid  =  1 # the next ID we are going to use
        cm.num_c     =  0 # number of chips.
        cm.max_cx    =  0 # the largest cx seen
        cm.max_cid   =  0 # the largest cid seen
        cm.max_roi   = [0,0,0,0]
               
        cm.x2_lbl = \
        {
            'cid'  : lambda _: cm.cx2_cid[_],\
            'nid'  : lambda _: cm.cx2_nid(_),\
            'gid'  : lambda _: cm.cx2_gid(_),\
            'gname': lambda _: cm.cx2_gname(_),\
            'name' : lambda _: cm.cx2_name(_),\

            'roi'  : lambda _: cm.cx2_roi[_],\
            'theta': lambda _: cm.cx2_theta[_],\
            'cx'   : lambda _: _ ,\
            'nx'   : lambda _: cm.cx2_nx[_] ,\
            'gx'   : lambda _: cm.cx2_gx[_] ,\
        }
        cm.default_fields = ['cid','gid','nid','roi','theta']

    def load_csv_line(cm, field_values, headers):
        field_values = map(lambda k: k.strip(' '), field_values)
        if headers is None: headers = cm.default_fields
        if len(headers) != len(field_values):
            logwarn('In chip_file. len(headers) != len(field_values) length mismatch\n'+\
                    str(headers)+'\n'+str(field_values))
        # Build field name -> field value map
        dmap = {k:v for (k,v) in zip(headers,field_values)}
        if cm.hs.core_prefs.legacy_bit:
            # Legacy: Be Backwards Compatible
            if 'imgindex' in dmap.keys():
                logwarn('Found imgindex')
                imgindex = int(dmap['imgindex'])
                gname = 'img-%07d.jpg' % imgindex
                cm.hs.gm.add_img(int(imgindex), gname, False)
                dmap['gid'] = imgindex
                dmap['cid'] = imgindex
                del dmap['imgindex']
            if 'animal_name' in dmap.keys():
                logwarn('Found animal_name')
                dmap['nid'] = cm.hs.nm.add_name(-1, dmap['animal_name'])
                del dmap['animal_name']
            if 'instance_id' in dmap.keys():
                dmap['cid'] = dmap['instance_id']
                del dmap['instance_id']
            if 'image_id' in dmap.keys():
                dmap['gid'] = dmap['image_id']
                del dmap['image_id']
            if 'name_id' in dmap.keys():
                dmap['nid'] = dmap['name_id']
                del dmap['name_id']

        cid = int(dmap['cid'])
        gid = int(dmap['gid'])
        nid = int(dmap['nid'])
        try:
            theta = np.float32(dmap['theta'])
        except KeyError as ex:
            theta = 0
        roi_str = re.sub('  *',' ', dmap['roi'].replace(']','').replace('[','')).strip(' ').rstrip()
        roi = map(lambda x: int(round(float(x))),roi_str.split(' '))
        nx  = cm.hs.nm.nid2_nx[nid]
        gx  = cm.hs.gm.gid2_gx[gid]
        if gx == 0 or nx == 0 or gid == 0 or nid == 0:
            logmsg('Adding Chip: (cid=%d),(nid=%d,nx=%d),(gid=%d,gx=%d)' % (cid, nid, nx, gid, gx))
            logerr('Chip has invalid indexes')
        cm.add_chip(cid, nx, gx, roi, theta, delete_prev=False)

    def get_csv_line(headers):
        cm.cx2_info(lbls=['cid','gid','nid','roi','theta'])
        pass

    def  chip_alloc(cm, nAlloc):
        logdbg('Allocating room for %d more chips' % nAlloc)
        cm.cx2_cid       = np.append(cm.cx2_cid, np.zeros(nAlloc,dtype=np.uint32))
        # Implicit Data Local Identifiers
        cm.cx2_nx        = np.append(cm.cx2_nx,  np.zeros(nAlloc,dtype=np.uint32))
        cm.cx2_gx        = np.append(cm.cx2_gx,  np.zeros(nAlloc,dtype=np.uint32)) 
        # Explicit Data
        cm.cx2_roi       = np.append(cm.cx2_roi,  np.zeros((nAlloc,4),dtype=np.uint32), axis=0)
        cm.cx2_theta     = np.append(cm.cx2_theta, np.zeros(nAlloc,dtype=np.float32), axis=0)
        cm.cx2_fpts      = np.append(cm.cx2_fpts, np.empty(nAlloc,dtype=object))
        cm.cx2_fdsc      = np.append(cm.cx2_fdsc, np.empty(nAlloc,dtype=object))
        # Feature Representation
        cm.cx2_dirty_bit = np.append(cm.cx2_dirty_bit, np.ones(nAlloc,dtype=np.bool))
        cm.cx2_transChip = np.append(cm.cx2_transChip, np.zeros(nAlloc,dtype=object))
        # Reverse Index
        idAlloc = len(cm.cid2_cx) - len(cm.cx2_cid)
        if idAlloc > 0:
            cm.cid2_cx = np.append(cm.cid2_cx, np.zeros(idAlloc,dtype=np.uint32))

    def cx2_info(cm, cxs=None, lbls=None):
        #returns info in formatted table
        if cxs is None: cxs = cm.get_valid_cxs()
        if lbls is None: lbls = cm.default_fields
        data_table_str = cm.x2_info(cxs, lbls)
        return '# ChipManager\n'+data_table_str

    # More convinence functions
    def get_valid_cxs(cm): 
        return find(cm.cx2_cid > 0)
    def get_invalid_cxs(cm):
        return find(cm.cx2_cid == 0)
    def invalid_cxs(cm):
        'depricated'
        return cm.get_invalid_cxs() 
    def cx2_num_other_chips(cm, cxs):
        'returns the number of other.hips beloning to the same name'
        return np.array(map(lambda x: len(x), cm.cx2_other_cxs(cxs)),dtype=np.uint32)
    def cx2_name(cm, cxs):
        nxs  = cm.cx2_nx[cxs]
        return cm.hs.nm.nx2_name[nxs]
    def cx2_gname(cm, cxs):
        gxs    = cm.cx2_gx[cxs]
        return cm.hs.gm.gx2_gname[gxs]
    def cx2_img(cm, cx):
        gx    = cm.cx2_gx[cx]
        return cm.hs.gm.gx2_img(gx)
    def cx2_img_list(cm, cx_list):
        gx_list = cm.cx2_gx[cx_list]
        return cm.hs.gm.gx2_img_list(gx_list)
    def cx2_nid(cm,cxs):
        nxs  = cm.cx2_nx[cxs]
        return cm.hs.nm.nx2_nid[nxs]
    def cx2_gid(cm,cxs):
        gxs  = cm.cx2_gx[cxs]
        return cm.hs.gm.gx2_gid[gxs]
    def cid2_nid(cm,cids):
        cxs  = cm.cid2_cx(cids)
        return cm.cx2_nid[cxs]
    def cx2_other_cxs(cm,cx_list):
        nm = cm.hs.nm
        nx_list = cm.cx2_nx[cx_list]
        other_cxs = nm.nx2_cx_list[nx_list]
        UNIDEN_NX = 1
        return [ocx if nx != UNIDEN_NX else [] for (nx,ocx) in zip(nx_list, other_cxs)]
        #return [nm.nx2_cx_list[nx] for nx in nx_list]
    def all_cxs(cm):
        return np.array(find(cm.cx2_cid > 0), dtype=np.uint32)
    def cid2_valid_bit(cm,cids): # Tests if CID is managed.
        if type(cids) is types.ListType:
            # Check InverseIndex Overflow
            valid_bit = np.array([id <= cm.max_cid for id in cids]) 
            valid_cxs = [cm.cid2_cx[cid] for cid in cids[valid_bit]] 
            valid_bit[valid_bit] = [cx > 0 for cx in valid_cxs]
        else: #Non-List Case
            valid_bit = cm.max_cid > cids and cm.cid2_cx[cids] > 0
        return valid_bit
    # --- ACTUAL WORK FUNCTIONS
    def add_chip(cm, cid, nx, gx, roi, theta, delete_prev=False):
        nm = cm.hs.nm
        gm = cm.hs.gm
        # Fails if cid is not available; cid = -1 means pick for you
        cx = -1
        if cid < 0:
            cid = cm.next_cid
        else:
            if cm.cid2_valid_bit(cid): #New CID must be invalid
                logerr('CID Already in database Chip Not Added')
                logerr('Offending String: (cid, nx, gx, [roi]) = (%d, %d, %d, %s)' % (cid, nx, gx, str(roi)))
                cid = 0
                return
        
        #Manage Memory
        cx = cm.next_cx
        logdbg('''
        Adding Chip =
               (  cid,   nx,   gx, [tl_x   tl_y      w      h ])
               ( %4d, %4d, %4d, %s)
               '''% (cid, nx, gx, str('[ %4.1f  %4.1f  %4.1f  %4.1f ]' % tuple(roi))))

        if cx >= len(cm.cx2_cid):
            curr_alloc = len(cm.cx2_cid)
            cm.chip_alloc((curr_alloc+1)*2+1)
        
        # Add the information to the flat table
        logdbg(' * Adding cx='+str(cx)+' to the tables')
        if nx == 0 or gx == 0 or len(roi) != 4:
            logerr('Chip information is invalid. Cannot add.') 
        if delete_prev:
            cm.delete_computed_cid(cid)
        cm.cx2_cid[cx]  = cid
        cm.cx2_nx [cx]  = nx
        cm.cx2_gx [cx]  = gx
        cm.cx2_roi[cx]  = roi
        cm.cx2_theta[cx]  = theta
        cm.max_roi      = map(lambda (a,b): max(a,b), zip(cm.max_roi, roi))
        # Add This Chip To Reverse Indexing
        if cid >= len(cm.cid2_cx):
            idAlloc = max(cid+1,len(cm.cid2_cx)*2 + 1)
            logdbg('Allocating: '+str(idAlloc)+' more cids')
            cm.cid2_cx = np.append(cm.cid2_cx, np.zeros(idAlloc,dtype=np.uint32))
        cm.cid2_cx[cid] = cx
        nm.nx2_cx_list[nx].append(cx)
        gm.gx2_cx_list[gx].append(cx)
        # Increment
        cm.next_cx  = max(cm.next_cx + 1, cx+1)
        cm.next_cid = max(cm.next_cid+1, cid+1)
        cm.max_cx   = max(cm.max_cx,     cx)
        cm.max_cid  = max(cm.max_cid,    cid)
        cm.num_c    = cm.num_c + 1
        cm.hs.vm.isDirty = True
        return cid

    def delete_computed_cid(cm, cid):
        iom = cm.hs.iom
        if np.iterable(cid): logerr('this function only works for a single cid')
        logmsg('Removing CID=%d\'s computed files' % cid)
        cid_fname_pattern = iom.get_chip_prefix(cid, [])+'*'
        iom.remove_computed_files_with_pattern(cid_fname_pattern)
            
    def remove_chip(cm, cx):
        cx_list = [cx]
        if type(cx) == types.ListType:
            cx_list = cx
        logdbg('Removing CXs '+str(cx_list))
        cm.hs.on_cx_modified(cx)
        for cx in cx_list:
            cid = cm.cx2_cid[cx]
            logmsg('Removing cid=%d' % cid)
            #Remove cx from other.data managers
            gx = cm.cx2_gx[cx]
            nx = cm.cx2_nx[cx]
            cm.hs.gm.gx2_cx_list[gx].remove(cx)
            cm.hs.nm.nx2_cx_list[nx].remove(cx)
            #Remove from search manager
            cm.hs.vm.index_dirty_bit = True
            cm.hs.vm.train_cid = np.setdiff1d(cm.hs.vm.train_cid, cid)
            #Remove data saved on disk and memory
            cm.unload_features(cx)
            cm.delete_computed_cid(cid)
            #Remove data saved in memory            
            cm.cx2_cid[cx]   = 0
            cm.cx2_nx[cx]    = 0
            cm.cx2_gx[cx]    = 0
            cm.cx2_roi[cx]   = np.array([0,0,0,0],dtype=np.uint32)
            cm.cx2_theta[cx] = 0
            cm.cid2_cx[cid]  = 0
    
    def change_orientation(cm, cx, new_theta):
        cid = cm.cx2_cid[cx]
        logmsg('Giving cid=%d new theta: %r' % (cid, new_theta))
        assert not new_theta is None
        cm.cx2_dirty_bit[cx] = True
        cm.unload_features(cx)
        cm.delete_computed_cid(cid)
        cm.cx2_theta[cx] = new_theta
        cm.hs.vm.isDirty = True # Mark vocab as dirty


    def change_roi(cm, cx, new_roi):
        cid = cm.cx2_cid[cx]
        logmsg('Giving cid=%d new roi: %r' % (cid, new_roi))
        assert not new_roi is None
        if new_roi is None:
            logerr('The ROI is np.empty')
        cm.cx2_dirty_bit[cx] = True
        cm.unload_features(cx)
        cm.delete_computed_cid(cid)
        cm.cx2_roi[cx] = new_roi
        cm.hs.vm.isDirty = True # Mark vocab as dirty
    
    def rename_chip(cm, cx, new_name):
        nm = cm.hs.nm
        cid     = cm.cid(cx)
        old_nx  = cm.cx2_nx[cx]
        old_name = nm.nx2_name[old_nx]
        logmsg('Renaming cid='+str(cid)+' from '+str(old_name)+' to '+new_name)
        if not new_name in nm.name2_nx.keys():
            nm.add_name(-1,new_name)
        old_nx  = cm.cx2_nx[cx]
        new_nx  = nm.name2_nx[new_name]
        #Debug
        old_nid = nm.nx2_nid[old_nx]
        new_nid = nm.nx2_nid[new_nx]
        logdbg('Old Name Info: cid=%d cx=%d,  nid=%d, nx=%d, name=%s' % (cid, cx, old_nid, old_nx, old_name))
        logdbg('New Name Info: cid=%d cx=%d,  nid=%d, nx=%d, name=%s' % (cid, cx, new_nid, new_nx, new_name))
        #EndDebug
        nm.nx2_cx_list[old_nx].remove(cx)
        nm.nx2_cx_list[new_nx].append(cx)
        cm.cx2_nx[cx] = new_nx

# --- Raw Image Representation of Chip ---
    def cx2_chip_list(cm, cx_list):
        if np.iterable(cx_list): 
            chip_fpath_list = [cm.cx2_chip_fpath(cx) for cx in iter(cx_list) ]
            return [np.asarray(Image.open(chip_fpath)) for chip_fpath in iter(chip_fpath_list)]
        else: 
            return [cm.cx2_chip(cx_list)]

    def cx2_chip(cm, cx):
        chip_fpath = cm.cx2_chip_fpath(cx)
        # Load chip and rotate it
        return np.asarray(
            Image.open(chip_fpath).rotate(
                cm.cx2_theta[cx]*180/np.pi, resample=Image.BICUBIC, expand=1))

    def cx2_chip_size(cm, cx):
        chip_fpath = cm.cx2_chip_fpath(cx)
        return Image.open(chip_fpath).size

    def cx2_transChip(cm, cx):
        return np.linalg.inv(cm.cx2_transImg(cx))

#------------------------------
# Steps to transform a detection from Chip Space to Image Space
#  (Chip Space): roi=[0, 0, cw, ch] 
#  * translate: -[cw, ch]/2
#  * rotate: -theta
#  * translate: [ucw, uch]/2
#  (Unoriented Chip Space) = roi=[0,0,ucw,ucw] 
#  * scale: scale_factor
#  * translate: rx, ry
#  (Image Space): roi=[rx,ry,rw,rh]
#------------------------------

    def cx2_transImg(cm, cx):
        (cw, ch) = cm.cx2_chip_size(cx)
        (rx, ry, rw, rh) = cm.cx2_roi[cx]
        theta = cm.cx2_theta[cx]
        sx = float(rw) / float(cw)
        sy = float(rh) / float(ch)
        tx = float(rx) # Translation happens after scaling
        ty = float(ry)
        #rot = np.array(([1, 0, 0],
                        #[0, 1, 0],
                        #[0, 0, 1]), dtype=np.float32)
        # Return Affine Transformation 
        scale = np.array(([sx,  0,  0],
                          [ 0, sy,  0],
                          [ 0,  0,  1]), dtype=np.float32)

        rot_trans_pre = np.array(([ 1,  0, cw/2],
                                  [ 0,  1, ch/2],
                                  [ 0,  0,    1]), dtype=np.float32)

        rot = np.array(([np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta),  np.cos(theta), 0],
                        [             0,             0, 1]), dtype=np.float32)

        rot_trans_post = np.array(([ 1,  0, -cw/2],
                                   [ 0,  1, -ch/2],
                                   [ 0,  0,     1]), dtype=np.float32)

        trans = np.array(([ 1,  0, tx],
                          [ 0,  1, ty],
                          [ 0,  0,  1]), dtype=np.float32)

        return trans.dot(rot_trans_pre.dot(rot.dot(rot_trans_post.dot(scale))))

    def cx2_chip_fpath(cm, cx):
        'Gets chip fpath with checks'
        iom = cm.hs.iom
        cid = cm.cid(cx)
        chip_fpath  = iom.get_chip_fpath(cid)
        if not filecheck(chip_fpath): 
            cm.compute_chip(cx)
        return chip_fpath
    
# --- Feature Representation Methods ---
    def  get_feats(cm, cx, force_recomp=False):
        # FIXME: If the algorithm changes, the dirty bit is not flipped
        if force_recomp or\
           cm.cx2_fpts[cx] is None or\
           cm.cx2_fdsc[cx] is None or\
           np.sum(cm.cx2_dirty_bit[cx]):
            cm.load_features(cx, force_recomp)
        return (cm.cx2_fpts[cx], cm.cx2_fdsc[cx])

    def  get_fpts(cm, cx, force_recomp=False):
        if force_recomp or cm.cx2_fpts[cx] is None or np.sum(cm.cx2_dirty_bit[cx]):
            cm.load_features(cx, force_recomp)
        return cm.cx2_fpts[cx]

    def  get_fdsc(cm, cx, force_recomp=False):
        if force_recomp or cm.cx2_fdsc[cx] is None or np.sum(cm.cx2_dirty_bit[cx]):
            cm.load_features(cx, force_recomp)
        return cm.cx2_fdsc[cx]
    
    def  cx2_nfpts(cm, cxs=None):
        if cxs == None:
            cxs = cm.all_cxs()
        if type(cxs) in [np.uint32, types.IntType]:
            cxs = np.array([cxs],dtype=np.uint32)
        return np.array([cm.cx2_fpts[cx].shape[0] for cx in cxs], dtype=np.uint32)
    
    
    # --- Internals  ---
    def _scaled_size(cm, cx, dtype=float):
        '''Returns the scaled size of cx. Without considering rotation
           Depends on the current algorithm settings
           dtype specifies the percision of return type'''
        # Get raw size and target sizze
        (_, __, rw, rh)  = cm.cx2_roi[cx]
        target_diag_pxls = cm.hs.am.algo_prefs.preproc.sqrt_num_pxls
        if target_diag_pxls == -1: # Code for just doubleing the size
            current_num_diag_pxls = np.sqrt(rw**2 + rh**2) * 2
            target_diag_pxls = current_num_diag_pxls*2
            #target_diag_pxls = max(current_num_diag_pxls * 2, 5000)
        # Get raw aspect ratio
        ar = np.float(rw)/np.float(rh) 
        if ar > 4 or ar < .25: logwarn(
            'Aspect ratio for cx=%d %.2f may be too extreme' % (cx, ar))
        # Compute the scaled chip's tenative width and height
        cw = np.sqrt(ar**2 * target_diag_pxls**2 / (ar**2 + 1))
        ch = cw / ar
        if dtype is np.float:
            return (cw, ch)
        elif np.dtype(dtype).kind == 'f':
            return dtype(cw), dtype(ch)
        else:
            return dtype(round(cw)), dtype(round(ch))

    def _rotated_scaled_size(cm, cx):
        '''Returns the scaled size of cx considering rotation'''
        sz = np.array(cm._scaled_size(cx), dtype=np.float)
        theta = -cm.cx2_theta[cx]
        rot = np.array(([np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]), dtype=np.float)
        # Get bbox points around the origin
        pts_00 = np.array([(0,0), (sz[0],0), sz, (0, sz[1])]) - sz/2
        rot_pts = pts_00.dot(rot) 
        xymin = rot_pts.min(0)
        xymax = rot_pts.max(0)
        rot_scaled_sz = xymax - xymin
        return rot_scaled_sz

    def _cut_out_roi(cm, img, roi):
        logdbg('Image shape is: '+str(img.shape))
        [gh, gw]        = [ x-1 for x in img.shape[0:2] ]
        [rx1,ry1,rw,rh] = [ max(0,x) for x in roi]
        rx2 = min(gw, rx1+rw)
        ry2 = min(gh, ry1+rh)
        logdbg('Cutting out chip using: '+str((ry1,ry2,rx1,rx2)))
        raw_chip = img[ ry1:ry2, rx1:rx2, : ]
        return raw_chip

    def cx2_raw_chip(cm, cx):
        # --- Cut out the Raw Chip from Img
        # TODO: Save raw chips to disk?
        gm = cm.hs.gm
        gx = cm.cx2_gx[cx]
        roi = cm.cx2_roi[cx]
        # Read Image
        img = gm.gx2_img(gx) 
        return cm._cut_out_roi(img, roi)

    # TODO: Just have a flag for each preprocessing step. 
    # Move this over from AlgorithmManager
    def cx2_pil_chip(cm, cx, scaled=True, preprocessed=True, rotated=False, colored=False):
        am = cm.hs.am
        # Convert the raw image to PIL, and uncolor unless otherwise requested
        if not colored: 
            pil_chip = Image.fromarray( cm.cx2_raw_chip(cx) ).convert('L')
        else:
            pil_chip = Image.fromarray( cm.cx2_raw_chip(cx) )
        # Scale the image to its processed size
        if scaled:
            new_size = cm._scaled_size(cx, dtype=int)
            pil_chip = pil_chip.resize(new_size, Image.ANTIALIAS)
        if preprocessed:
            pil_chip = cm.hs.am.preprocess_chip(pil_chip)
        # Default do not rotate. Preprocessing is done beforehand
        if rotated:
            angle_degrees = cm.cx2_theta[cx]*180/np.pi
            pil_chip = pil_chip.rotate(angle_degrees, resample=Image.BICUBIC, expand=1)
        return pil_chip

    def  compute_chip(cm, cx):
        #TODO Save a raw chip and thumb
        iom = cm.hs.iom
        am  = cm.hs.am
        cid = cm.cx2_cid[cx]
        chip_fpath  = iom.get_chip_fpath(cid)
        chip_fname = os.path.split(chip_fpath)[1]
        logmsg(('\nComputing Chip: cid=%d fname=%s\n'+am.get_algo_name(['preproc'])) % (cid, chip_fname))
        # --- Preprocess the Raw Chip
        # Chip will be roated on disk np.load. Just scale for now
        chip = cm.cx2_pil_chip(cx, scaled=True, preprocessed=True,
                               rotated=False, colored=False)
        logdbg('Saving Computed Chip to :'+chip_fpath)
        chip.save(chip_fpath, 'PNG')
        # --- Write Chip and Thumbnail to disk
        chip_thumb_fpath  = iom.get_chip_thumb_fpath(cid)
        (cw, ch) = chip.size
        thumb_size = cm.hs.dm.draw_prefs.thumbnail_size
        thumb_scale = min(thumb_size/float(cw), thumb_size/float(ch))
        (tw, th) = (int(round(cw)), int(round(ch)))
        chip_thumb = chip.resize((tw, th), Image.ANTIALIAS)
        logdbg('Saving Computed Chip Thumb to :'+chip_thumb_fpath)
        chip_thumb.save(chip_thumb_fpath, 'JPEG')

    def load_features(cm, _cxs=None, force_recomp=False):
        if _cxs is None:
            _cxs = cm.get_valid_cxs()
        elif type(_cxs) is types.ListType:
            cxs = np.array(_cxs)
        elif type(_cxs) in [types.IntType, np.uint32]:
            cxs = np.array([_cxs])
        else: 
            cxs = _cxs
        count_feat = 0
        is_dirty  = np.bitwise_or(cm.cx2_dirty_bit[cxs], force_recomp)
        num_samp  = cxs.size
        num_dirty = np.sum(is_dirty)
        load_cx   = cxs[is_dirty]
        num_clean = num_samp - num_dirty
        #logdbg('Loading Features: Dirty=%d ; #Clean=%d' % (num_dirty, num_clean))
        if num_dirty == 0:
            return
        logio('Loading %d Feature Reps' % num_dirty)
        am = cm.hs.am
        for cx in load_cx:
            cid = cm.cx2_cid[cx]
            if cid <= 0:
                logwarn('WARNING: IX='+str(cx)+' is invalid'); continue
            chiprep_fpath = cm.hs.iom.get_chiprep_fpath(cid)
            if not force_recomp and filecheck(chiprep_fpath):
                logdbg('Loading features in '+chiprep_fpath)
                #Reload representation
                npz  = np.load(chiprep_fpath)
                fpts = npz['arr_0'] 
                fdsc = npz['arr_1']
                npz.close()
            else:
                #Extract and save representation
                logio('Computing and saving features of cid='+str(cid))
                [fpts, fdsc] = am.compute_features(cm.cx2_chip(cx))
                np.savez(chiprep_fpath, fpts, fdsc)
            cm.cx2_fpts[cx]  = fpts
            cm.cx2_fdsc[cx]  = fdsc
            cm.cx2_dirty_bit[cx] = False
            count_feat += len(fpts)
        
        logdbg('* Loaded '+str(count_feat)+' keypoints and fdscriptors' )
        return True
    
    def unload_features(cm, cxs):
        if not np.iterable(cxs):
            cxs = [cxs]
        nRequest = len(cxs)
        nUnload  = nRequest - np.sum(cm.cx2_dirty_bit[cxs])
        # Print unloaded cxs unless there are more than 3
        logdbg('Unloading features: %r' % cxs)
        logmsg('Unloading %d/%d features: ' % (nUnload, nRequest))
        cm.cx2_fpts[cxs] = np.empty(nUnload,dtype=object)
        cm.cx2_fdsc[cxs] = np.empty(nUnload,dtype=object)
        cm.cx2_fdsc[cxs] = np.empty(nUnload,dtype=object)
        cm.cx2_dirty_bit[cxs] = True

