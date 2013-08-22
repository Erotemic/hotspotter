from other.logger import logmsg, logdbg, logerr, logwarn
from other.helpers import alloc_lists
from other.AbstractPrintable import AbstractDataManager
from pylab import find
from PIL          import Image
from shutil       import copyfile
from numpy        import array, asarray, append, zeros, uint32, bool, logical_and, iterable
from hotspotter.helpers import img_ext_set
import os.path

'''
This file has responcibility over managing
the data tranfer of images

 Notation used in this file: 
 g     = imaGe
 gx    = imaGe indeX
 gid   = image ID
 gname = image file name+extension
 aif   = all images found 
 cxs   = chip indexes belonging to this image
'''
class ImageManager(AbstractDataManager):

    # --- GID CONVINENCE FUNCTIONS ---
    def is_valid(gm, gid):
       return gid < len(gm.gid2_gx) and gid >= 0 and gm.gid2_gx[gid] > 0
    def gx(gm, gid):
        if not gm.is_valid(gid):
            logerr('GID=%s is invalid' % str(gid))
        return gm.gid2_gx[gid]
    def cid_list(gm, gid):
        cx_list = gm.gx2_cx_list[gm.gx(gid)]
        return gm.hs.cm.cx2_cid[cx_list]
    def info(gm, gid_list=None, lbls=None):
        return gm.gx2_info(gm.gx(gid_list))

    def get_empty_gxs(gm):
        'returns gxs without any cxs'
        return find(logical_and(array(map(len, gm.gx2_cx_list)) == 0, gm.gx2_gid > 0))

    #TODO: Should move chip stuff to chip?
    def __init__(gm,hs=None):
        super( ImageManager, gm ).__init__( hs )
        logdbg('Creating Image Manager')
        # --- Flat Table ---
        gm.gx2_gid     = array([], dtype=uint32) 
        gm.gx2_gname   = array([], dtype=object)       
        gm.gx2_aif_bit = array([], dtype=bool)
        gm.gx2_cx_list     = array([], dtype=list) 
        # --- Reverse Indexes ---
        gm.gname2_gid  = {}
        gm.gid2_gx     = array([], dtype=uint32)
        # --- Statistics --
        gm.next_gx     = 1 
        gm.next_gid    = 1 
        gm.num_g       = 0
        gm.max_gx      = 0
        gm.max_gid     = 0
        gm.max_gnamelen = 0
        gm.max_gname = ''
        #----------
        gm.hs = hs #Parent
        gm.x2_lbl = \
        {
            'gid'   : lambda x: gm.gx2_gid[x],\
            'aif'   : lambda x: gm.gx2_aif_bit[x],\
            'cxs'   : lambda x: gm.gx2_cx_list[x],\
            'cids'  : lambda x: str(gm.gx2_cids(x)),\
            'gname' : lambda x: gm.gx2_gname[x] ,\
            'num_c' : lambda x: gm.gx2_num_c(x),\
        }
        gm.valid_img_extensions = img_ext_set 

    def img_alloc(gm, nAlloc):
        logdbg('Allocating room for %d more images' % nAlloc)
        #-- Forward Allocation
        gm.gx2_gid     = append(gm.gx2_gid,     zeros(nAlloc,dtype=uint32))
        gm.gx2_gname   = append(gm.gx2_gname,   zeros(nAlloc,dtype=object))
        gm.gx2_aif_bit = append(gm.gx2_aif_bit, zeros(nAlloc,dtype=bool))
        gm.gx2_cx_list     = append(gm.gx2_cx_list, alloc_lists(nAlloc))
        #-- Inverse Allocation
        idAlloc        = len(gm.gx2_gid) - len(gm.gid2_gx)
        if idAlloc > 0:
            gm.gid2_gx = append(gm.gid2_gx, zeros(idAlloc,dtype=uint32))

    def gx2_info(gm, gxs=None, lbls=None):
        if gxs is None:
            gxs = gm.get_valid_gxs()
        if lbls is None: 
            lbls = ['gid','gname','aif']
        data_table_str = gm.x2_info(gxs, lbls)
        return '# ImageManager\n'+data_table_str

    def gx2_img_fpath(gm, gx): # returns full path
        'Gets the img fpath with checks'
        iom = gm.hs.iom
        gname = gm.gx2_gname[gx]
        if gname is None: logerr('There is no image for GX='+str(gx))
        img_fpath = iom.get_img_fpath(gname)
        if not os.path.exists(img_fpath):
            logerr('The data is gone!\nGX=%d, img_fpath=%s' % (gx, img_fpath))
        return img_fpath

    def gx2_img_list(gm, gx_list): 
        if not iterable(gx_list): return gm.gx2_img_list([gx_list])
        img_fpath_list = [ gm.gx2_img_fpath(gx) for gx in iter(gx_list)  ]
        return [ asarray(Image.open(img_fpath)) for img_fpath in iter(img_fpath_list) ]

    def gx2_img(gm, gx): # returns actual image
        img_fpath = gm.gx2_img_fpath(gx)
        return asarray(Image.open(img_fpath)) 

    def gx2_img_size(gm,gx): # returns height / width
        img_fpath = gm.gx2_img_fpath(gx)
        img = Image.open(img_fpath)
        return img.size

    def gx2_img_thumb(gm, gx):
        iom = gm.hs.iom
        gname = gm.gx2_gname[gx]
        gid   = gm.gx2_gid[gx]
        img_thumb_fpath = iom.get_img_thumb_fpath(gname)
        if not os.path.exists(img_thumb_fpath):
            logdbg('Computing thumbnail of GID=' + str(gid))
            raw_img = gm.gx2_img(gx)
            thumb_size = cm.hs.dm.draw_prefs.thumbnail_size
            raw_img.thumbnail((thumb_size,thumb_size), Image.ANTIALIAS).save\
                    (img_thumb_fpath, 'JPEG')
            logdbg('Wrote thumbnail.')
        return asarray(Image.open(img_thumb_fpath))

    def gx2_cids(gm, gxs): # returns cids belonging to gx
        return  map(lambda cxs: gm.hs.cm.cx2_cid[cxs], gm.gx2_cx_list[gxs])

    def gx2_num_c(gm, gxs): # returns cids belonging to gx
        return len(gm.gx2_cx_list[gxs])

    def add_img(gm, gid=None, gname=None, aif=False, src_img=''):
        logdbg('Adding Image: gid=%r, gname=%r, aif=%r, src_img=%r' % (gid, gname, aif, src_img))
        if src_img != '': #This is an new image
            (dir_part, nameext_part)  = os.path.split(src_img)
            (name_part, ext_part) = os.path.splitext(nameext_part)
            if not ext_part.lower() in gm.valid_img_extensions:
                logerr('Invalid Image: %s' % src_img)
            if gname is None:
              gname   = name_part + ext_part
            db_img  = os.path.join(gm.hs.iom.get_img_dpath(), gname)
            if os.path.abspath(src_img) == os.path.abspath(db_img):
                logmsg('Readding existing dbimg:'+src_img)
            else:
                logmsg('Copying '+src_img+' to '+db_img)
                copyfile(src_img, db_img)
        db_img  = os.path.join(gm.hs.iom.get_img_dpath(), gname)
        if not os.path.exists(db_img):
            # Try to add an extension if it wasn't given
            ext_fallback_list = ['.jpg','.jpeg','.JPG','.JPEG','.png','.tif']
            fb_sucess_bit = False
            (db_img_noext, old_ext)  = os.path.splitext(db_img)
            (gname_noext, old_ext2) = os.path.splitext(gname)
            for ext_fb in ext_fallback_list:
                db_img_fb = db_img_noext + ext_fb
                if os.path.exists(db_img_fb):
                    db_img = db_img_fb
                    gname = gname_noext+ext_fb
                    fb_sucess_bit = True; break
            if not fb_sucess_bit:
                logwarn('Trying to add a nonexistant image: '+db_img)
                return
        if gname in gm.gname2_gid.keys():
            logdbg('Trying to add a GNAME that is already managed: '+gname)
            return
        if gid is None or gid < 1:
            gid = gm.next_gid
        else:
            if (gid < len(gm.gid2_gx)-1 and gm.gid2_gx[gid] > 0):
                logdbg('Trying to add a GID that is already managed: '+str(gid))
                return gid
        #Check key values before memory managment
        #Manage Memory
        gx = gm.next_gx
        if gx >= len(gm.gx2_gid):
            gm.img_alloc((len(gm.gx2_gid)+1)*2+1)
        #Flat indexing
        gm.gx2_gid[gx]   = gid
        gm.gx2_gname[gx] = gname
        if len(gname) > gm.max_gnamelen:
            gm.max_gnamelen = len(gname)
            gm.max_gname = gname
        
        gm.gx2_aif_bit[gx]   = aif
        #X Reverse indexing
        if len(gm.gid2_gx) <= gid:
            gid_extend = (gid - len(gm.gid2_gx)) + 1
            gm.gid2_gx = append(gm.gid2_gx, zeros(gid_extend,dtype=uint32))
        gm.gid2_gx[gid] = gx
        gm.gname2_gid[gname] = gid
        #Increment
        gm.next_gx = gm.next_gx + 1
        gm.next_gid = max(gm.next_gid+1, gid+1)
        gm.num_g = gm.num_g + 1
        gm.max_gx  = max(gm.max_gx,  gx)
        gm.max_gid = max(gm.max_gid, gx)
        return gid

    def get_valid_gxs(cm): 
        return find(cm.gx2_gid > 0)

    def get_invalid_gxs(cm):
        return find(cm.gx2_cid == 0)

