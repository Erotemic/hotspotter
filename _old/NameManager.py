from other.AbstractPrintable import AbstractDataManager
from other.logger import logdbg, logerr, logwarn
from other.helpers import alloc_lists
from numpy import array, append, zeros, empty, uint32
from pylab import find

#TODO: Name should eventually become category.
# Name Manager handle the names/labels/categories
class NameManager(AbstractDataManager):

    def name2_cx_list(nm, name):
        try:
            nx = nm.name2_nx[name]
        except KeyError: 
            return []
        return nm.nx2_cx_list[nx]

    def __init__(nm,hs=None):
        super( NameManager, nm ).__init__( hs )
        # --- Flat Table ---
        nm.nx2_nid     = array([], dtype=uint32) # index to imaGe id
        nm.nx2_name    = array([], dtype=object) # index to full img label excluding extension
        #--
        nm.nx2_cx_list     = array([], dtype=list)  # index to chip cxs it contains
        # --- Reverse Indexes ---
        nm.name2_nx    = {}
        nm.nid2_nx     = array([], dtype=uint32)
        # --- Statistics --
        nm.next_nx     =  1 #the next indeX we are going to use
        nm.next_nid    =  1 #the next indeX we are going to use
        nm.num_n       =  0
        nm.max_nx      =  0
        nm.max_nid     =  0
        nm.max_namelen =  0
        nm.max_name =  ''
        nm.UNIDEN_NID = 1
        nm.add_name(nm.UNIDEN_NID,'____')
        nm.x2_lbl = \
        { 'nid'  : lambda x: nm.nx2_nid[x],
          'cxs'  : lambda x: nm.nx2_cx_list[x],
          'cids' : lambda x: str(nm.nx2_cids(x)),
          'name' : lambda x: nm.nx2_name[x]} #for x2_info
    # --- Convienience functions
    def  get_valid_nxs(nm, min_chips=1, uniden_is_valid=False):
        ''' Returns Identified Names that have Chips'''
        all_numc = nm.nx2_numc()
        if not uniden_is_valid:
            all_numc[nm.UNIDEN_NID] = 0
        return find(all_numc >= min_chips)
    def UNIDEN_NX(nm):
        return nm.nid2_nx[nm.UNIDEN_NID]
    def UNIDEN_NAME(nm):
        return nm.nx2_name[nm.UNIDEN_NX()]
    def iter_nx(nm):
        return xrange(0,nm.max_nx+1)
    def  nx2_cids(nm,nxs):
        return map(lambda cxs: nm.hs.cm.cx2_cid[cxs], nm.nx2_cx_list[nxs])
    def  nx2_numc(nm, nxs=None):
        if nxs == None:
            cxs = nm.nx2_cx_list
        else:
            cxs = nm.nx2_cx_list[nxs]
        return array(map(lambda x: len(x), cxs),dtype=uint32)
    def  get_new_name(nm):
        name_num = nm.next_nx
        novel_name = 'AutoName_'+str(name_num)
        while novel_name in nm.name2_nx.keys():
            name_num = name_num + 1
            novel_name = 'AutoName_'+str(name_num)
        return novel_name
    def  rename_all(nm, nx, new_name):
        nm.nx2_name[nx] = new_name
    # --- Work functions
    def  name_alloc(nm, xAlloc):
        logdbg('Allocating room for '+str(xAlloc)+' more names')
        nm.nx2_nid    = append(nm.nx2_nid,  zeros(xAlloc,dtype=uint32))
        nm.nx2_name   = append(nm.nx2_name, empty(xAlloc,dtype=object))
        nm.nx2_cx_list    = append(nm.nx2_cx_list, alloc_lists(xAlloc))
        idAlloc       = len(nm.nx2_nid) - len(nm.nid2_nx) + 2
        if idAlloc > 0:
            logdbg('Allocating room for '+str(idAlloc)+' more reverse nids')
            nm.nid2_nx = append(nm.nid2_nx, zeros(idAlloc,dtype=uint32)) 

    def nx2_info(nm, nxs=None, lbls=None):
        'returns string formmated in a table'
        if nxs is None:
            nxs = nm.get_valid_nxs()
        if lbls is None: 
            lbls = ['nid', 'name']
        data_table_str = nm.x2_info(nxs,lbls)
        return '# NameManager\n'+data_table_str

    def  add_name(nm, nid_, name_):
        'Adds a name. If nid == -1 a new nid will be assigned. Returns nid'
        logdbg('Adding nid='+str(nid_)+' name='+name_)
        nid = nid_
        name = name_.strip()
        if name == '':
            logerr('Cannot add an empty name!')
        if nid < 0:
            # Generate new nid if not specified
            nid = nm.next_nid
        else:
            #If NID already exists and has an entry,
            #do not increment, and do nothing
            #This is essentially like doing a rename
            #it doesn't actually change anything
            if (nid < len(nm.nid2_nx) and nm.nid2_nx[nid] > 0):
                logwarn('NID '+str(nid)+'already exists')
                nx = nm.nid2_nx[nid]
                return -1
        if name in nm.name2_nx.keys():
            conflict_nx = nm.name2_nx[name]
            conflict_nid = nm.nx2_nid[conflict_nx]
            conflict_msg = 'Name %s already exists in database!\n' % name + \
                           'NID=%d; NEXT_NX=%d\n' % (nid, nm.next_nx) + \
                           'CONFLICT_NID=%d\n' % conflict_nid 
            if nid_ == -1:
                logwarn(conflict_msg)
            elif nid_ > -1:
                logerr(conflict_msg)
            nid = conflict_nid
            nx  = conflict_nx
            return nid
        #Manage Memory
        nx = nm.next_nx
        logdbg(' * nx = '+str(nx))
        if nx >= len(nm.nx2_nid):
            nm.name_alloc((len(nm.nx2_nid)+1)*2+1)
        #Add to flat table
        nm.nx2_nid[nx]   = nid
        nm.nx2_name[nx]  = name
        if len(name) > nm.max_namelen:
            nm.max_namelen = len(name)
            nm.max_name = name
        #X Reverse indexing
        if len(nm.nid2_nx) <= nid:
            idAlloc = (nid - len(nm.nid2_nx)) + 1
            nm.nid2_nx = append(nm.nid2_nx, zeros(idAlloc,dtype=uint32)) 
        logdbg( ' * nid2_nx['+str(nid)+']='+str(nx) )
        nm.nid2_nx[nid]   = nx
        nm.name2_nx[name] = nx
        #Increment
        nm.next_nx  = nm.next_nx + 1
        nm.next_nid = max(nm.next_nid + 1, nid + 1)
        nm.max_nx   = max(nm.max_nx , nx)
        nm.max_nid  = max(nm.max_nid, nid)
        nm.num_n    = nm.num_n + 1
        logdbg('Added nid='+str(nid)+' name='+name)
        return nid

