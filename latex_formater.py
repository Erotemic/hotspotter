import helpers
import textwrap
import numpy as np

def latex_multicolumn(data, ncol=2):
    return r'\multicolumn{%d}{|c|}{%s}' % (ncol, data)
def latex_multirow(data, nrow=2):
    return r'\multirow{%d}{*}{|c|}{%s}' % (nrow, data)
def latex_mystats(lbl, data):
    stats_ = helpers.mystats(data);
    min_, max_, mean, std, shape = stats_.values()
    fmttup1 = (int(min_), int(max_), float(mean), float(std))
    fmttup = tuple(map(helpers.num_fmt, fmttup1))
    lll = ' '*len(lbl)
    #fmtstr = r'''
    #'''+lbl+r''' stats &{ max:%d, min:%d\\
    #'''+lll+r'''       & mean:%.1f, std:%.1f}\\'''
    fmtstr = r'''
    '''+lbl+r''' stats & max ; min = %s ; %s\\
    '''+lll+r'''       & mean; std = %s ; %s\\'''
    latex_str = textwrap.dedent(fmtstr % fmttup).strip('\n')+'\n'
    return latex_str
def latex_scalar(lbl, data):
    return (r'%s & %s\\' % (lbl, helpers.num_fmt(data)))+'\n'


def make_stats_tabular():
    'tabular for dipslaying statistics'
    pass

def ensure_rowvec(arr):
    arr = np.array(arr)
    arr.shape = (1, arr.size)
    return arr

def ensure_colvec(arr):
    arr = np.array(arr)
    arr.shape = (arr.size, 1)
    return arr

def padvec(shape=(1,1)):
    pad = np.array([[' ' for c in xrange(shape[1])] for r in xrange(shape[0])])
    return pad

def make_scoring_tabular(row_lbls, col_lbls, scores, title=None):
    'tabular for displaying scores'
    # Abbreviate based on common substrings
    SHORTEN_ROW_LBLS = True
    common_rowlbl = None
    if SHORTEN_ROW_LBLS:
        row_lbl_list = row_lbls.flatten().tolist()
        longest = long_substr(row_lbl_list)
        common_strs = []
        while len(longest) > 10: 
            common_strs += [longest]
            row_lbl_list = [row.replace(longest,'...') for row in row_lbl_list] 
            longest = long_substr(row_lbl_list)
        common_rowlbl = '...'.join(common_strs)
        row_lbls = row_lbl_list

    # Stack into a tabular body
    col_lbls = ensure_rowvec(col_lbls)
    row_lbls = ensure_colvec(row_lbls)
    _0 = np.vstack([padvec(), row_lbls])
    _1 = np.vstack([col_lbls, scores])
    body = np.hstack([_0, _1])

    ALIGN_BODY = True
    if ALIGN_BODY: 
        # Equal length columns
        new_body_cols = []
        for col in body.T:
            colstrs = map(str, col.tolist())
            collens = map(len, colstrs)
            maxlen = max(collens)
            newcols = [str_ + (' '*(maxlen-len(str_))) for str_ in colstrs]
            new_body_cols += [newcols]
        body = np.array(new_body_cols).T


    AUTOFIX_LATEX = True
    if AUTOFIX_LATEX:
        for r in xrange(body.shape[0]):
            for c in xrange(body.shape[1]):
                body[r,c] = body[r,c].replace('#','\\#')

    # Build Body (and row layout)
    HLINE_SEP = True
    rowsep = ''
    colsep = '&'
    endl = '\\\\\n'
    hline = r'\hline'
    extra_rowsep_pos_list = [1]
    if HLINE_SEP:
        rowsep = hline+'\n'
    rowstr_list = [colsep.join(row)+endl for row in body]
    # Insert title
    if not title is None:
        tex_title = latex_multicolumn(title, body.shape[1])+endl
        rowstr_list = [tex_title] + rowstr_list
        extra_rowsep_pos_list += [2]
    # Apply an extra hline (for label)
    for pos in sorted(extra_rowsep_pos_list)[::-1]:
        rowstr_list.insert(pos, '')
    tabular_body = rowsep.join(rowstr_list)

    # Build Column Layout
    col_layout_sep = '|'
    col_layout_list = ['l']*body.shape[1]
    extra_collayoutsep_pos_list = [1]
    for pos in  sorted(extra_collayoutsep_pos_list)[::-1]:
        col_layout_list.insert(pos, '')
    col_layout = col_layout_sep.join(col_layout_list)

    tabular_head = (r'\begin{tabular}{|%s|}' % col_layout)+'\n'
    tabular_tail = r'\end{tabular}'

    tabular_str = rowsep.join([tabular_head, tabular_body, tabular_tail])

    if not common_rowlbl is None: 
        tabular_str += '\nThe following perameters were held fixed: '+common_rowlbl
    return tabular_str


def _tabular_header_and_footer(col_layout):
    tabular_head = dedent(r'\begin{tabular}{|%s|}' % col_layout)
    tabular_tail = dedent(r'\end{tabular}')
    return tabular_head, tabular_tail
    
def _tabular_title():
    pass

def _make_data_cols(data):
    pass

def _make_rowlbl_col(rob_lbls):
    col_layout_aug = 'l||'

def long_substr(strlist):
    # Longest common substring
    # http://stackoverflow.com/questions/2892931/longest-common-substring-from-more-than-two-strings-python
    substr = ''
    if len(strlist) > 1 and len(strlist[0]) > 0:
        for i in range(len(strlist[0])):
            for j in range(len(strlist[0])-i+1):
                if j > len(substr) and is_substr(strlist[0][i:i+j], strlist):
                    substr = strlist[0][i:i+j]
    return substr

def is_substr(find, strlist):
    if len(strlist) < 1 and len(find) < 1:
        return False
    for i in range(len(strlist)):
        if find not in strlist[i]:
            return False
    return True




