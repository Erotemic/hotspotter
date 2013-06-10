cid_blocks = open('D:\data\work\LF_Bajo_bonito\.hs_internals\computed\expt__2013-05-09_19-10\subset\matches.txt').read().split("\n\n")


x = []
for c in cid_blocks:
    if c != '\n':
        x += [int(c[c.find('score')+6:c.find('score')+10])]
    else:
        x += [0]

outfile = open('D:\data\work\LF_Bajo_bonito\.hs_internals\computed\expt__2013-05-09_19-10\subset\sorted-matches.txt','w')
outfile.write( '\n\n'.join(np.array(cid_blocks)[np.array(x).argsort()[::-1]]))
outfile.close()
