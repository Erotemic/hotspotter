#/usr/bin/env python
from __future__ import print_function, division
import sys

if __name__ == '__main__':
    # Only profiled functions that are run are printed
    input_fname = sys.argv[1]
    output_fname = sys.argv[2]

    with open(input_fname) as file_:
        text = file_.read()
        list_ = text.split('File:')
        for ix in xrange(1, len(list_)):
            list_[ix] = 'File: ' + list_[ix]

        newlist = []
        for ix in xrange(len(list_)):
            block = list_[ix]
            time_key = 'Total time:'
            timepos = block.find(time_key)
            if timepos == -1:
                newlist.append(block)
                continue
            timepos += len(time_key)
            nlpos = block[timepos:].find('\n')
            timestr = block[timepos:timepos + nlpos]
            total_time = float(timestr.replace('s', '').strip())
            if total_time != 0:
                newlist.append(block)

        with open(output_fname, 'w') as file2_:
            file2_.write('\n'.join(newlist))
