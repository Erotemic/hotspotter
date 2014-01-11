#kernprof.py -l dev.py --db MOTHERS -t vsmany_best --nocache-query
#python -m line_profiler dev.py.lprof
kernprof.py -l dev.py --db NAUTS -t dists
python -m line_profiler dev.py.lprof
