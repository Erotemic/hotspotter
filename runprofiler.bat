set SCRIPTNAME=dev.py
:: set SCRIPTARGS=--db NAUTS --nocache-feats --serial
:: set SCRIPTARGS=--db MOTHERS -t vsmany_best --all-gt-cases --nocache-query --nocache-feats --serial
set SCRIPTARGS=--db MOTHERS -t vsmany_best --all-gt-cases --nocache-query --serial
rm %SCRIPTNAME%.lprof
kernprof.py -l %SCRIPTNAME% %SCRIPTARGS%
python -m line_profiler %SCRIPTNAME%.lprof
