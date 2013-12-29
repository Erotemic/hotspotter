set SCRIPTNAME=dev.py
:: set SCRIPTARGS=--db NAUTS --nocache-feats --serial
:: set SCRIPTARGS=--db MOTHERS -t vsmany_best --all-gt-cases --nocache-query --nocache-feats --serial
set SCRIPTARGS=--db MOTHERS -t vsmany_best --nocache-query --qcid 1 2 3 4 5 6 7 9 10 --serial
:: rm %SCRIPTNAME%.lprof
kernprof.py -l %SCRIPTNAME% %SCRIPTARGS%

:: rm %SCRIPTNAME%.prof
kernprof.py %SCRIPTNAME% %SCRIPTARGS%

python -m line_profiler %SCRIPTNAME%.lprof
runsnake %SCRIPTNAME%.prof
