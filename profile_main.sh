#!/bin/sh
kernprof.py -l main.py $@

export TIMESTAMP=$(date -d "today" +"%Y-%m-%d_%H-%M-%S")
export FULL_PROFILE=main.py.$TIMESTAMP.full.profile.txt
export FIXED_PROFILE=main.py.$TIMESTAMP.fixed.profile.txt

python -m line_profiler main.py.lprof >> $FULL_PROFILE

python fix_profile.py $FULL_PROFILE $FIXED_PROFILE

cat $FIXED_PROFILE
