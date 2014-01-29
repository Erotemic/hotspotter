#!/bin/sh
# Globals
export TIMESTAMP=$(date -d "today" +"%Y-%m-%d_%H-%M-%S")
export RAW_SUFFIX=raw.prof
export CLEAN_SUFFIX=clean.prof

# Input
export pyscript=$1
# Output Stage 1
export line_profile_output=$pyscript.lprof
# Output Stage 2
export raw_profile=raw_profile.$pyscript.$TIMESTAMP.$RAW_SUFFIX
export clean_profile=clean_profile.$pyscript.$TIMESTAMP.$CLEAN_SUFFIX

echo "Profiling $pyscript"
# Line profile the python code w/ command line args
kernprof.py -l $@
# Dump the line profile output to a text file
python -m line_profiler $line_profile_output >> $raw_profile
# Clean the line profile output
python _scripts/profiler_cleaner.py $raw_profile $clean_profile
# Print the cleaned output
cat $clean_profile

remove_profiles()
{
    rm *.profile.txt
}
