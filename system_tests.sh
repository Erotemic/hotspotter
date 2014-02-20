
nightly_tests()
{
    python _tests/newdb_test.py
    python _tests/coverage_test.py --db NAUTS
    python _tests/query_test.py --db JAG --qcid 1 --nosteal --noshare
    python _tests/query_test.py --db MOTHERS --qcid 1 --nosteal --noshare

    python _tests/query_test.py --db NAUTS --qcid 1 --nocache-query --nocache-feats --nocache-chips --strict
    python _tests/query_test.py --db JAG --qcid 1 --nocache-query --nocache-feats --nocache-chips --strict
    python _tests/query_test.py --db MOTHERS --qcid 28 --nocache-query --nocache-feats --nocache-chips --strict
}

experiment_tets()
{
    python _tests/query_test.py --db NAUTS --indent
    python dev.py --db NAUTS --delete-cache -t scale_test
    #python dev.py --dbdir D:/data/work/FROG_tufts --delete-cache 
    python dev.py --dbdir D:/data/work/FROG_tufts -t small_scale_test --all-gt-cases --print-colscore
    python dev.py --dbdir D:/data/work/WD_Siva -t small_scale_test --all-gt-cases --print-colscore


}

continuous_tests()
{
    #python _tests/query_test.py
    python _tests/coverage_test.py --db NAUTS
    python _tests/query_test.py --db NAUTS
    #--nocache-query --nocache-feats --nocache-chips --strict
}

export TEST_TYPE="continuous"
# Check to see if nightly specified
if [[ $# -gt 0 ]] ; then
    if [[ "$1" = "nightly" ]] ; then
        export TEST_TYPE="nightly"
    fi
fi

echo "TEST_TYPE=$TEST_TYPE"

#normrule_test
if [[ "$TEST_TYPE" = "continuous" ]] ; then
    run_continuous $@
fi

if [[ "$TEST_TYPE" = "nightly" ]] ; then
    run_nightly
fi
