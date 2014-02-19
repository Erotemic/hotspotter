
nightly_tests()
{
    python _tests/newdb_test.py
    python _tests/coverage_test.py --db NAUTS
    python _tests/query_test.py --db NAUTS --nocache-query --nocache-feats --nocache-chips --strict
    python _tests/query_test.py --db JAG --nocache-query --nocache-feats --nocache-chips --strict
    python _tests/query_test.py --db MOTHERS --nocache-query --nocache-feats --nocache-chips --strict
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
