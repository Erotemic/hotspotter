
run_nightly()
{
    python _tests/newdb_test.py $@
    python _tests/coverage_test.py --db NAUTS
    python _tests/query_test.py --db JAG --qcid 1 --nosteal --noshare
    python _tests/query_test.py --db MOTHERS --qcid 1 --nosteal --noshare

    python _tests/query_test.py --db NAUTS --qcid 1 --nocache-query --nocache-feats --nocache-chips --strict
    python _tests/query_test.py --db JAG --qcid 1 --nocache-query --nocache-feats --nocache-chips --strict
    python _tests/query_test.py --db MOTHERS --qcid 28 --nocache-query --nocache-feats --nocache-chips --strict
}


clean_databases()
{
    python dev.py --db NAUT_DAN --delete-cache 
    python dev.py --db FROG_tufts --delete-cache 
    python dev.py --db WD_Siva --delete-cache 
}

run_experiment()
{
    gz_chipsize_experiment $@
}

gz_chipsize_experiment()
{
    export GZ_HARD="--qcid 140 183 184 231 253 276 277 287 289 306 311 316 329
    339 340 425 430 435 436 441 442 443 444 445 446 450 451 453 454 456 460 463
    465 501 550 553 589 661 662 681 694 720 786 802 803 812 815 817 838 908 941
    981 1043 1044 1045 1046 1047"

    python dev.py --db GZ --print-colscore -t chipsize_test --all-gt-cases $@
}

big_experiment()
{
    export TEST_NAME="shortlist_test chipsize_test scale_test overnight_k"
    export DEV_ARGS="--all-gt-cases --print-colscore -t $TEST_NAME $@"
    export DB_LIST="NAUT_Dan WD_Siva PZ_SweatwaterSmall HSDB_zebra_with_mothers GZ_ALL WY_Toads Wildebeast"
    for DB in $DB_LIST
    do
        python dev.py --db $DB $DEV_ARGS
    done
}


default()
{
    export TEST_NAME="shortlist_test chipsize_test scale_test "
    export DEV_ARGS="--all-gt-cases --print-colscore -t $TEST_NAME $@"
    #export DB_LIST="NAUTS FROGS"
    export DB_LIST="NAUTS"
    for DB in $DB_LIST
    do
        python dev.py --db $DB $DEV_ARGS
    done

}

run_continuous()
{
    #python _tests/query_test.py
    python _tests/coverage_test.py --db NAUTS $@
    python _tests/query_test.py --db NAUTS $@
    #--nocache-query --nocache-feats --nocache-chips --strict
}

export TEST_TYPE=""
# Check to see if nightly specified
if [[ $# -gt 0 ]] ; then
    if [[ "$1" = "nightly" ]] ; then
        export TEST_TYPE="nightly"
    elif [[ "$1" = "big" ]] ; then
        export TEST_TYPE="big"
    elif [[ "$1" = "continuous" ]] ; then
        export TEST_TYPE="continuous"
    elif [[ "$1" = "experiment" ]] ; then
        export TEST_TYPE="experiment"
    elif [[ "$1" = "cleandb" ]] ; then
        export TEST_TYPE="cleandb"
    fi
fi

echo "TEST_TYPE=$TEST_TYPE"

#normrule_test
if [[ "$TEST_TYPE" = "continuous" ]] ; then
    run_continuous ${@:2}
elif [[ "$TEST_TYPE" = "experiment" ]] ; then
    run_experiment ${@:2}
elif [[ "$TEST_TYPE" = "nightly" ]] ; then
    run_nightly ${@:2}
elif [[ "$TEST_TYPE" = "big" ]] ; then
    big_experiment ${@:2}
elif [[ "$TEST_TYPE" = "cleandb" ]] ; then
    clean_databases ${@:2}
else
    default $@
fi
