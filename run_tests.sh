#python investigate_chip.py --dbG --tests vsmany_srule --all-gt-cases --sthresh 30 80  --printoff 
#python investigate_chip.py --dbG --tests vsmany_srule --all-gt-cases --sthresh 20 80  --printoff 
#python investigate_chip.py --dbG --tests vsmany_srule --all-gt-cases --sthresh 10 80  --printoff 
#python investigate_chip.py --dbG --tests vsmany_srule --all-gt-cases --sthresh  0 80  --printoff 
#python investigate_chip.py --dbG --tests vsmany_srule --all-gt-cases --sthresh  0 9001   --printoff 
#python investigate_chip.py --dbG --tests vsmany_srule --all-gt-cases --printoff 

cython_vs_python()
{
    python setup.py clean
    ./profiler.sh dev.py --db MOTHERS -t rr --nocache-query
    python setup.py cython
    ./profiler.sh dev.py --db MOTHERS -t rr --nocache-query
}
#cython_vs_python

simple_tests()
{
    python dev.py --db NAUTS -t best --all-gt-cases

}

mother_hard()
{

    export MOTHERS_HARD="--qcid 28 44 49 50 51 53 54 60 66 68 69 97 106 110"

    --print-rowlbl --print-colscore  --print-collbl --print-hardcase --all-gt-cases --print-hardcase --echo-hardcase

    python dev.py --db MOTHERS -t k_small $MOTHERS_HARD

    python dev.py --db MOTHERS -t k_big $MOTHERS_HARD --print-bestcfg

    python dev.py --db MOTHERS -t k_big --qcid 28 50 51 54 68 --print-bestcfg

    python dev.py --db MOTHERS -t adaptive_test $MOTHERS_HARD --print-bestcfg --view-all --dump
    

}

gz_hard()
{
    # cd /media/Store/data/work/GZ_ALL/_hsdb/computed/results/analysis
    export GZ_HARD="--qcid 140 183 184 231 253 276 277 287 289 306 311 316 329
    339 340 425 430 435 436 441 442 443 444 445 446 450 451 453 454 456 460 463
    465 501 550 553 589 661 662 681 694 720 786 802 803 812 815 817 838 908 941
    981 1043 1044 1045 1046 1047"

    # Not interesting: 
    # near duplicates
    # 183, 184, 276, 277, 329, 276
    # background matches
    # 253, 435, 430

    python dev.py --db GZ -t adaptive_test $GZ_HARD --print-bestcfg --view-all --dump

    python dev.py --db GZ -t k_small --print-rowlbl --print-colscore  --print-collbl --print-hardcase --all-gt-cases --print-hardcase --echo-hardcase

    python dev.py --db GZ -t k_big $GZ_HARD --print-bestcfg



    python dev.py --db GZ -t adaptive_test --qcid 140 183 184 231 253 276 277 287 289 306 311 316 329 339 340 425 430 435 436 441 442 443 444 445 446 450 451 453 454 456 460 463 465 501 550 553 589 661 662 681 694 720 786 802 803 812 815 817 838 908 941 981 1043 1044 1045 1046 1047 --print-bestcfg

    python dev.py --db GZ -t adaptive_test --all-gt-cases --print-bestcfg


    python dev.py --db GZ -t k_big --qcid 140 183 184 253 276 277 289 306 311 339 340 425 430 436 441 442 443 444 445 446 450 451 453 454 456 460 463 465 501 550 661 662 681 720 802 838 981 1044 1045 1046 1047 255 329 435 694 786 803 812 817 941 1014 1021 1043

}


run_overnight()
{
    overnight_huge_fn()
    {
        echo "!!!!!!!!!!!!!!!!!!!"
        echo "!!!!!!!!!!!!!!!!!!!"
        echo "==================="
        echo "OVERNIGHT HUGE TEST"
        echo "-------------------"
        echo "$1"
        echo "==================="
        echo "!!!!!!!!!!!!!!!!!!!"
        echo "!!!!!!!!!!!!!!!!!!!"
        python dev.py -t overnight_huge --all-gt-cases --dbdir $@
    }
    overnight_huge_fn GZ_ALL $@
    overnight_huge_fn HSDB_zebra_with_mothers $@
    overnight_huge_fn PZ_SweatwaterSmall $@

    #python dev.py --dbdir ~/data/work/PZ_FlankHack
    #python dev.py --dbdir ~/data/work/PZ_Marianne
    #python dev.py --dbdir ~/data/work/PZ_FlankHack
    #python dev.py --db GZ -t overnight_huge --all-gt-cases && python dev.py --db MOTHERS -t overnight_huge --all-gt-cases && 
}

dev_test()
{
    python dev.py --db GZ -t k_small --all-gt-cases --echo-hardcase


    python dev.py --db MOTHERS -t k_big --qcid 28 44 49 50 51 53 54 60 66 68 69 97 110
}

normrule_test()
{
    python dev.py --db MOTHERS -t normrule_test --all-gt-cases
}

verbose_test()
{
    echo "  +------------------------------------------------------------+"
    echo " /                                                            /"
    echo "+-------------------------------------------------------------"
    echo "| RUN TEST $@"
    echo "+-------------------------------------------------------------"
    python $@ #> nightly_output.txt
}

nightly_tests()
{
    export NIGHTLY_ARGS="--all-gt-cases --print-bestcfg --quiet $@"
    export TEST_LIST='coverage adaptive_test k_big normrule overnight_k'
    verbose_test dev.py $NIGHTLY_ARGS -t $TEST_LIST
    #verbose_test python dev.py $NIGHTLY_ARGS -t adaptive_test
    #verbose_test dev.py $NIGHTLY_ARGS -t adaptive_test 
    #verbose_test python dev.py $NIGHTLY_ARGS -t overnight_k
}

run_nightly()
{
    echo "Starting Nightly"
    nightly_tests --db MOTHERS
    nightly_tests --db GZ
    nightly_tests --db LF_ALL
    echo "Finished Nightly"
}

export TEST_TYPE="overnight"
# Check to see if nightly specified
if [[ $# -gt 0 ]] ; then
    if [[ "$1" = "nightly" ]] ; then
        export TEST_TYPE="nightly"
    fi
fi

echo "TEST_TYPE=$TEST_TYPE"

#normrule_test
if [[ "$TEST_TYPE" = "overnight" ]] ; then
    run_overnight $@
fi

if [[ "$TEST_TYPE" = "nightly" ]] ; then
    run_nightly
fi

#ic --db GZ --tests vsmany_big_social --all-gt-cases | tail
#ic --db GZ --tests vsmany_score --all-gt-cases | tail
#ic --db GZ --tests vsmany_sv --all-gt-cases | tail
#ic --db GZ --tests vsmany_k --all-gt-cases | tail

#ic --db MOTHERS --tests vsmany_big_social --all-gt-cases | tail
#ic --db MOTHERS --tests vsmany_score --all-gt-cases | tail
#ic --db MOTHERS --tests vsmany_sv --all-gt-cases | tail
#ic --db MOTHERS --tests vsmany_k --all-gt-cases | tail


#python investigate_chip.py --dbG --tests test-cfg-vsmany-3 --all-gt-cases --printoff 

#--nocache-query
#--nocache-query 
#--nocache-query 
#--nocache-query 
#alias icG='python investigate_chip.py --db GZ'
#alias icM='python investigate_chip.py --db MOTHERS'
#set icM=python investigate_chip.py --db MOTHERS
#set icG=python investigate_chip.py --db MOTHERS
#set icD=python investigate_chip.py --db MOTHERS
#alias icD=icM


# Scale Tests
#icM --tests test-cfg-vsmany-3 --all-gt-cases --sthresh 0 80
#icM --tests test-cfg-vsmany-3 --all-gt-cases --sthresh 5   80  --noprint | tail
# This seems to win
#icM --tests test-cfg-vsmany-3 --all-gt-cases --sthresh 10  80  --noprint | tail
#icM --tests test-cfg-vsmany-3 --all-gt-cases --sthresh 0  100  --noprint | tail
#icM --tests test-cfg-vsmany-3 --all-gt-cases --sthresh 5  100  --noprint | tail
#icM --tests test-cfg-vsmany-3 --all-gt-cases --sthresh 10 100  --noprint | tail
#icM --tests test-cfg-vsmany-3 --all-gt-cases --sthresh 5  150  --noprint | tail
#icM --tests test-cfg-vsmany-3 --all-gt-cases --sthresh 0 9001  --noprint | tail

# Visualize scales
#icM --tests kpts-scale --sthresh  0 9001 --printoff | tail

#python investigate_chip.py --dbG --histid 4 5 7 8 10 11 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46

#python investigate_chip.py --dbG --r 0 --c 0 1 2 3 --tests test-cfg-vsmany-3 --histid 4 
#python investigate_chip.py --dbG --r 0 --c 0 1 2 3 --tests test-cfg-vsmany-3 --histid 5 


# TODO: In show match annote
# click a keypoint match to see it in detail
# click a keypoint on the query to see the image that it matched to
