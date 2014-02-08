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
python dev.py --db MOTHERS -t k_small --print-rowlbl --print-colscore  --print-collbl --print-hardcase --all-gt-cases --print-hardcase --echo-hardcase
    
python dev.py --db MOTHERS -t k_big --qcid 28 44 49 50 51 53 54 60 66 68 69 97 106 110 --print-bestcfg

python dev.py --db MOTHERS -t k_big --qcid 28 50 51 54 68 --print-bestcfg

}

gz_hard()
{

python dev.py --db GZ -t adaptive_test
    
python dev.py --db GZ -t k_small --print-rowlbl --print-colscore  --print-collbl --print-hardcase --all-gt-cases --print-hardcase --echo-hardcase

python dev.py --db GZ -t k_big --qcid 140 183 184 253 276 277 289 306 311 339 340 425 430 436 441 442 443 444 445 446 450 451 453 454 456 460 463 465 501 550 661 662 681 720 802 838 981 1044 1045 1046 1047 231 287 316 329 435 553 589 694 786 803 812 815 817 908 941 1043 --print-bestcfg


python dev.py --db GZ -t adaptive_test --qcid 140 183 184 253 276 277 289 306 311 339 340 425 430 436 441 442 443 444 445 446 450 451 453 454 456 460 463 465 501 550 661 662 681 720 802 838 981 1044 1045 1046 1047 231 287 316 329 435 553 589 694 786 803 812 815 817 908 941 1043 --print-bestcfg

python dev.py --db GZ -t adaptive_test --all-gt-cases --print-bestcfg


python dev.py --db GZ -t k_big --qcid 140 183 184 253 276 277 289 306 311 339 340 425 430 436 441 442 443 444 445 446 450 451 453 454 456 460 463 465 501 550 661 662 681 720 802 838 981 1044 1045 1046 1047 255 329 435 694 786 803 812 817 941 1014 1021 1043

}


run_overnight()
{
    python dev.py --dbdir ~/data/work/PZ_FlankHack
    python dev.py --dbdir ~/data/work/PZ_Marianne
    python dev.py --dbdir ~/data/work/PZ_FlankHack
    python dev.py --db GZ -t overnight_huge --all-gt-cases && python dev.py --db MOTHERS -t overnight_huge --all-gt-cases && 
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
#normrule_test
dev_test

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
