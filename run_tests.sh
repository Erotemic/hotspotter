#python investigate_chip.py --dbG --tests vsmany_srule --all-gt-cases --sthresh 30 80  --printoff 
#python investigate_chip.py --dbG --tests vsmany_srule --all-gt-cases --sthresh 20 80  --printoff 
#python investigate_chip.py --dbG --tests vsmany_srule --all-gt-cases --sthresh 10 80  --printoff 
#python investigate_chip.py --dbG --tests vsmany_srule --all-gt-cases --sthresh  0 80  --printoff 
#python investigate_chip.py --dbG --tests vsmany_srule --all-gt-cases --sthresh  0 9001   --printoff 
#python investigate_chip.py --dbG --tests vsmany_srule --all-gt-cases --printoff 

ic --db MOTHERS --tests vsmany_srule --all-gt-cases
ic --db MOTHERS --tests vsmany_nosv --all-gt-cases


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
