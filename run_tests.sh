#python investigate_chip.py --dbG --tests test-cfg-vsmany-3 --all-gt-cases --sthresh 30 80  --printoff 
#python investigate_chip.py --dbG --tests test-cfg-vsmany-3 --all-gt-cases --sthresh 20 80  --printoff 
#python investigate_chip.py --dbG --tests test-cfg-vsmany-3 --all-gt-cases --sthresh 10 80  --printoff 
#python investigate_chip.py --dbG --tests test-cfg-vsmany-3 --all-gt-cases --sthresh  0 80  --printoff 
#python investigate_chip.py --dbG --tests test-cfg-vsmany-3 --all-gt-cases --sthresh  0 9001   --printoff 
#python investigate_chip.py --dbG --tests test-cfg-vsmany-3 --all-gt-cases --printoff 

#--nocache-query
#--nocache-query 
#--nocache-query 
#--nocache-query 
alias icG='python investigate_chip.py --dbG'
alias icM='python investigate_chip.py --dbM'

icG --tests test-cfg-vsmany-3 --all-gt-cases --sthresh 10 80   --printoff | tail
icG --tests test-cfg-vsmany-3 --all-gt-cases --sthresh  0 80   --printoff | tail
icG --tests test-cfg-vsmany-3 --all-gt-cases --sthresh  0 100  --printoff | tail
icG --tests test-cfg-vsmany-3 --all-gt-cases --sthresh 10 100  --printoff | tail
icG --tests test-cfg-vsmany-3 --all-gt-cases --sthresh 10 150  --printoff | tail
icG --tests test-cfg-vsmany-3 --all-gt-cases --sthresh  0 9001 --printoff | tail

icG --tests kpts-scale --sthresh  0 9001 --printoff | tail

#python investigate_chip.py --dbG --histid 4 5 7 8 10 11 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46

#python investigate_chip.py --dbG --r 0 --c 0 1 2 3 --tests test-cfg-vsmany-3 --histid 4 
#python investigate_chip.py --dbG --r 0 --c 0 1 2 3 --tests test-cfg-vsmany-3 --histid 5 


# TODO: In show match annote
# click a keypoint match to see it in detail
# click a keypoint on the query to see the image that it matched to
