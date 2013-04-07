mkdir $hs/sandbox 
export sandbox=$hs/sandbox
ln -t $sandbox *.py
ln -t $sandbox front/*.py
ln -t $sandbox back/*.py
ln -t $sandbox other/*.py
ln -t $sandbox back/algo/*.py
ln -t $sandbox back/tests/*.py
