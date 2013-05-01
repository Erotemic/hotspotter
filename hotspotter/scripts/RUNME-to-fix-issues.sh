#export hs_git_host=git@hyrule.cs.rpi.edu:
#export hs_git_host=https://github.com/Erotemic/
#export hs_tpl_repo=tpl-hotspotter

#TODO: Incorporate into setup.py
git submodule add https://github.com/Erotemic/tpl-hotspotter.git hotspotter/tpl
git submodule update --init
git submodule init 
git submodule update

chmod +x ../tpl/lib/darwin/*.mac
chmod +x ../tpl/lib/linux2/*.ln
chmod +x ../scripts/run-*

python main.py --delete-preferences

#git submodule init
#git submodule add $hs_git_host$hs_tpl_repo.git ./tpl

# To remove submodules
#if ["False" == "True"]
#    export submodulepath=tpl
#    git config -f .git/config --remove-section submodule.$submodulepath
#    git config -f .gitmodules --remove-section submodule.$submodulepath
#    git rm --cached $submodulepath
#    rm -rf $submodulepath
#    rm -rf .git/modules/$submodulepath
#fi
