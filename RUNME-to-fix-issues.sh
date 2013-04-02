export hs_git_host=git@hyrule.cs.rpi.edu:
#export hs_git_host=https://github.com/Erotemic/

export hs_tpl_repo=tpls-hotspotter

git submodule add $hs_git_host$hs_tpl_repo.git tpl
