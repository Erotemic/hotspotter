set sandbox=%hotspotter%\sandbox

call rob flat_dlink "%sandbox%" "%hotspotter%/other/*.py" True 

call rob flat_dlink "%sandbox%" "%hotspotter%/back/*.py" True 

call rob flat_dlink "%sandbox%" "%hotspotter%/front/*.py" True 

call rob flat_dlink "%sandbox%" "%hotspotter%/*.py"
