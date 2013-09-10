mkdir build
cd build 
cmake ..
make



saliency D:\data\work\HSDB_zebra_with_mothers\images\Nid-01_106--Cid-mom-01_106.tif

cd C:\Users\joncrall\code\saliency-map\src


python -c '
import os
dirs = os.listdir("D:\data\work\HSDB_zebra_with_mothers\images")
for imgname in dirs:
    os.system("python main.py "+os.path.join("D:/data/work/HSDB_zebra_with_mothers/images/",imgname))
    '


Just do a gaussian weighting on the important points as selected by a host of trained scientists. 
Detect those points. 
    
    #D: \data\work\HSDB_zebra_with_mothers\images\Nid-01_106--Cid-mom-01_106.tif")
