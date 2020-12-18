
### NOTICE (2014-2-27):

HOTSPOTTER IS A LEGACY REPOSITORY FOR THE LATEST AND GREATEST CHECK OUT IBEIS

https://github.com/Erotemic/ibeis


### NOTICE (2020-12-18): 

It looks like the main RPI website has been removed, although it can still be accessed via the wayback machine:
https://web.archive.org/web/20170602093944/http://cs.rpi.edu/hotspotter/


Dropbox should still be hosting the download links: 

* Win32 Installer: https://www.dropbox.com/s/5j1xyx2hq1wzqz2/hotspotter-win32-setup.exe?dl=0 

* OSX Installer: https://www.dropbox.com/s/q0vzz3xnjbxhsda/hotspotter_installer_mac.dmg?dl=0

If you have access to a Linux machine or virtual machine I would recommend using IBEIS (https://github.com/Erotemic/ibeis) over HotSpotter. It's an improved version of the software. If you have linux with a python environment you can simply `pip install ibeis`, and then run `ibeis` to use it. The most recent README has installation and usage instructions. Also note, that while I'm not supporting HotSpotter with software updates (distribution of win32 exes was magic I don't know if I'll ever be able to capture again), I will support the IBEIS software with bug fixes and perhaps updates. 


Known Issues
------------

HotSpotter has known bugs. 


Occasionally it doesn't save a CSV correctly which results in this error:

```
error while blocking gui:ex=AttributeError("'NoneType' object has no attribute'dtype''’,)
```

To fix it you need to open the image and annotation csv files corresponding to your database. There will likely be several lines in the annotation files that reference an image that doesn't exist in the images csv. Removing these lines and re-saving the CSV will resolve the issue. 


If you get a memory error, you are likely running out of RAM because your
database is too big for you computer.



Original README
---------------

Hotspotter is a work in progress (although it is no longer in development), and
getting setup on a new system has not fully been hashed out yet. Its doable.
Best of luck.  Message me on github if you need help. 

This will be open source we have not completely settled on a licence yet. 


Prereqs: 
PyQt4
opencv 2.4.8
the hotspotter repos of hesaff and flann

Quick Instructions: 

If you encounter errors on a Linux or Mac system  before doing anything else
try running: 

`python setup.py fix_issues`

Command Line Instructions: 

To Run Hotspotter from the command line:

`python main.py`

To Run HotSpotter from the command line with a command line interface 

`python main.py --cmd`
