
# Installation Steps that I took: 
#http://ctrl-dev.com/2012/02/compiling-boost-python-with-mingw/
# Installed MinGW
# 
# Installed Boost.Python
# wget(http://sourceforge.net/settings/mirror_choices?projectname=boost&filename=boost/1.53.0/boost_1_53_0.zip)
# unzip boost_1_53_0.zip C:\boost_1_53_0
# cd C:\boost_1_53_0
# bootstrap.bat mingw
# .\b2 
#
# Then Create a File in your boost directory
# user-config.jam 
'''
import toolset : using ;
using python : 2.7 : 'C:/Python27' : 'C:/Python27/include' : 'C:/Python27/libs' ;
'''
#bjam toolset=gcc link=shared --with-python --user-config=user-config.jam
#
# pyvlfeat has the wrong name for boost_python. Fix it
# Ok, try doing boost with
# bjam toolset=msvc-10.0 link=shared --build-type=complete --with-thread
# bjam toolset=gcc link=shared --build-type=complete --with-thread --with-python --with-system
# 
#import subprocess
#def make_link(target='', dest=''):
    #try: 
        #if exists(dest):
            #print 'Already Exists Error: Destination File: '+dest
            #return
        #if not exists(target):
            #print 'Doesnt Exists Error: Target File: '+dest
            #return
        #if isfile(target):
            #command = 'mklink /H '+dest+' '+target
            #print subprocess.call(command)
        #if isdir(target):
            #command = 'mklink /D '+dest+' '+target
            #print subprocess.call(command)
    #except Exception as ex:
        #print 'Make Link Failed: Error:\n'+str(ex)
        #raise ex
    #make_link(\
        #target=r'C:\boost_1_53_0\stage\lib\libboost_python-mgw46-mt-1_53.dll', 
        #dest=r'C:\boost_1_53_0\stage\lib\boost_python-mt-py26.dll')
    #make_link(\
        #target=r'C:\boost_1_53_0\stage\lib\libboost_python-mgw46-mt-1_53.dll.a', 
        #dest=r'C:\boost_1_53_0\stage\lib\boost_python-mt-py26.dll.a')

try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    from distutils.command.build_py import build_py
    from distutils.command.build_scripts import build_scripts

# http://docs.python.org/2/distutils/setupscript.html
# 
# This isn't working yet, I just took the package from Theno and
# changed some variables. I haven't understood how to work this yet.



#try:
#from setuptools import setup
#except ImportError:



#else:
    #exclude_fixers = ['fix_next', 'fix_filter']
    #from distutils.util import Mixin2to3
    #from lib2to3.refactor import get_fixers_from_package
    #Mixin2to3.fixer_names = [f for f in get_fixers_from_package('lib2to3.fixes')
                             #if f.rsplit('.', 1)[-1] not in exclude_fixers]
    #from distutils.command.build_scripts import build_scripts_2to3 as build_scripts
