#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import print_function, division
from hsdev import main_api
import sys

dbname = sys.argv[1]
print('cloning: ' + dbname)
clonename = main_api.clone_database(dbname, dryrun=False)
