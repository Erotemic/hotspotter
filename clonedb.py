#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import print_function, division
from hsdev import test_api
import sys

dbname = sys.argv[1]
print('cloning: ' + dbname)
clonename = test_api.clone_database(dbname, dryrun=False)
