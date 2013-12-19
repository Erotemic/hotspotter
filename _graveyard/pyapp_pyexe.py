
# Old PyExe and PyApp code
# def package_application():
#     write_version_py()
#     # ---------
#     setup_kwargs = get_info_setup_kwarg()
#     # ---------
#     # Force inclusion the modules that may have not been explicitly included
#     includes_modules = MODULES
#     # Import whatever database module you can
#     packed_db_module = False
#     DB_MODULES = ['dbhash', 'gdbm', 'dbm', 'dumbdbm']
#     for dbmodule in DB_MODULES:
#         try:
#             __import__(dbmodule)
#             includes_modules.append(dbmodule)
#         except ImportError:
#             pass
#     # --------
#     # Get Data Files 
#     data_files = get_hotspotter_datafiles()
#     setup_kwargs.update({'data_files' : data_files})
#     # ---------
#     run_with_console = True
#     py2_appexe_universal = {
#             'optimize'     : 0, # 0,1,2
#             'includes'     : includes_modules
#     }
#     # WINDOWS EXECUTABLE SETUP
#     if sys.platform == 'win32':
#         ensure_findable_windows_dlls()
#         # ---------
        
#         # Construct py2exe options
#         py2exe_options = py2_appexe_universal
#         py2exe_options.update({
#             'unbuffered'   : True,
#             'skip_archive' : True, #do not place Python bytecode files in an
#                                 #archive, put them directly in the file system
#             'compressed'   : False, #(boolean) create a compressed zipfile
#             'bundle_files' : 3 #1=all, 2=all-Interpret, 3=dont bundle
#         })
#         setup_options={'py2exe' : py2exe_options}
#         run_cmd = [{'script': 'main.py',
#                     'icon_resources': [(0, 'hsicon.ico')]}]
#         run_type = 'console' if run_with_console else 'windows'
#     # 
#     # MAC APPLICATION SETUP
#     if sys.platform == 'darwin':
#         import py2app
#         # Construct py2app options
#         setup_kwargs.update({'setup_requires':['py2app']})
#         py2app_options = py2_appexe_universal
#         py2app_options.update({
#             'argv_emulation': False,
#             'iconfile':'hsicon.icns',
#             'plist': {'CFBundleShortVersionString':'1.0.0',}
#         })
#         py2app_options.update(py2_appexe_universal)
#         setup_options={'py2app' : py2app_options}
#         run_type = 'app'
#         run_cmd = ['main.py']
#     # add windows/mac stuff to setup keyword arguments
#     setup_kwargs.update({run_type : run_cmd})
#     setup_kwargs.update({'options' : setup_options})
#     # ---------
#     # 
#     # Do actual setup
#     print('Running package setup with args: ')
#     for key, val in setup_kwargs.iteritems():
#         print(key+' : '+repr(val))
#     setup(**setup_kwargs)

#     if sys.platform == 'darwin':
#         subprocess.call(['cp', '-r', 'hotspotter/_tpl/lib/darwin',
#                          'dist/HotSpotter.app/Contents/Resources/lib/'])
