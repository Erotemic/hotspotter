'''
if __name__ == '__main__':
    multiprocessing.freeze_support()
    #import multiprocessing
    #np.set_printoptions(threshold=5000, linewidth=5000)
    #print('[dev]-----------')
    #print('[dev] main()')
    #df2.DARKEN = .5
    #main_locals = dev.dev_main()
    #exec(helpers.execstr_dict(main_locals, 'main_locals'))

    if sys.argv > 1:
        import sys
        path = params.DEFAULT
        db_version = get_database_version(path)
        print('db_version=%r' % db_version)
        if not db_version is None:
            db_stats = DatabaseStats(path, db_version, params.WORK_DIR)
            print_database_stats(db_stats)
        sys.exit(0)

    # Build list of directories with database in them
    root_dir_list = [
        params.WORK_DIR,
        params.WORK_DIR2
    ]
    DO_EXTRA = True  # False
    if sys.platform == 'linux2' and DO_EXTRA:
        root_dir_list += [
            #'/media/Store/data/raw',
            #'/media/Store/data/gold',
            '/media/Store/data/downloads']

    # Build directory statistics
    dir_stats_list = [DirectoryStats(root_dir) for root_dir in root_dir_list]

    # Print Name Stats
    print('\n\n === Num File Stats === ')
    for dir_stats in dir_stats_list:
        print('--')
        print(dir_stats.print_db_stats())

    # Print File Stats
    print('\n\n === All Info === ')
    for dir_stats in dir_stats_list:
        print('--' + dir_stats.name())
        dir_stats.print_databases(' * ')

    print('\n\n === NonDB Dirs === ')
    for dir_stats in dir_stats_list:
        print('--' + dir_stats.name())
        dir_stats.print_nondbdirs()

    # Print File Stats
    print('\n\n === Num File Stats === ')
    for dir_stats in dir_stats_list:
        print('--')
        print(dir_stats.num_files_stats())
'''
