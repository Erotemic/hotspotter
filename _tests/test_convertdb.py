'''
#----- Specific Databases ----
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    helpers.PRINT_CHECKS = True
    if 'paris' in sys.argv:
        convert_from_oxford_style(params.PARIS)
    if 'oxford' in sys.argv:
        convert_from_oxford_style(params.OXFORD)
    if 'wildebeast' in sys.argv:
        wildid_xlsx_to_tables(params.WILDEBEAST)
    if 'toads' in sys.argv:
        wildid_csv_to_tables(params.TOADS)

'''
