        import scripts
        import generate_training
        import sys

        def do_encounters(seconds=None):
            if not 'seconds' in vars() or seconds is None:
                seconds = 5
            scripts.rrr()
            do_enc_loc = scripts.compute_encounters(hs, back, seconds)
            return do_enc_loc

        def do_extract_encounter(eid=None):
            #if not 'eid' in vars() or eid is None:
            #eid = 'ex=269_nGxs=21'
            eid = 'ex=61_nGxs=18'
            scripts.rrr()
            extr_enc_loc = scripts.extract_encounter(hs, eid)
            export_subdb_locals = extr_enc_loc['export_subdb_locals']
            return extr_enc_loc, export_subdb_locals

        def do_generate_training():
            generate_training.rrr()
            return generate_training.generate_detector_training_data(hs, (256, 448))

        def do_import_database():
            scripts.rrr()
            #from os.path import expanduser, join
            #workdir = expanduser('~/data/work')
            #other_dbdir = join(workdir, 'hsdb_exported_138_185_encounter_eid=1 nGxs=43')

        def vgd():
            return generate_training.vgd(hs)

        #from PyQt4.QtCore import pyqtRemoveInputHook
        #from IPython.lib.inputhook import enable_qt4
        #pyqtRemoveInputHook()
        #enable_qt4()

