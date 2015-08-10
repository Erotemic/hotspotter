if __name__ == '__main__':
    import guitools
    import Preferences
    import DataStructures as ds

    class Pref2(Pref.Pref):
        def __init__(self):
            super(Pref2, self).__init__()
            self.prop1 = 'hello'
            self.prop2 = None
            self.prop3 = 123
            self.prop4 = True
            self.prop5 = 0
            self.prop6 = 0.0
            self.prop7 = [1, 2, 3, 4]
            self.prop8 = (1, 2, 3, 4,)
            self.prop9 = Pref.Pref()
            self.prop9.subprop = ''

    guitools.configure_matplotlib()
    app, is_root = guitools.init_qtapp()
    backend = None
    #guitools.make_dummy_main_window()
    prefs = Preferences.Pref()
    r = prefs
    r.a = Preferences.Pref()
    r.b = Preferences.Pref('pref value 1')
    r.c = Preferences.Pref('pref value 2')
    r.a.d = Preferences.Pref('nested1')
    r.a.e = Preferences.Pref()
    r.a.f = Preferences.Pref('nested3')
    r.a.e.g = Preferences.Pref('nested4')

    pref2 = Pref2()

    r.pref2 = pref2
    chip_cfg = ds.make_chip_cfg()
    r.chip_cfg = chip_cfg
    #feat_cfg = ds.make_feat_cfg()
    #r.feat_cfg = feat_cfg
    #query_cfg = ds.make_vsmany_cfg()
    #r.query_cfg = query_cfg

    print(prefs)
    prefWidget = prefs.createQWidget()
    guitools.run_main_loop(app, is_root, backend)
