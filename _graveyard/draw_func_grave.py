def test_img(index=0):
    import matplotlib.cbook as cbook
    from PIL import Image
    sample_fnames = ['grace_hopper.jpg',
                     'lena.png',
                     'ada.png']
    if index <= len(sample_fnames):
        test_file = cbook.get_sample_data(sample_fnames[index])
    else:
        import load_data2
        chip_dir  = load_data2.DEFAULT+load_data2.RDIR_CHIP
        test_file = chip_dir + '/CID_%d.png' % (1 + index - len(sample_fnames))
    test_img = np.asarray(Image.open(test_file).convert('L'))
    return test_img


