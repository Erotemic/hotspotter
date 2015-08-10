'''
def test2(chip, chip_mask):
    im(chip, 1)
    im(chip_mask, 2)

    chip_hsv = cv2.cvtColor(chip, cv2.COLOR_RGB2HSV)
    chip_H = chip_hsv[:, :, 0]
    chip_S = chip_hsv[:, :, 1]
    chip_V = chip_hsv[:, :, 2]

    im(chip_H, 3)
    im(chip_S, 4)
    im(chip_V, 5)

    #chip_H *= chip_mask
    #chip_S *= chip_mask
    chip_V *= chip_mask

    im(chip_V, 6)

    chip_hsv[:, :, 0] = chip_H
    chip_hsv[:, :, 1] = chip_S
    chip_hsv[:, :, 2] = chip_V

    seg_chip = cv2.cvtColor(chip_hsv, cv2.COLOR_HSV2RGB)

    im(seg_chip, 8)
    df2.present()

#if __name__ == '__main__':
    #from multiprocessing import freeze_support
    #freeze_support()
    #print('[segm] __main__ = segmentation.py')
    #from hsviz import draw_func2 as df2
    #df2.reset()
    #import dev
    #main_locals = dev.dev_main()
    #hs = main_locals['hs']
    #cx = 0
    #test(hs, cx)
    ##cx = int(sys.argv[1])
    #exec(df2.present())
'''
