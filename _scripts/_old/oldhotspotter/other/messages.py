workflow_help = '''--- Open The Database ---

(File->Open Database)[(Ctrl+O)]
Open a database or select an EMPTY folder as a new database 

(File->Import Images)[Ctrl+I] 
Import images into the database.
  * You may select more than one image at a time.

--- Select the Chips ---

(Actions->Add ROI)[A]
Click two points to select a regions of interest (ROI) in an image.
  * ROIs become Chips
  * (Convenience->Convert All Image To Chips) Adds and ROI to each full image.
  * (Actions->Reselect Orientation)[O] Rotates an image for better matching.
  * (Actions->Reselect ROI)[R] Allows a missed selection to be fixed.
  * (Actions->Remove Chip)[Ctrl+Delete] Deletes a chip
  * (Actions->Next)[N] Moves to the next unidentified chip or unROIed image.

  
--- Discover Matching Animals ---

Select a chip (an ROIed animal) in the image or chip table
(Actions->Query)[Q]
  * You will be brought to the results table. 
    You can edit the Chip Name to mark an animal as identified.
  * When you get a feeling for what score is a sure match try
    (Convenience->Assign Matches Above Threshold) will automatically query 
    each animal in the database and assign matches to the queries which score
    above the threshold. 

--- Displaying Results ---

The Ticker Box in the bottom left corner will change the figure drawn in. 
0 is defaulted to be inside HotSpotter's internal PlotWidget, but any other
will be drawn to a new window. You can resize, zoom in, and save the image. 

(Options -> Toggle Ellipses)[E]
Toggles drawing of the 'HotSpot' regions

(Options -> Toggle Points)[P]
Toggles drawing of the 'HotSpot' points

(Options -> Toggle Plot Widget)
Like the Ticker Box, but the Plot Widget is removed for extra space
* NOTE: You can only select ROIs and Orientation in the PlotWidget *
'''

preference_help = '''
--- Preferences ---
Algorithm Prefs:
    Query: 
        k - The number of matches a hotspot can have
        spatial_thresh - How geometrically consistent matches must be
        method - If COUNT is too inaccurate consider switching to LNRAT
        score - cscore is useful for unknown images nscore is useful otherwise.
        match_threshold - The minimum score to be counted as a match
        min_num_results - Number of results per maximum result per query 
        max_num_results - Number of results per minimum result per query 
        extra_num_results - Extra results for context, not subject to min or max
        
    Chip Preprocessing:
      sqrt_num_pxls - Changes chip size. Decrease if too slow, increase if too
                        inaccurate.
      bilateral_filt_bit   - If on, filters some noise out of chips
      hist_eq_bit          - If on corrects for some global lighting issues
      adapt_hist_eq_bit    - If on corrects for some local lighting issues
      contrast_stretch_bit - If on tries stretches the contrast of the image
    
'''

cmd_help = '''
For those brave enough to run HotSpotter with the --cmd option

Look in hotspotter/Facade.py for all the functions

Format: command [required_args] <optional_args> - description

    logs - writes debugging logs

    print_help - shows this message

    open_db <db_path> - create a new database or open an old one
    save_db - saves your changes

    import_images - select a list of images to add to the database

    selc [cid] - selects and displays the chip-id
    selg [gid] - selects and displays the image-id

    query  - query's the selected chip-id
    rename_cid [new_name] <cid> - changes the selected chip-id's name to [new_name]
                        (YOU WILL NEED QUOTES AROUND THE NAME)
    remove_cid  - deletes the selected chip-id
    reselect_roi - the user reselects the selcted chip-id's roi

    add_chip - the user adds a chip to the selected image-id

    ctbl - prints chip table
    gtbl - prints image table
    ntbl - prints name table

    stat - prints stats: selection mode (image or chip) and selected id

    vd <path> - view current directory
    vdd - view data directory

    select_next - picks one of the following:
    next_empty_image - selects an image without an ROI
    next_uniden_chip - selects a chip without a name
    '''

troubles_help = '''
When in doubt, restart. 

If the images you've imported aren't showing up, you can always re-import the
images in your '<db_dir>/images directory'. 

If something looks corrupted or ROIs are being oddly drawn 
consider deleting your computed directory. 
Run (Convenience->View Internal Directory) and then delete the computed
directory. This will simply cause the program to recompute its data. 
You may have to restart HotSpotter. 

HotSpotter keeps a small set of preference files in your home directory.
These files remember the last database you had open as well as other
preferences. When updating to new versions these can sometimes cause
problems. Deleting the ~/.hotspotter folder may fix some issues. 


If all else fails you can send an email to hotspotter.ir@gmail.com. Please include
a detailed description of the error, what you were doing when it happened, and
the output of the (Convenience->Write Logs) command if possible
'''
