workflow_help = '''
    + Open a database with 'File->Open Database'
       A database is a folder.It can be:
           a StripSpotter database (look for a folder with SightingData.csv)
           a HotSpotter database, (look for a folder with an images directory)
           an empty folder, (this will create a new HotSpotter database) 

    + 'File->import_images' will add one or more images to the database
       (Remember to save after doing this, or you will have to do it again!)

    + Select an image from the image table. 

    + Click Add ROI, and click two points on the image 
        (this will be the bounding box)
        (make sure to get ALL of the animal in the bounding box.)
        (This will add a new chip to the chip table)

    + If you made a mistake use Reselect ROI to do just that. 

    + Select a chip from the chip table by clicking it. 

    + Click query to find probable identity matches in the database. 

    + Edit the chip's name in the table to rename it.
    
    + File->Save Database saves any changes. REMEMBER TO SAVE!
       (This also writes flat_table.csv to the data directory and can be freely used and edited)
       (Use 'View->Open Data Directory' to qickly access it)
'''

cmd_help = '''
HotSpotter - Python Version - PreAlpha

General Workflow: 
    + Use open_db to open or create a new database 
       (typically this is a directory with an images folder
        HotSpotter should read older stripe-program formats
        a HotSpotter-Python database will have a .hs_internals
        folder next to the images folder)
    + Use import_images to add images to the database
    + Use selg to select an image
    + Use add_roi to add a chip to the image
        ( the figure you need to click on is in another window, 
         dont click on the image that shows up in the console) 
    + Use selc to select a chip
    + Use query to perform a seach
    + Use rename to manage the names of animals
    + save_db saves your changes
       (This will also save a flat_table.csv file to the database
       folder. You can quickly access this using the command vdd)

General Tips: 
    * Pressing the up arrow on your keyboard reissues the last command
    * Pressing tab on the keyboard will autocomplete a command
    * HotSpotter runs in an IPython Environment, you have the power of 
      the python scripting language at your fingertips.
    * SAVE OFTEN! THIS IS A PRE-ALPHA!

Format: command [required_args] <optional_args> - description

    print_help - shows this message

    open_db <db_path> - create a new database or open an old one
    save_db - saves your changes

    import_images - select a list of images to add to the database

    selc [cid] - selects and displays the chip-id
    selg [gid] - selects and displays the image-id

    query  - query's the selected chip-id
    rename [new_name] - changes the selected chip-id's name to [new_name]
                        (YOU WILL NEED QUOTES AROUND THE NAME)
    remove_chip  - deletes the selected chip-id
    reselect_roi - the user reselects the seelcted chip-id's roi

    add_roi - the user adds a chip to the selected image-id

    ctbl - prints chip table
    gtbl - prints image table
    ntbl - prints name table

    stat - prints stats: selection mode (image or chip) and selected id

    vd <path> - view current directory
    vdd - view data directory

    next_empty_image - selects an image without an ROI
    next_uniden_chip - selects a chip without a name
    '''

troubles_help = '''
This is an Alpha Release of HotSpotter. You may encounter some errors. 

I will refer to your current database directory as <db_dir>. Commands
and directory-paths will be put in single quotes: ''

If the program freezes, see if you can still enter the 'save_database()'
command in the IPython command window. 

The program execution can be restarted by pressing 'Ctrl .' in the 
IPython command window and entering the command '%run main.py'

If the images you've imported aren't showing up, you can always re-import
the images in your '<db_dir>/images directory'. 

If you think something was corrupted or ROIs are being drawn weird, 
you may want to consider recomputing the information. This can be 
done by deleting the files in the computed directory: 
    <db_dir>/.hs_internals/computed 
You may need to restart HotSpotter. You can recompute everything
by running a query

As an open source python project, you have the same potential to fix
bugs as the developer does. If you are tech-savey you can edit the source
code to add features that you like or fix anoying bugs. The 'logs' command will
print out a detailed debug report of the last few things the program has done.
'write_logs' will output this to a file and display it in a text editor. 

Some preferences aren't exposed to the user but still work. They are stored
in a python dictionary. Type 'print hs.prefs' to see what exists. 

If all else fails you can send an email to hotspotter.ir@gmail.com 
Please include a detailed description of the error, what you were 
doing when it happened, and the file dumped by the write_logs function if possible.
'''


gui_help = '''
The HotSpotter GUI is not yet fully developed. 

Currently the chip and image tabel will be displayed on the left. 
Clicking on an item will select the chip or image. 

The current chip and image selection are shown in the bottom of the gui. 

Add ROI will add a new chip to the selected image

Reselect ROI will let the user redraw an ROI for a query. (Make sure you save)

Query will query the selected chip against the database. 

All progress will be indicated in the IPython command window.
'''
