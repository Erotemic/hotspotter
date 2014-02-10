def get_common_paths(output_path_list):
    # Takes a list of paths and extracts the common relative paths
    dir_list = [dirname(fpath) for fpath in output_path_list]
    fname_list = [split(fpath)[1] for fpath in output_path_list]
    unique_dirs = list(set(dir_list))
    # this checks exists quickly, but only works if everything is in the
    # same directory. It would be better if an output directory was
    # specified instead. Then this can be even faster as well as elegant.
