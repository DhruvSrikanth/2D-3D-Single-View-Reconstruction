
# ----------------------------------------------Import required Modules----------------------------------------------- #

import collections

# ----------------------------------------------Utility Functions----------------------------------------------------- #

def model_sort(model_file_paths):
    '''
    Sort saved model file paths.
    :param model_file_paths: list of saved model file paths
    :return: sorted list of saved model file paths
    '''
    epoch_list = [int(model_file_paths[i].split("_")[-1].split(".")[0]) for i in range(len(model_file_paths))]
    model_file_paths_dict = {k: v for (k, v) in zip(epoch_list, model_file_paths)}
    model_file_paths_dict = collections.OrderedDict(sorted(model_file_paths_dict.items()))
    model_file_paths_ = list(model_file_paths_dict.values())
    return model_file_paths_