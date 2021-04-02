# ----------------------------------------------Import required Modules----------------------------------------------- #

import pandas as pd
import os

# --------------------------------------------Create skeleton dataframes--------------------------------------------- #

df_skeleton_iou = pd.DataFrame(
    columns=['Epoch_no', 'aeroplane', 'bench', 'cabinet', 'car', 'chair', 'display', 'lamp', 'speaker', 'rifle', 'sofa',
             'table', 'telephone', 'watercraft'])

df_skeleton_loss = pd.DataFrame(columns=['Epoch_no', 'Loss'])


# --------------------------Function to deal with switch cases to save IoU values per Epoch-------------------------- #

def switch_case_iou(field):
    switcher = {
        1 : 'Training-IOU-per-Epoch-per-Class.csv',
        2 : 'Validation-IOU-per-Epoch-per-Class.csv',
        3 : 'Testing-IOU-per-Epoch-per-Class.csv'
    }
    return switcher.get(field)

# -----------------------------Function to deal with switch case to save Loss per Epoch------------------------------ #

def switch_case_loss(field):
    switcher = {
        1 : 'Training-Loss-per-Epoch.csv',
        2 : 'Testing-Loss-per-Epoch.csv'
    }
    return switcher.get(field)


# -------------------------------------Function to save values of IoU per Epoch------------------------------------- #

def record_iou_data(field, epoch, iou):

    # Field = 1 -> Training
    # Field = 2 -> Validation
    # Field = 3 -> Testing
    # Assign respective field with file to be saved
    save_file = switch_case_iou(field)

    # Check if CSV exists - if yes then open file as dataframe; if not then create dataframe as per skeleton file
    if os.path.isfile(os.path.join(os.getcwd(), 'logs', save_file)):
        df = pd.read_csv(os.path.join(os.getcwd(), 'logs', save_file))
    else:
        df = df_skeleton_iou

    # Below If-statements can be commmented out in case of running over the whole dataset
    # If key in dictionary has a value accoiated it with -> update value into datafram else leave it as previous value or 0
    if '02691156' not in iou:
        iou['02691156'] = 0
    if '02828884' not in iou:
        iou['02828884'] = 0
    if '02933112' not in iou:
        iou['02933112'] = 0
    if '02958343' not in iou:
        iou['02958343'] = 0
    if '03001627' not in iou:
        iou['03001627'] = 0
    if '03211117' not in iou:
        iou['03211117'] = 0
    if '03636649' not in iou:
        iou['03636649'] = 0
    if '03691459' not in iou:
        iou['03691459'] = 0
    if '04090263' not in iou:
        iou['04090263'] = 0
    if '04256520' not in iou:
        iou['04256520'] = 0
    if '04379243' not in iou:
        iou['04379243'] = 0
    if '04401088' not in iou:
        iou['04401088'] = 0
    if '04530566' not in iou:
        iou['04530566'] = 0

    # Append values into Dataframe
    df.loc[epoch] = [epoch] + [iou['02691156']] + [iou['02828884']] + [iou['02933112']] + [iou['02958343']] + [iou['03001627']
    ] + [iou['03211117']] + [iou['03636649']] + [iou['03691459']] + [iou['04090263']] + [iou['04256520']] + [iou['04379243']
    ] + [iou['04401088']] + [iou['04530566']]
    
    # Save Dataframe as CSV file
    df.to_csv(os.path.join(os.getcwd(), 'logs', save_file), index=False)


# ------------------------------------------Function to save Loss per Epoch------------------------------------------ #

def record_loss(field, epoch, loss):

    # Field = 1 -> Training
    # Field = 2 -> Testing
    # Assign respective field with file to be saved
    save_file = switch_case_loss(field)

    # Check if CSV exists - if yes then open file as dataframe; if not then create dataframe as per skeleton file
    if os.path.isfile(os.path.join(os.getcwd(), 'logs', save_file)):
        df = pd.read_csv(os.path.join(os.getcwd(), 'logs', save_file))
    else:
        df = df_skeleton_loss

    # Append data to dataframe
    df.loc[epoch] = [epoch] + [loss]

    # Save dataframe as CSV file
    df.to_csv(os.path.join(os.getcwd(), 'logs', save_file), index=False)