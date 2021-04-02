# ----------------------------------------------Import required Modules----------------------------------------------- #

import pandas as pd
import os

# --------------------------------------------Create skeleton dataframes--------------------------------------------- #

df_skeleton_iou = pd.DataFrame(
    columns=['Epoch_no', 'aeroplane', 'bench', 'cabinet', 'car', 'chair', 'display', 'lamp', 'speaker', 'rifle', 'sofa',
             'table', 'telephone', 'watercraft'])

df_skeleton_loss = pd.DataFrame(columns=['Epoch_no', 'Loss'])

# ------------------------------------------Set initial variables to 0------------------------------------------------ #

# Training variables per class
train_aeroplane = 0
train_bench = 0
train_cabinet = 0
train_car = 0
train_chair = 0
train_display = 0
train_lamp = 0
train_speaker = 0
train_rifle = 0
train_sofa = 0
train_table = 0
train_telephone = 0
train_watercraft = 0

# Validation variables per class
val_aeroplane = 0
val_bench = 0
val_cabinet = 0
val_car = 0
val_chair = 0
val_display = 0
val_lamp = 0
val_speaker = 0
val_rifle = 0
val_sofa = 0
val_table = 0
val_telephone = 0
val_watercraft = 0

# Testing variables per class
test_aeroplane = 0
test_bench = 0
test_cabinet = 0
test_car = 0
test_chair = 0
test_display = 0
test_lamp = 0
test_speaker = 0
test_rifle = 0
test_sofa = 0
test_table = 0
test_telephone = 0
test_watercraft = 0


# -----------------------Function to append Training data from each epoch into a row in the CSV----------------------- #

def record_iou_data_train(epoch, iou):

    # IoU values for each class after previous epoch
    global train_aeroplane, train_bench, train_cabinet, train_car, train_chair, train_display, train_lamp 
    global train_speaker, train_rifle, train_sofa, train_table, train_telephone, train_watercraft

    # Check if CSV exists - if yes then open file as dataframe; if not then create dataframe as per skeleton file
    if os.path.isfile(os.path.join(os.getcwd(), 'logs', 'Training-IOU-per-Epoch-per-Class.csv')):
        df_train = pd.read_csv(os.path.join(os.getcwd(), 'logs', 'Training-IOU-per-Epoch-per-Class.csv'))
    else:
        df_train = df_skeleton_iou

    # If key in dictionary has a value accoiated it with -> update value into datafram else leave it as previous value or 0
    if '02691156' in iou:
        train_aeroplane = iou['02691156']
    if '02828884' in iou:
        train_bench = iou['02828884']
    if '02933112' in iou:
        train_cabinet = iou['02933112']
    if '02958343' in iou:
        train_car = iou['02958343']
    if '03001627' in iou:
        train_chair = iou['03001627']
    if '03211117' in iou:
        train_display = iou['03211117']
    if '03636649' in iou:
        train_lamp = iou['03636649']
    if '03691459' in iou:
        train_speaker = iou['03691459']
    if '04090263' in iou:
        train_rifle = iou['04090263']
    if '04256520' in iou:
        train_sofa = iou['04256520']
    if '04379243' in iou:
        train_table = iou['04379243']
    if '04401088' in iou:
        train_telephone = iou['04401088']
    if '04530566' in iou:
        train_watercraft = iou['04530566']

    # Append values into Dataframe
    df_train.loc[epoch] = [epoch] + [train_aeroplane] + [train_bench] + [train_cabinet] + [train_car] + [
        train_chair] + [train_display] + [train_lamp] + [train_speaker] + [train_rifle] + [train_sofa] + [
            train_table] + [train_telephone] + [train_watercraft]
    
    # Save Dataframe as CSV file
    df_train.to_csv(os.path.join(os.getcwd(), 'logs', 'Training-IOU-per-Epoch-per-Class.csv'), index=False)


# ----------------------Function to append Validation data from each epoch into a row in the CSV---------------------- #

def record_iou_data_val(epoch, iou):

    # IoU values for each class after previous epoch
    global val_aeroplane, val_bench, val_cabinet, val_car, val_chair, val_display, val_lamp, val_speaker
    global val_rifle, val_sofa, val_table, val_telephone, val_watercraft

    # Check if CSV exists - if yes then open file as dataframe; if not then create dataframe as per skeleton file
    if os.path.isfile(os.path.join(os.getcwd(), 'logs', 'Validation-IOU-per-Epoch-per-Class.csv')):
        df_val = pd.read_csv(os.path.join(os.getcwd(), 'logs', 'Validation-IOU-per-Epoch-per-Class.csv'))
    else:
        df_val = df_skeleton_iou

    # If key in dictionary has a value accoiated it with -> update value into datafram else leave it as previous value or 0
    if '02691156' in iou:
        val_aeroplane = iou['02691156']
    if '02828884' in iou:
        val_bench = iou['02828884']
    if '02933112' in iou:
        val_cabinet = iou['02933112']
    if '02958343' in iou:
        val_car = iou['02958343']
    if '03001627' in iou:
        val_chair = iou['03001627']
    if '03211117' in iou:
        val_display = iou['03211117']
    if '03636649' in iou:
        val_lamp = iou['03636649']
    if '03691459' in iou:
        val_speaker = iou['03691459']
    if '04090263' in iou:
        val_rifle = iou['04090263']
    if '04256520' in iou:
        val_sofa = iou['04256520']
    if '04379243' in iou:
        val_table = iou['04379243']
    if '04401088' in iou:
        val_telephone = iou['04401088']
    if '04530566' in iou:
        val_watercraft = iou['04530566']

    # Append values into Dataframe
    df_val.loc[epoch] = [epoch] + [val_aeroplane] + [val_bench] + [val_cabinet] + [val_car] + [val_chair] + [
        val_display] + [val_lamp] + [val_speaker] + [val_rifle] + [val_sofa] + [val_table] + [val_telephone] + [val_watercraft]
    
    # Save Dataframe as CSV file
    df_val.to_csv(os.path.join(os.getcwd(), 'logs', 'Validation-IOU-per-Epoch-per-Class.csv'), index=False)


# -------------_---------Function to append Testing data from each epoch into a row in the CSV------------------------ #

def record_iou_data_test(epoch, iou):

    # IoU values for each class after previous epoch
    global test_aeroplane, test_bench, test_cabinet, test_car, test_chair, test_display, test_lamp, test_speaker
    global test_rifle, test_sofa, test_table, test_telephone, test_watercraft

    # Check if CSV exists - if yes then open file as dataframe; if not then create dataframe as per skeleton file
    if os.path.isfile(os.path.join(os.getcwd(), 'logs', 'Testing-IOU-per-Epoch-per-Class.csv')):
        df_test = pd.read_csv(os.path.join(os.getcwd(), 'logs', 'Testing-IOU-per-Epoch-per-Class.csv'))
    else:
        df_test = df_skeleton_iou

    # If key in dictionary has a value accoiated it with -> update value into datafram else leave it as previous value or 0
    if '02691156' in iou:
        test_aeroplane = iou['02691156']
    if '02828884' in iou:
        test_bench = iou['02828884']
    if '02933112' in iou:
        test_cabinet = iou['02933112']
    if '02958343' in iou:
        test_car = iou['02958343']
    if '03001627' in iou:
        test_chair = iou['03001627']
    if '03211117' in iou:
        test_display = iou['03211117']
    if '03636649' in iou:
        test_lamp = iou['03636649']
    if '03691459' in iou:
        test_speaker = iou['03691459']
    if '04090263' in iou:
        test_rifle = iou['04090263']
    if '04256520' in iou:
        test_sofa = iou['04256520']
    if '04379243' in iou:
        test_table = iou['04379243']
    if '04401088' in iou:
        test_telephone = iou['04401088']
    if '04530566' in iou:
        test_watercraft = iou['04530566']

    # Append values into Dataframe
    df_test.loc[epoch] = [epoch] + [test_aeroplane] + [test_bench] + [test_cabinet] + [test_car] + [test_chair] + [
        test_display] + [test_lamp] + [test_speaker] + [test_rifle] + [test_sofa] + [test_table] + [test_telephone] + [test_watercraft]
    
    # Save Dataframe as CSV file
    df_test.to_csv(os.path.join(os.getcwd(), 'logs', 'Testing-IOU-per-Epoch-per-Class.csv'), index=False)


# -------------------------------------Function to save training loss per epoch--------------------------------------- #

def record_training_loss(epoch,train_loss):

    # Check if CSV exists - if yes then open file as dataframe; if not then create dataframe as per skeleton file
    if os.path.isfile(os.path.join(os.getcwd(), 'logs', 'Training-Loss-per-Epoch.csv')):
        df_train_loss = pd.read_csv(os.path.join(os.getcwd(), 'logs', 'Training-Loss-per-Epoch.csv'))
    else:
        df_train_loss = df_skeleton_loss

    # Append data to dataframe
    df_train_loss.loc[epoch] = [epoch] + [train_loss]

    # Save dataframe as CSV file
    df_train_loss.to_csv(os.path.join(os.getcwd(), 'logs', 'Training-Loss-per-Epoch.csv'), index=False)


# -------------------------------------Function to save testing loss per epoch--------------------------------------- #

def record_testing_loss(epoch,test_loss):

    # Check if CSV exists - if yes then open file as dataframe; if not then create dataframe as per skeleton file
    if os.path.isfile(os.path.join(os.getcwd(), 'logs', 'Testing-Loss-per-Epoch.csv')):
        df_test_loss = pd.read_csv(os.path.join(os.getcwd(), 'logs', 'Testing-Loss-per-Epoch.csv'))
    else:
        df_test_loss = df_skeleton_loss

    # Append data to dataframe
    df_test_loss.loc[epoch] = [epoch] + [test_loss]

    # Save dataframe as CSV file
    df_test_loss.to_csv(os.path.join(os.getcwd(), 'logs', 'Testing-Loss-per-Epoch.csv'), index=False)