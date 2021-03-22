import pandas as pd
import os

df_train = pd.DataFrame(
    columns=['Epoch_no', 'aeroplane', 'bench', 'cabinet', 'car', 'chair', 'display', 'lamp', 'speaker', 'rifle', 'sofa',
             'table', 'telephone', 'watercraft'])
aeroplane = 0
bench = 0
cabinet = 0
car = 0
chair = 0
display = 0
lamp = 0
speaker = 0
rifle = 0
sofa = 0
table = 0
telephone = 0
watercraft = 0


def record_iou_train(epoch, iou):
    global aeroplane, bench, cabinet, car, chair, display, lamp, speaker, rifle, sofa, table, telephone, watercraft

    # Below line for reference
    # df_train.loc[epoch] = [epoch]+[iou['02691156']]+[iou['02828884']]+[iou['02933112']]+[iou['02958343']]+[iou['03001627']]+[iou['03211117']]+[iou['03636649']]+[iou['03691459']]+[iou['04090263']]+[iou['04256520']]+[iou['04379243']]+[iou['04401088']]+[iou['04530566']]

    # If key in dictionary has a value accoiated it with -> update value into datafram else leave it as previous value or 0
    if '02691156' in iou:
        aeroplane = iou['02691156']
    if '02828884' in iou:
        bench = iou['02828884']
    if '02933112' in iou:
        cabinet = iou['02933112']
    if '02958343' in iou:
        car = iou['02958343']
    if '03001627' in iou:
        chair = iou['03001627']
    if '03211117' in iou:
        display = iou['03211117']
    if '03636649' in iou:
        lamp = iou['03636649']
    if '03691459' in iou:
        speaker = iou['03691459']
    if '04090263' in iou:
        rifle = iou['04090263']
    if '04256520' in iou:
        sofa = iou['04256520']
    if '04379243' in iou:
        table = iou['04379243']
    if '04401088' in iou:
        telephone = iou['04401088']
    if '04530566' in iou:
        watercraft = iou['04530566']

    df_train.loc[epoch] = [epoch] + [aeroplane] + [bench] + [cabinet] + [car] + [chair] + [display] + [lamp] + [
        speaker] + [rifle] + [sofa] + [table] + [telephone] + [watercraft]


def saveioufile():

    df_train.to_csv(os.path.join(os.getcwd(), 'IOU-per-Epoch-per-Class.csv'), index=False)

    df_train.to_csv('IOUperEpochClass.csv', index=False)

