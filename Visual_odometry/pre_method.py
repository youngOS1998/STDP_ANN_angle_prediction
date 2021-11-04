from Visual_odometry._globals import *


def get_angle_data(csv_name, bTrue = True):
    '''
    this method is used to read the angle data of the running car
    '''
    with open(ANGLE_data_path + ANGLE_data_name) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)

        for row in f_csv:
            print(row[1])