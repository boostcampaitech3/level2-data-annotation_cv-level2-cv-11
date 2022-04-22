import os 


path = './images'
file_list = os.listdir(path)
folder = './images/'

for name in file_list:
    source = folder + name
    j_name = name.replace('간판_세로형간판_', '')
    dest = folder + j_name
    # print(j_name)
    os.rename(source, dest)

path = './images'
file_list = os.listdir(path)
folder = './images/'

for name in file_list:
    source = folder + name
    j_name = name.replace('간판_세로형간판_', '')
    dest = folder + j_name
    # print(j_name)
    os.rename(source, dest)