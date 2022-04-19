import os 


path = './labels'
file_list = os.listdir(path)
folder = './labels/'

for name in file_list:
    source = folder + name
    j_name = name.replace('간판_실내안내판_', '')
    dest = folder + j_name
    print(j_name)
    os.rename(source, dest)