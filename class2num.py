import os
def new_file(file_PATH,old_str,new_str):
    '''
    该函数实现批量读入文件，并寻找替换某个字符串，将修改后的字符串重新写入文件
    file_PATH:主文件夹路径
    folder_path：子文件夹路径
    file_path：文件路径
    old_str:待修改的字符串
    new_str：修改后的字符串
    '''
    folder_list=os.listdir(file_PATH)#文件夹下的子文件夹列表
    for folder in folder_list:
        folder_path=os.path.join(file_PATH,folder)#子文件夹路径
        file_list=os.listdir(folder_path)#子文件夹下的文件列表
        for file in file_list:
            file_path=os.path.join(folder_path,file)#文件路径
            with open(file_path, "r") as f:  # 以只读方式打开文件
                data = f.read()  # 读取文件，读取为一个字符串
                str_replace = data.replace(old_str,new_str)#将字符串中的某个字符进行替换
                with open(file_path, "w") as f:#重新打开文件，选择写入模式
                    f.write(str_replace)      # 将修改后的字符串重新写入文件
#函数执行

file_PATH='./data/sdd_师兄/'
path_list = ['train','val','test']
# old_str=['Pedestrian','Skater','Biker','Car','Cart','Bus']
# new_str=['0','1','2','3','4','5']
old_str = '3t'
new_str = '4'
# for i in range(len(path_list)):
#     file = file_PATH + path_list[i]
# for j in range(len(old_str)):
new_file(file_PATH=file_PATH,old_str=old_str,new_str=new_str)
