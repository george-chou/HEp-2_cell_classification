#move data to different fileder
import csv
import shutil
import os

def main():
    file_path=('./')
    change_train_path=(file_path+'dataset/tra/')
    change_validation_path=(file_path+'dataset/val/')
    change_test_path=(file_path+'dataset/tes/')
    original_train_path=(file_path+'data/training/')
    original_validation_path=(file_path+'data/validation/')
    original_test_path=(file_path+'data/test/')

    with open('gt_training.csv',"rt", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
        for row in rows:
            if os.path.exists(original_train_path+row[0]+'.png'):
                full_path = original_train_path+row[0]+'.png'
                shutil.move(full_path, change_train_path + row[1] +'/')
            elif os.path.exists(original_validation_path+row[0]+'.png'):
                full_path = original_validation_path+row[0]+'.png'
                shutil.move(full_path, change_validation_path + row[1] +'/')
            elif os.path.exists(original_test_path+row[0]+'.png'):
                full_path = original_test_path+row[0]+'.png'
                shutil.move(full_path, change_test_path + row[1] +'/')
    print('finish trans')

if __name__ == "__main__":
    main()