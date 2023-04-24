import os

data_path = "T91"
img_names = os.listdir(data_path)

list_file = open('my_dataset.txt', w)
for img_name in img_names:
    