import os, shutil

# unzip
os.system('unzip slugtest-master.zip -d .')

# copy package ./slugtest-master/slugtest/ to ./slugtest/
parent_dir = "./slugtest-master/"
from_dir = "./slugtest-master/slugtest/"
to_dir = "./slugtest/"

os.mkdir(to_dir)

files = os.listdir(from_dir)
for file in files:
  shutil.copyfile(from_dir + file, to_dir + file)

# copy modules from ./slugtest-master/ to ./
files = os.listdir(parent_dir)
for file in files:
  if file[-3:len(file)] == '.py':
    shutil.copyfile(parent_dir + file, "./" + file)

# delete parent dir
shutil.rmtree(parent_dir)
