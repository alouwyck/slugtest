import os, shutil

# unzip
os.system('unzip slugtest-master.zip -d .')

# copy package ./slugtest-master/slugtest/ to ./slugtest/

from_dir = "./slugtest-master/slugtest/"
to_dir = "./slugtest/"

os.mkdir(to_dir)

files = os.listdir(from_dir)
for file in files:
  shutil.copyfile(from_dir + file, to_dir + file)
  
shutil.rmtree("./slugtest-master/")