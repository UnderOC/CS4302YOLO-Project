import shutil
import os


CurDir = os.path.dirname(os.path.abspath(__file__))
RootDir = "/home/wjxie/tnas/pdc/"
print("CurDir:",CurDir)
print("RootDir:",RootDir)

if not os.path.exists(RootDir):
    print("Usage: replace the RootDir to your own path")
    exit(0)

mapping = {
    CurDir+"/native_functions.yaml":RootDir+"pytorch/aten/src/ATen/native/native_functions.yaml", 
    CurDir+"/OurConv2d.cu":RootDir+"pytorch/aten/src/ATen/native/OurConv2d.cu",
}



# Copy the file to the target directory
for key, value in mapping.items():
    print("Copying",key,"to",value)
    shutil.copy(key, value)

print("Done")