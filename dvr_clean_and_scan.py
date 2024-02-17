import os
import subprocess
import argparse

# parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("folder", help="the folder to process")
args = parser.parse_args()

# create cleaned folder if it doesn't exist
cleaned_folder = os.path.join(args.folder, "cleaned")
if not os.path.exists(cleaned_folder):
    os.makedirs(cleaned_folder)

# loop through files in folder
for filename in os.listdir(args.folder):
    if filename.endswith(".avi"):
        # construct input and output paths
        input_path = os.path.join(args.folder, filename)
        output_path = os.path.join(cleaned_folder, 'cl_'+filename)
        # check if the file already exists in the cleaned folder
        if os.path.exists(output_path):
            print(f"{filename} already exists in the cleaned folder, skipping...")
        else:
            # clean video using ffmpeg
            # subprocess.call(['ffmpeg', '-i', input_path, '-c:v', 'copy', '-c:a', 'copy', output_path]) # V1 : only copy
            subprocess.call(['ffmpeg', '-i', input_path, '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '30' , output_path]) # V2 reconvert file entirely, increased crf from 21

# create output folder if it doesn't exist
output_folder = os.path.join(args.folder, "output")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# run dvr-scan command on all cleaned video files
for filename in os.listdir(cleaned_folder):
    if filename.endswith(".avi"):
        input_path = os.path.join(cleaned_folder, filename)
        subprocess.call(['dvr-scan', '-i',input_path, '-d', output_folder, '-t', '0.75', '-df', '6', '-l', '2s', '-k','-1']) # adjusted param added -fs and increase -t from 0.75 to 0.80 and df from 6 to 4 : (, '-df', '4', '-fs','5') added -l to have a 2sec period before detecting and added -k to adapt the script to the input resolution
