# DVR AVI files auto clean and scan
## Requirements
requires ffmpeg, open cv and dvr-scan to work.

## Description
Python scripts that allows to loop trough a folder of dvr extracted video files (avi format) and clean them if files are broken or corrupted (happens with some cheap dvr) and then run dvr scan to quickly extract moment where movement is detected.


## Usage 

```
python dvr_clean_and_scan.py *path_to_folder*
```
