from pathlib import Path
import zipfile as zf
"""
Python script for initial data wrangling.
"""

# global root folder, change to yours so that the data importing works.
root_folder = Path(r"C:\Temp\Data")

# merge the zip file parts into one for extraction:
if not (root_folder / "forecastdemand_nsw.csv").exists():

    # look for the complete zip file, if not merge it together.
    if not (root_folder / "forecastdemand_nsw.csv.zip").exists():
        with open(root_folder / "forecastdemand_nsw.csv.zip", "wb") as outfile:
            for f in root_folder.glob("*part*"):
                    with open(f, "rb") as infile:
                        outfile.write(infile.read())

    with zf.ZipFile(root_folder / "forecastdemand_nsw.csv.zip", "r") as z:
        z.extractall(root_folder)



if not (root_folder / "temperature_nsw.csv").exists():
    with zf.ZipFile(root_folder / "temperature_nsw.csv.zip", "r") as z:
        z.extractall(root_folder)

if not (root_folder / "totaldemand_nsw.csv").exists():
    with zf.ZipFile(root_folder / "totaldemand_nsw.csv.zip", "r") as z:
        z.extractall(root_folder)
