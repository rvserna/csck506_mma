''' Extract the files from the ZIP folder '''

import zipfile

zip_path = "C:/my-project/csck506_mma/csck506_mma/archive.zip"
extract_to = "C:/my-project/csck506_mma/csck506_mma/archive"

# Extract files
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("The files have been successfully extracted")
