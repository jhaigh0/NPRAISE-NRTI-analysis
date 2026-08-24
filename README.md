# NPRAISE-NRTI-analysis

## Accessing data on IDAaaS

To run the loading code on IDAaaS,

1. Enter `smb://isis.cclrc.ac.uk/shares/` into the file browser search bar to mount the shares dir
2. Create a symlink (mantid does not cope with the path) to `~/nGEM-data/DATA` with the command `ln -s ~/.gvfs/'smb-share:server=isis.cclrc.ac.uk,share=shares'/nGEM-Imaging/DATA ~/nGEM-data/DATA`

You will have to repeat step 1 with each new IDAaaS instance.
