# Face
**CSE 473/573 Face Detection and Recognition Project.**
#### <font color=red>You can only use opencv 4.5.4 for this project.</font>


**task 1 validation set**
```bash
# Face detection on validation data
python task1.py --input_path validation_folder/images --output ./result_task1_val.json

# Validation
python ComputeFBeta/ComputeFBeta.py --preds result_task1_val.json --groundtruth validation_folder/ground-truth.json
```

**task 1 test set running**

```bash
# Face detection on test data
python task1.py --input_path test_folder/images --output ./result_task1.json
```

**task 2 running**
```bash
python task2.py --input_path faceCluster_5 --num_cluster 5
```

**Pack your submission**
Note that when packing your submission, the script would run your code before packing.
```bash
sh pack_submission.sh <YourUBITName>
```
Change **<YourUBITName>** with your UBIT name.
The resulting zip file should be named **"submission\_<YourUBITName>.zip"**, and it should contain 3 files, named **"result_task1.json"**, **"result_task2.json,"**, and **"UB\_Face.py"**. If not, there is something wrong with your code/filename, please go back and check.

You should only submit the zip file.
