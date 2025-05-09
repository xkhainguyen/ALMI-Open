Using this repo to collect ALMI-X dataset. And we also upload our dataset at <https://huggingface.co/datasets/TeleEmbodied/ALMI-X>.

## 1. Select Motion

### 1.1 Create conda environments
create conda enviroment:
``` bash
conda create -n almi-data_collection python==3.8
conda activate almi-data_collection
cd Data_Collection
pip install -r requirements_dc.txt

conda create -n create_text_annotation python==3.10
conda activate create_text_annotation
cd Data_Collection
pip install -r requirements_text.txt

```

### 1.2 Filter motions that are dominated by upper body movements

We reorganized the format of AMASS dataset and select required parts as:
 - motion_name.pkl: motion name of each motion; in KIT dataset, is like: "0-KIT_[idx]_[motion_simple_description]_[idx]_poses"
   - size: list[num_motions]
 - motion_dof_pos.pkl: dof pos of robot
   - size: list[num_motions] - list[num_frames] - numpy.ndarray[1] - numpy.ndarray[19](the order is following the H1-2 urdf(without 3 wrist joints for each arm)) 
 - motion_rg_pos.pkl: link position of robot
   - size: list[num_motions] - list[num_frames] - numpy.ndarray[1] - numpy.ndarray[20](num_links) - numpy.ndarray[3](global position)
 - motion_root_rot.pkl: robot global orientation
   - size: list[num_motions] - list[num_frames] - numpy.ndarray[1] - numpy.ndarray[4](global orientation)
 - motion_root_vel.pkl: robot global linear velocity
   - size: list[num_motions] - list[num_frames] - numpy.ndarray[1] - numpy.ndarray[3](global linear velocity, x,y,z directions)
 - motion_root_ang_vel.pkl: robot global angular velocity
   - size: list[num_motions] - list[num_frames] - numpy.ndarray[1] - numpy.ndarray[3](global angular velocity, roll,pitch,yaw directions)

Put these files to `Data_Collection/origin_data`.

Filter satisfied motions from original dataset.

You can also download selected motions files at <https://huggingface.co/datasets/TeleEmbodied/ALMI-X/blob/main/select_motions.zip>, and unzip them into `select_data`.

``` bash
cd Data_collection
conda activate almi-data_collection
python select_motion_first.py
```

Then the selected motions' data will be exported to `Data_Collection/select_data` following same format. And a new file named `upper_motion_text` remaining the text description for motion is also in `select_data`

### 1.3 Create text annotation for each motion

``` bash
cd Data_collection
conda activate create_text_annotation
python save_text.py
```

Then the text descriptions will be saved in `./Data_Collection/text_no_expand`, totally 34235 (41x835). And also  `./Data_Collection/cmd/cmd_list.pkl` and `./Data_Collection/npy_name_list.pkl` used for collection trajectory dataset later.


### 1.4 Filter motions by hand
Then filter these 835 motions by hand:
  - delete motions which is illegible or excessive amplitude.
  - the subsequences of the actual completed actions in the original motion sequence are intercepted.

``` bash
cd Data_collection
conda activate almi-data_collection
python ./mujoco/check_motions.py
```

Then there still 680 motions satisfying the requirements. A table file is saved in `./Data_Collection/mujoco/frame_text.csv`, which the format is like:
``` python
264,396,cast a box
123,386,play the guitar using left hand
162,577,wave both
0,616,no
0,387,wave right
0,366,throw left
324,440,cast a box
...
```
where "no" in third column means delete this motion from dataset.

### 1.5 Expand the motions and texts
``` bash
cd Data_collection
conda activate almi-data_collection
python get_motion_nums.py
```

This will create a `Data_Collection/motion_num_dict.pkl` file saving the number that each motion's expanding number. (If motion is deleted, the expanding number is 0).

Then expand the text description
``` bash
cd Data_collection
conda activate almi-data_collection
python ./expand_text.py
```
Then the text total number is: 81549

## 2. Collect Mujoco Datasets

``` bash
cd Data_collection
conda activate almi-data_collection
python ./mujoco/save_trajectory_data.py
```

The the trajectory data will be saved in `./Data_Collection/obs`.