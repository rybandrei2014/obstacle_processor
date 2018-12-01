# Obstacle_processor
ROS package for obstacle segmentation in a point cloud scene.
## Installation
* Before you install the package, you have to configure your RGB-D sensor and calibrate it. You can also run this package offline i.e. streaming point cloud ROS topic from .bag file.
* Clone the repository inside **src/** directory of your catkin workspace
```bash
mkdir obstacle_processor
cd obstacle_processor/
git clone name_of_repository
```
* Run CMake to compile source code
```bash
catkin_make
```
* Source your workspace
```bash
source catkin_ws/devel/setup.bash
```
* Setup your robot platform on a ground and remove the all objects in front of it for calibration purposes and run calibration node
```bash
roslaunch obstacle_processor calibration.launch
```
* Now you can run **obstacle_processor** detection algorithm by either of 5 launch commands (two last commands launch **obstacle_processor_node** along with **kinect2_bridge** package from <a href="https://github.com/code-iai/iai_kinect2">iai_kinect2</a> package, but can be replaced for whatever bridge package compatible with your RGB-D sensor that produces point cloud ROS topic)
```bash
roslaunch obstacle_processor obstacle_processor.launch
```
or
```bash
roslaunch obstacle_processor obstacle_processor_rviz.launch
```
or
```bash
roslaunch obstacle_processor obstacle_processor_rviz_debug.launch
```
or
```bash
roslaunch obstacle_processor obstacle_processor_launch_all.launch
```
or
```bash
roslaunch obstacle_processor obstacle_processor_launch_all_rviz.launch
```
