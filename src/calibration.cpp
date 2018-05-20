#include <ros/ros.h>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/conversions.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/angles.h>
#include <pcl/common/transforms.h>
#include <pcl/ModelCoefficients.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <fstream>
#include <boost/make_shared.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/console/time.h>


typedef pcl::PointXYZRGB Point;
bool flag;

class Calibrator
{
  ros::NodeHandle nh;
  ros::Subscriber cloud_sub;
public:

  Calibrator(ros::NodeHandle nh_): nh(nh_)
  {
    cloud_sub = nh.subscribe("/kinect2/qhd/points", 1,
                             &Calibrator::cloud_cb, this);
  }

  void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &input)
  {
    pcl::PointCloud<Point>::Ptr cloud_input(new pcl::PointCloud<Point>);
    pcl::PointCloud<Point>::Ptr cloud_filtered(new pcl::PointCloud<Point>);
    pcl::PointCloud<Point>::Ptr cloud_downsampled(new pcl::PointCloud<Point>);

    pcl::fromROSMsg(*input, *cloud_input);

    pcl::PassThrough<Point> pass_filter;
    pass_filter.setInputCloud(cloud_input);
    pass_filter.setFilterFieldName("z");
    pass_filter.setFilterLimits(0.0, 2.7);
    pass_filter.filter(*cloud_filtered);

    pcl::VoxelGrid<Point> voxel;
    voxel.setInputCloud(cloud_filtered);
    voxel.setLeafSize(0.01, 0.01, 0.01);
    voxel.setDownsampleAllData(true);
    voxel.filter(*cloud_downsampled);

    pcl::ModelCoefficients::Ptr coefficients_floor(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_floor(new pcl::PointIndices);
    pcl::SACSegmentation<Point> seg;
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setOptimizeCoefficients(true);
    seg.setMaxIterations(500);
    seg.setDistanceThreshold(0.02);
    seg.setInputCloud(cloud_downsampled);
    seg.segment(*inliers_floor, *coefficients_floor);
    Eigen::Vector3f XZ_norm(0.0, -1.0, 0.0);
    Eigen::Vector3f floor_norm(coefficients_floor->values[0], coefficients_floor->values[1],
        coefficients_floor->values[2]);
    float angle = acos(floor_norm.dot(XZ_norm));
    std::fstream f;
    f.open("cal_data.dat");
    // File has not been created
    if (!f.is_open())
    {
      f.open("cal_data.dat", std::ios::out|std::ios::in|std::ios::trunc);
      if (!f.is_open())
      {
        std::cerr << "Error while creating calibration file.\n";
        f.close();
        return;
      }
      f << angle;
      std::cerr << "File has been created.\n";
    }
    else // File already exists
    {
      std::cerr << "Changing angle value in calibration file.\n";
      f.clear();
      f.seekg(0);
      f << angle;
    }
    f.close();
    flag = false;
  }
};

int main(int argc, char **argv)
{
  flag = true;
  ros::init(argc, argv, "calibration_node");
  ros::NodeHandle nh;
  Calibrator cal(nh);
  while (flag) {
    ros::spinOnce();
  }
}
