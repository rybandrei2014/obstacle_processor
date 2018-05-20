#include <iostream>
#include <ros/ros.h>
#include <tf/transform_listener.h>
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
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/keypoints/harris_3d.h>

#include <pcl/point_types.h>
#include <pcl/console/time.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/region_growing.h>
#include <cstdlib>
#include <ctime>
#include <string>
#include <boost/make_shared.hpp>
#include <boost/assign/std/vector.hpp>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/filters/project_inliers.h>
#include <obstacle_processor/processed_box_data.h>
#include <obstacle_processor/processed_cylinder_data.h>
#include <obstacle_processor/processed_undefined_data.h>
#include <obstacle_processor/processed_wall_like_object_data.h>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pcl/geometry/eigen.h>
#include <pcl/sample_consensus/ransac.h>
#include <fstream>
#include <pcl/surface/gp3.h>

using namespace boost::assign;
using namespace std;
using namespace obstacle_processor;
using namespace Eigen;
using namespace pcl;
typedef PointXYZRGB Point;
typedef Normal Normal;
typedef sensor_msgs::PointCloud2 PCloud2;
typedef ModelCoefficients MCoef;
typedef PointIndices PIndices;
double plane_seg_thresh, outliers_thresh, cluster_tol, point_min_z, point_max_z, dist_weight, thresh2, tolerance, dist_weight_circ, dist_thresh_circ, height_thresh_min, height_thresh_max;
float min_z, max_z, voxel_size, alfa;
bool rviz_out, cmd_out, transform_to_base_link;
string pc_topic;
int min_cluster, max_cluster, iter, meanK, K_search, iter2, iter_circ;
tf::TransformListener* tf_listener;

class ObstacleProcessor
{
  ros::NodeHandle nh;
  ros::Subscriber cloud_sub_;
  ros::Publisher pub_clusters, pub_cylinder, pub_front_plane, pub_projected, pub_projected_flat, pub_hull, pub_box, pub_undefined, pub_wall_like_object, pub_floor, pub_cloud, pub_top_planes, pub_backup;
  processed_box_data box_data;
  processed_cylinder_data cylinder_data;
  processed_undefined_data undefined_data;
  processed_wall_like_object_data wall_like_object_data;
  pcl::console::TicToc tt;

public:
  vector<double> transformToBaseLink(bool transform_to_base_link, Vector3f& centroid, geometry_msgs::PointStamped kinect_centroid)
  {
    if (transform_to_base_link)
    {
      geometry_msgs::PointStamped base_link_centroid;
      kinect_centroid.point.x = centroid(0);
      kinect_centroid.point.y = centroid(1);
      kinect_centroid.point.z = centroid(2);
      try
      {
        tf_listener->transformPoint("base_link", ros::Time(0), kinect_centroid, "camera", base_link_centroid);
      }
      catch (tf::TransformException& exc)
      {
        ROS_ERROR("    ! Error during transformation");
        cerr << exc.what() << endl;
      }
      centroid << base_link_centroid.point.x, base_link_centroid.point.y, base_link_centroid.point.z;
    }
    vector<double> centroid_vector(centroid.data(), centroid.data() + centroid.rows() * centroid.cols());
    return centroid_vector;
  }

  vector<double> projectAndProcess(bool hasTopPlaneParallel, bool& isCylinder, PointCloud<Point>::Ptr cloud,
                                   MCoef::Ptr projection_plane, PointCloud<Point>::Ptr project_all,
                                   PointCloud<Point>::Ptr project_flat_all,
                                   PointCloud<Point>::Ptr convex_hull_all, Vector3f& centroid)
  {
    vector<double> dims(2,0);
    PointCloud<Point>::Ptr cloud_project(new PointCloud<Point>), cloud_project_flat(new PointCloud<Point>),
                           cloud_convex_hull(new PointCloud<Point>);
    Point search_point;
    KdTreeFLANN<Point> search_tree;
    ProjectInliers<Point> project;
    ConvexHull<Point> convex_hull;
    Quaternionf quart;
    Affine3f transform(Affine3f::Identity());
    Matrix3f rot, inv_rot;
    Vector3f projection_plane_normal(projection_plane->values[0], projection_plane->values[1], projection_plane->values[2]);
    double min_area = 100000000;
    double area = 0;

    if (cmd_out) {
      ROS_INFO("Creating hull object for top box plane...");
      tt.tic();
    }
    // Project top plane into 2d, using floor model coefficients
    project.setModelType(SACMODEL_PLANE);
    project.setInputCloud(cloud);
    project.setModelCoefficients(projection_plane);
    project.filter(*cloud_project);
    if (rviz_out) *project_all += *cloud_project;

    // Rotate plane so that normal vector of projection is collinear with XY plane's normal
    quart.setFromTwoVectors(projection_plane_normal, Vector3f::UnitZ());
    Matrix3f plane_rot = quart.toRotationMatrix();
    Matrix3f inv_plane_rot = plane_rot.inverse();
    transform.rotate(plane_rot);
    transformPointCloud(*cloud_project, *cloud_project_flat, transform);
    if (rviz_out) *project_flat_all += *cloud_project_flat;

    // Find the convex hull
    convex_hull.setInputCloud(cloud_project_flat->makeShared());
    convex_hull.setDimension(2);
    convex_hull.reconstruct(*cloud_convex_hull);
    int top_plane_hull_points = cloud_convex_hull->points.size();
    if (cmd_out) ROS_INFO("Done: %3.3f ms", tt.toc());

    // Searching for the minimum area for a given set of points to find dimensions
    if (cmd_out) {
      ROS_INFO("Searching the minimum area for a given set of points...");
      tt.tic();
      ROS_INFO("Top plane cloud hull: %3d", top_plane_hull_points);
    }
    for (size_t i=0; i<top_plane_hull_points; i++)
    {
      // For each pair of hull points, determine the angle to rotate coordinate axes
      PointCloud<Point>::Ptr transformed_hull(new PointCloud<Point>);

      double delta_y = cloud_convex_hull->points[(i+1)%top_plane_hull_points].y - cloud_convex_hull->points[i].y;
      double delta_x = cloud_convex_hull->points[(i+1)%top_plane_hull_points].x - cloud_convex_hull->points[i].x;
      double delta_l = sqrt((delta_x*delta_x) + (delta_y*delta_y));
      double sin_fi = delta_y / delta_l;
      double cos_fi = delta_x / delta_l;

      // Build rotation matrix from change of basis
      rot(0,0) = cos_fi;
      rot(0,1) = sin_fi;
      rot(0,2) = 0.0;
      rot(1,0) = -sin_fi;
      rot(1,1) = cos_fi;
      rot(1,2) = 0.0;
      rot(2,0) = 0.0;
      rot(2,1) = 0.0;
      rot(2,2) = 1.0;

      inv_rot = rot.inverse();

      double x_min = 1000.0;
      double x_max = -1000.0;
      double y_min = 1000.0;
      double y_max = -1000.0;
      for (size_t i=0; i<top_plane_hull_points; i++)
      {
        Point p;
        p.getVector3fMap() = rot * cloud_convex_hull->points[i].getVector3fMap();
        transformed_hull->push_back(p);
        if (p.x < x_min) x_min = p.x;
        if (p.x > x_max) x_max = p.x;
        if (p.y < y_min) y_min = p.y;
        if (p.y > y_max) y_max = p.y;
      }
      area = fabs(x_max-x_min)*fabs(y_max-y_min);

      if (area < min_area)
      {
        // raw_box_data[0] length of top plane
        dims[0] = fabs(x_max - x_min);

        // raw_box_data[1] width of top plane
        dims[1] = fabs(y_max - y_min);

        // Check an assumption on a long edge
        if (dims[0] < dims[1])
        {
          double temp = dims[0];
          dims[0] = dims[1];
          dims[1] = temp;
        }

        // Find the center of a box in kinect frame
        if (hasTopPlaneParallel)
        {
          centroid(0) = (x_max+x_min) / 2.0;
          centroid(1) = (y_max+y_min) / 2.0;
          centroid(2) = transformed_hull->points[0].z;
          centroid = inv_rot * centroid;
        }
        min_area = area;
      }
    }

    if (cmd_out) ROS_INFO("Done: %3.3f ms", tt.toc());
    search_tree.setInputCloud(cloud_convex_hull);
    double rad = (dims[1] + dims[0]) / 4;
    double dist;
    double devs;

    if (hasTopPlaneParallel)
    {
      search_point.x = centroid(0);
      search_point.y = centroid(1);
      search_point.z = centroid(2);
    }
    else
    {
      computeCentroid(*cloud_convex_hull, search_point);
    }
    search_point.r = 255;
    search_point.g = 0;
    search_point.b = 0;

    for (int i=0; i<top_plane_hull_points; i++)
    {
      dist = euclideanDistance(cloud_convex_hull->points[i], search_point);
      devs += fabs(rad - dist);
    }
    devs /= top_plane_hull_points;
    cloud_convex_hull->points.push_back(search_point);
    centroid = inv_plane_rot * centroid;
    if (rviz_out) *convex_hull_all += *cloud_convex_hull;
    isCylinder = devs < 0.01? true: false;
    return dims;
  }

  vector<double> findMinMax3D(PointCloud<Point>::Ptr cloud_in)
  {
    vector<double> data(6,0);
    data[0] = 1000;
    data[1] = 1000;
    data[2] = 1000;
    data[3] = -1000;
    data[4] = -1000;
    data[5] = -1000;

    for (size_t i=0; i < cloud_in->points.size(); i++)
    {
      if(cloud_in->points[i].x <= data[0])
        data[0] = cloud_in->points[i].x;
      else if(cloud_in->points[i].y <= data[1] )
        data[1] = cloud_in->points[i].y;
      else if(cloud_in->points[i].z <= data[2] )
        data[2] = cloud_in->points[i].z;
      else if(cloud_in->points[i].x >= data[3] )
        data[3] = cloud_in->points[i].x;
      else if(cloud_in->points[i].y >= data[4] )
        data[4] = cloud_in->points[i].y;
      else if(cloud_in->points[i].z >= data[5] )
        data[5] = cloud_in->points[i].z;
    }
    return data;
  }

  float angleBetweenPlanes(Vector3f plane1_norm, Vector3f plane2_norm)
  {
    float angle = acos(plane1_norm.dot(plane2_norm)) / M_PI*180;
    if (cmd_out) ROS_INFO("# Plane 1 normal: (%3.3f, %3.3f, %3.3f)\n# Plane 2 normal: (%3.3f, %3.3f, %3.3f)\n# Angle between planes: %3.3f deg", plane1_norm(0), plane1_norm(1), plane1_norm(2), plane2_norm(0), plane2_norm(1), plane2_norm(2), angle);
    return angle;
  }

  bool isPerpendicularToPlane(bool wallDetection, Vector3f plane1_norm, Vector3f plane2_norm, float& angleY)
  {
    Vector3f Y_vector(0.0, -1.0, 0.0);
    plane1_norm(2) = fabs(plane1_norm(2));
    plane2_norm(2) = fabs(plane2_norm(2));
    float angle = this->angleBetweenPlanes(plane1_norm, plane2_norm);
    float angleY_temp = this->angleBetweenPlanes(plane1_norm, Y_vector);
    angleY = wallDetection? angleY_temp: (angleY_temp < 45? angleY_temp: fabs(90 - angleY_temp));
    if (((fabs(angle) < 90 + tolerance) && (fabs(angle) > 90 - tolerance)))
      return true;
    return false;
  }

  bool isTopParallelPlane(MCoef::Ptr plane1, MCoef::Ptr plane2, double height, double z_dim)
  {
    Vector3f plane1_normal(plane1->values[0], plane1->values[1], plane1->values[2]);
    Vector3f plane2_normal(plane2->values[0], plane2->values[1], plane2->values[2]);
    float angle = this->angleBetweenPlanes(plane1_normal, plane2_normal);
    bool result = false;
    if (fabs(angle) < tolerance || fabs(angle-180) < tolerance)
    {
      result = true;
      if (height / z_dim < 0.5)
      {
        result = false;
      }
    }
    return result;
  }

  void setParams(const ros::NodeHandle& nh)
  {
    nh.getParam("/obstacle_processor_node/min_z", min_z);
    nh.getParam("/obstacle_processor_node/max_z", max_z);
    nh.getParam("/obstacle_processor_node/iter", iter);
    nh.getParam("/obstacle_processor_node/voxel_size", voxel_size);
    nh.getParam("/obstacle_processor_node/cluster_tol", cluster_tol);
    nh.getParam("/obstacle_processor_node/min_cluster", min_cluster);
    nh.getParam("/obstacle_processor_node/max_cluster", max_cluster);
    nh.getParam("/obstacle_processor_node/plane_seg_thresh", plane_seg_thresh);
    nh.getParam("/obstacle_processor_node/outliers_thresh", outliers_thresh);
    nh.getParam("/obstacle_processor_node/meanK", meanK);
    nh.getParam("/obstacle_processor_node/point_min_z", point_min_z);
    nh.getParam("/obstacle_processor_node/point_max_z", point_max_z);
    nh.getParam("/obstacle_processor_node/dist_weight", dist_weight);
    nh.getParam("/obstacle_processor_node/iter2", iter2);
    nh.getParam("/obstacle_processor_node/thresh2", thresh2);
    nh.getParam("/obstacle_processor_node/tolerance", tolerance);
    nh.getParam("/obstacle_processor_node/K_search", K_search);
    nh.getParam("/obstacle_processor_node/rviz_out", rviz_out);
    nh.getParam("/obstacle_processor_node/cmd_out", cmd_out);
    nh.getParam("/obstacle_processor_node/transform_to_base_link", transform_to_base_link);
    nh.getParam("/obstacle_processor_node/dist_weight_circ", dist_weight_circ);
    nh.getParam("/obstacle_processor_node/dist_thresh_circ", dist_thresh_circ);
    nh.getParam("/obstacle_processor_node/height_thresh_min", height_thresh_min);
    nh.getParam("/obstacle_processor_node/height_thresh_max", height_thresh_max);
    nh.getParam("/obstacle_processor_node/iter_circ", iter_circ);
  }
  ObstacleProcessor(ros::NodeHandle nh_): nh(nh_)
  {
    nh.getParam("/obstacle_processor_node/pc_topic", pc_topic);
    cloud_sub_=nh.subscribe(pc_topic, 1, &ObstacleProcessor::cloud_cb, this);
    pub_clusters = nh.advertise<PCloud2>("/clusters", 1);
    pub_backup = nh.advertise<PCloud2>("/clusters_backup", 1);
    pub_cloud = nh.advertise<PCloud2>("/cloud", 1);
    pub_top_planes = nh.advertise<PCloud2>("/top_planes", 1);
    pub_projected = nh.advertise<PCloud2>("/projected_planes", 1);
    pub_projected_flat = nh.advertise<PCloud2>("/projected_planes_flat", 1);
    pub_hull = nh.advertise<PCloud2>("/convex_hulls", 1);
    pub_front_plane = nh.advertise<PCloud2>("/front_planes", 1);
    pub_box = nh.advertise<processed_box_data>("/box_data", 1, true);
    pub_cylinder = nh.advertise<processed_cylinder_data>("/cylinder_data", 1, true);
    pub_undefined = nh.advertise<processed_undefined_data>("/undefined_data", 1, true);
    pub_wall_like_object = nh.advertise<processed_wall_like_object_data>("/wall_like_object_data", 1, true);
    pub_floor = nh.advertise<PCloud2>("/cloud_floor", 1, true);
  }

  void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &input)
  {
    this->setParams(nh);
    PointCloud<Point>::Ptr cloud_input(new PointCloud<Point>), cloud_downsampled(new PointCloud<Point>),
                           cloud_processed(new PointCloud<Point>), cloud_filtered(new PointCloud<Point>),
                           cloud_clusters(new PointCloud<Point>), cloud_floor(new PointCloud<Point>),
                           cloud_top_plane_all(new PointCloud<Point>), front_plane_all(new PointCloud<Point>),
                           top_plane_project_all(new PointCloud<Point>), top_plane_project_flat_all(new PointCloud<Point>),
                           top_plane_convex_hull_all(new PointCloud<Point>), cloud_cluster_all(new PointCloud<Point>);
    PCloud2::Ptr cloud_out(new PCloud2), backup_out(new PCloud2), clusters_out(new PCloud2), top_planes_out(new PCloud2),
                 projected_out(new PCloud2), projected_flat_out(new PCloud2), convex_hull_out(new PCloud2),
                 front_planes_out(new PCloud2), cloud_floor_out(new PCloud2);
    geometry_msgs::PointStamped kinect_centroid;
    kinect_centroid.header.frame_id = "camera";
    kinect_centroid.header.stamp = input->header.stamp;
    MCoef::Ptr coefficients_floor(new MCoef);
    PIndices::Ptr inliers_floor(new PIndices);
    SACSegmentation<Point> seg;
    SACSegmentationFromNormals<Point, Normal> seg_norm;
    ExtractIndices<Normal> extract_normals;
    StatisticalOutlierRemoval<Point> sor;
    NormalEstimationOMP<Point, Normal> ne;
    search::KdTree<Point>::Ptr tree(new search::KdTree<Point>);
    PassThrough<Point> pass_filter;
    VoxelGrid<Point> voxel;
    ExtractIndices<Point> extract;
    vector<PIndices> clusters_indices;
    EuclideanClusterExtraction<Point> ec_ext;
    Quaternionf quart;
    Affine3f transform_glob(Affine3f::Identity());
    Affine3f trans_glob(Affine3f::Identity());
    srand((int)time(0));

    fromROSMsg(*input, *cloud_input);

    if (cmd_out) {
      ROS_INFO("Pass filtering...");
      tt.tic();
    }
    pass_filter.setInputCloud(cloud_input);
    pass_filter.setFilterFieldName("z");
    pass_filter.setFilterLimits(min_z, max_z);
    pass_filter.filter(*cloud_filtered);
    if (cmd_out) ROS_INFO("Done: %3.3f ms, points = %3d", tt.toc(), cloud_filtered->points.size());

    if (cmd_out) {
      ROS_INFO("Downsampling...");
      tt.tic();
    }
    voxel.setInputCloud(cloud_filtered);
    voxel.setLeafSize(voxel_size, voxel_size, voxel_size);
    voxel.setDownsampleAllData(true);
    voxel.filter(*cloud_downsampled);
    if (cmd_out) ROS_INFO("Done: %3.3f ms, points = %3d", tt.toc(), cloud_downsampled->points.size());

    if (cmd_out) {
      ROS_INFO("Starting ground plane segmentation...");
      tt.tic();
    }
    seg.setModelType(SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(SAC_RANSAC);
    seg.setOptimizeCoefficients(true);
    seg.setMaxIterations(iter);
    seg.setEpsAngle(7.5 * (M_PI / 180));
    seg.setAxis(Vector3f(0.0, cos(alfa), sin(alfa)));
    seg.setDistanceThreshold(plane_seg_thresh);
    seg.setInputCloud(cloud_downsampled);
    seg.segment(*inliers_floor, *coefficients_floor);

    if (inliers_floor->indices.size() == 0)
    {
      ROS_ERROR("Couldn't estimate a planar model from dataset!");
      return;
    }
    Vector3f floor_normal(coefficients_floor->values[0], coefficients_floor->values[1], coefficients_floor->values[2]);
    if (cmd_out) {
      ROS_INFO("Ground plane's normal: (%3.3f, %3.3f, %3.3f)", floor_normal[0], floor_normal[1], floor_normal[2]);
      ROS_INFO("Ground plane's size: %3d", inliers_floor->indices.size());
    }

    // Extract the planar inliers from input
    extract.setInputCloud(cloud_downsampled);
    extract.setIndices(inliers_floor);
    extract.setNegative(false);
    extract.filter(*cloud_floor);
    extract.setNegative(true);
    extract.filter(*cloud_processed);
    if (cmd_out) ROS_INFO("Done: %3.3f ms", tt.toc());

    // Rotate so that z axis is colinear with floor normal
    if (cmd_out) {
      ROS_INFO("Rotating and translating point clouds...");
      tt.tic();
    }
    quart.setFromTwoVectors(floor_normal, Vector3f::UnitZ());
    Matrix3f rot_glob = quart.toRotationMatrix();
    Matrix3f rot_glob_inv = rot_glob.inverse();
    transform_glob.rotate(rot_glob);
    transformPointCloud(*cloud_processed, *cloud_processed, transform_glob);
    transformPointCloud(*cloud_floor, *cloud_floor, transform_glob);

    // Translate so that an origin is situated on a floor plane
    trans_glob.translation() << 0.0, 0.0, -1*(cloud_floor->points[0].z);
    transformPointCloud(*cloud_processed, *cloud_processed, trans_glob);
    transformPointCloud(*cloud_floor, *cloud_floor, trans_glob);
    if (cmd_out) ROS_INFO("Done: %3.3f ms", tt.toc());

    if (cmd_out) {
      ROS_INFO("Obtaining model coef of ground plane...");
      tt.tic();
    }
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(iter);
    seg.setDistanceThreshold(plane_seg_thresh);
    seg.setInputCloud(cloud_floor);
    seg.segment(*inliers_floor, *coefficients_floor);
    Vector4f floor_coef(coefficients_floor->values[0], coefficients_floor->values[1], coefficients_floor->values[2], coefficients_floor->values[3]);
    floor_normal << coefficients_floor->values[0], coefficients_floor->values[1], coefficients_floor->values[2];
    if (cmd_out) ROS_INFO("Ground plane's coef after transformation: (%3.3f, %3.3f, %3.3f, %3.3f)", floor_normal(0), floor_normal(0), floor_normal(0), coefficients_floor->values[3]);
    if (cmd_out) ROS_INFO("Done: %3.3f ms", tt.toc());

    // Creating the KdTree object for the search method of the extraction
    if (cmd_out) {
      ROS_INFO("Starting cluster extraction...");
      tt.tic();
    }
    ec_ext.setClusterTolerance(cluster_tol);
    ec_ext.setMinClusterSize(min_cluster);
    ec_ext.setMaxClusterSize(max_cluster);
    ec_ext.setSearchMethod(tree);
    ec_ext.setInputCloud(cloud_processed);
    ec_ext.extract(clusters_indices);
    if (cmd_out) ROS_INFO("Done: %3.3f ms", tt.toc());

    for (vector<PIndices>::const_iterator it = clusters_indices.begin(); it != clusters_indices.end(); ++it)
    {
      search::KdTree<Point>::Ptr cluster_tree(new search::KdTree<Point>);
      PointCloud<Normal>::Ptr cluster_normals(new PointCloud<Normal>), normals_copy(new PointCloud<Normal>),
                              normals_copy_new(new PointCloud<Normal>);
      PointCloud<Point>::Ptr cluster_copy(new PointCloud<Point>), cluster_copy_new(new PointCloud<Point>),
                             cloud_top_plane(new PointCloud<Point>), front_plane(new PointCloud<Point>),
                             cloud_cluster(new PointCloud<Point>), cloud_cluster_backup(new PointCloud<Point>);
      MCoef::Ptr coef_top_plane(new MCoef), coef_front_plane(new MCoef);
      PIndices::Ptr inliers_top_plane(new PIndices), inliers_front_plane(new PIndices);
      vector<double> cluster_dims, dims(2,0), centroid_vector;
      Vector3f centroid(0, 0, 0), front_plane_norm(0.0, 0.0, 0.0), z_vector(0.0, 0.0, 1.0);
      double h = 0;
      float fi = 0;
      double *height = &h;
      float *frontAngle = &fi;
      bool isWallLike = false, hasPlane = true, isCylinder = false, hasParallelPlaneToFloor = false;
      double pointsCount = 0.0, pointsCountHeight = 0.0, cluster_size;
      int r = rand()%255, g = rand()%255, b = rand()%255;

      if (cmd_out) {
        ROS_INFO("Applying filter to a cluster...");
        tt.tic();
      }
      for (vector<int>::const_iterator point_it = it->indices.begin(); point_it != it->indices.end(); ++point_it)
      {
        Point p = cloud_processed->points[*point_it];
        if ((p.z > point_min_z) & (p.z < point_max_z)) ++pointsCount;
        if ((p.z > height_thresh_min) & (p.z < height_thresh_max)) ++pointsCountHeight;
        cloud_cluster->points.push_back(p);
        if (rviz_out) {
          p.r = r;
          p.g = g;
          p.b = b;
          cloud_cluster_backup->points.push_back(p);
        }
      }
      cluster_size = cloud_cluster->points.size();
      if (cmd_out) ROS_INFO("Done: %3.3f ms", tt.toc());

      if (cluster_size > 12000)
      {
        isWallLike = true;
      }
      else
      {
        if (((pointsCount / cluster_size) < 0.25) && ((pointsCountHeight / cluster_size) > 0.025))
        {
          // Remove outliers
          if (cmd_out) {
            ROS_INFO("  Cluster size = %3.3f", cluster_size);
            ROS_INFO("Removing outliers...");
            tt.tic();
          }
          sor.setInputCloud(cloud_cluster);
          sor.setMeanK(meanK);
          sor.setStddevMulThresh(outliers_thresh);
          sor.filter(*cloud_cluster);
          cluster_size = cloud_cluster->points.size();
          if (cmd_out) ROS_INFO("Done: %3.3f ms", tt.toc());

          // Normal estimation
          if (rviz_out) *cloud_cluster_all += *cloud_cluster_backup;
          if (cmd_out) {
            ROS_INFO("Estimating cluster normals...");
            tt.tic();
          }
          ne.setSearchMethod(cluster_tree);
          ne.setInputCloud(cloud_cluster);
          ne.setKSearch(K_search);
          ne.compute(*cluster_normals);
          if (cmd_out) ROS_INFO("Done: %3.3f ms", tt.toc());

          // Copying clouds
          if (cmd_out) {
            ROS_INFO("Creating shallow copies of cluster and normal clouds...");
            tt.tic();
          }
          copyPointCloud(*cloud_cluster, *cluster_copy);
          copyPointCloud(*cloud_cluster, *cluster_copy_new);
          copyPointCloud(*cluster_normals, *normals_copy);
          copyPointCloud(*cluster_normals, *normals_copy_new);
          if (cmd_out) ROS_INFO("Done: %3.3f ms", tt.toc());

          // Checking the wall-like condition
          if (cluster_size > 2500)
          {
            if (cmd_out) {
              ROS_INFO("Checking the wall-like object condition...");
              tt.tic();
            }
            seg_norm.setModelType(SACMODEL_NORMAL_PLANE);
            seg_norm.setNormalDistanceWeight(dist_weight);
            seg_norm.setMethodType(SAC_RANSAC);
            seg_norm.setMaxIterations(iter2);
            seg_norm.setDistanceThreshold(thresh2);
            seg_norm.setInputCloud(cloud_cluster);
            seg_norm.setInputNormals(cluster_normals);
            seg_norm.segment(*inliers_front_plane, *coef_front_plane);
            extract.setInputCloud(cloud_cluster);
            extract.setIndices(inliers_front_plane);
            extract.setNegative(false);
            extract.filter(*front_plane);
            if (front_plane->points.size() / cluster_size > 0.65)
            {
              front_plane_norm << coef_front_plane->values[0], coef_front_plane->values[1], coef_front_plane->values[2];
              if (this->isPerpendicularToPlane(true, front_plane_norm, z_vector, *frontAngle))
              {
                if (cmd_out) ROS_INFO("Found frontal plane: (%3.3f, %3.3f, %3.3f, %3.3f), size = %3d", front_plane_norm(0), front_plane_norm(1), front_plane_norm(2), coef_front_plane->values[3], front_plane->points.size());
                isWallLike = true;
              }
            }
            else
            {
              if (cmd_out) ROS_INFO("No frontal plane found...");
            }
            if (cmd_out) ROS_INFO("Done: %3.3f ms", tt.toc());
          }
        }
        else
        {
          continue;
        }
      }

      if (!isWallLike)
      {
        cluster_dims = this->findMinMax3D(cloud_cluster);
        double x_dim = fabs(cluster_dims[3] - cluster_dims[0]);
        double z_dim = fabs(cluster_dims[5] - cluster_dims[2]);
        if (cmd_out) {
          ROS_INFO("Searching for a top plane... ");
          tt.tic();
        }
        // Searching for a top plane
        do
        {
          seg_norm.setModelType(SACMODEL_NORMAL_PLANE);
          seg_norm.setNormalDistanceWeight(dist_weight);
          seg_norm.setMethodType(SAC_RANSAC);
          seg_norm.setMaxIterations(iter2);
          seg_norm.setDistanceThreshold(thresh2);
          seg_norm.setOptimizeCoefficients(true);
          seg_norm.setInputCloud(cloud_cluster);
          seg_norm.setInputNormals(cluster_normals);
          seg_norm.segment(*inliers_top_plane, *coef_top_plane);

          extract.setInputCloud(cloud_cluster);
          extract.setIndices(inliers_top_plane);
          extract.setNegative(false);
          extract.filter(*cloud_top_plane);
          // Check if top plane was found
          if (cloud_top_plane->points.size() / cluster_size > 0.05)
          {
            // Estimate height
            double height_temp;
            double size = cloud_top_plane->points.size();
            double sum_height = 0;
            Point p;
            for (size_t i=0; i < size; i++)
            {
              p = cloud_top_plane->points[i];
              height_temp = 1000;
              height_temp = pointToPlaneDistance(p, floor_coef);
              sum_height += height_temp;
            }
            *height = sum_height / size;
            // Check whether found plane is parallel to the floor
            if (this->isTopParallelPlane(coefficients_floor, coef_top_plane, *height, z_dim))
            {
              hasPlane = false;
              hasParallelPlaneToFloor = true;
              if (rviz_out) *cloud_top_plane_all += *cloud_top_plane;
            }
            else {
              extract.setNegative(true);
              extract.filter(*cloud_cluster);
              extract_normals.setNegative(true);
              extract_normals.setInputCloud(cluster_normals);
              extract_normals.setIndices(inliers_top_plane);
              extract_normals.filter(*cluster_normals);
            }
          }
          else
          {
            if (cmd_out) ROS_INFO("Cannot find plane parallel to the ground plane");
            hasPlane = false;
          }
        } while(hasPlane);
        if (cmd_out) ROS_INFO("Done: %3.2f ms.", tt.toc());

        if (hasParallelPlaneToFloor)
        {
          dims = this->projectAndProcess(hasParallelPlaneToFloor, isCylinder, cloud_top_plane, coefficients_floor, top_plane_project_all, top_plane_project_flat_all, top_plane_convex_hull_all, centroid);
          centroid(2) = *height / 2.0;
          centroid = rot_glob_inv * (centroid - trans_glob.translation());
          centroid_vector = this->transformToBaseLink(transform_to_base_link, centroid, kinect_centroid);

          if (isCylinder)
          {
            hasPlane = true;
            double ratio;
            double area = *height * dims[1];
            double ratio_med = 0;
            int count = 0;
            double area_ratio = area / ((2 * area) + (dims[1] * dims[1]));
            if (cmd_out) {
              ROS_INFO("Searching for planes perpendicular to the ground plane... ");
              tt.tic();
            }
            do
            {
              seg_norm.setModelType(SACMODEL_NORMAL_PLANE);
              seg_norm.setNormalDistanceWeight(dist_weight_circ);
              seg_norm.setMethodType(SAC_RANSAC);
              seg_norm.setMaxIterations(iter_circ);
              seg_norm.setDistanceThreshold(dist_thresh_circ);
              seg_norm.setInputCloud(cluster_copy_new);
              seg_norm.setInputNormals(normals_copy_new);
              seg_norm.segment(*inliers_front_plane, *coef_front_plane);
              extract.setInputCloud(cluster_copy_new);
              extract.setIndices(inliers_front_plane);
              extract.setNegative(false);
              extract.filter(*front_plane);
              ratio = ((double) front_plane->points.size()) / cluster_size;
              if (ratio > (0.15 * area_ratio))
              {
                front_plane_norm << coef_front_plane->values[0], coef_front_plane->values[1], coef_front_plane->values[2];
                if (this->isPerpendicularToPlane(false, front_plane_norm, z_vector, *frontAngle))
                {
                  count++;
                  ratio_med += ratio;
                }
                extract.setNegative(true);
                extract.filter(*cluster_copy_new);
                extract_normals.setNegative(true);
                extract_normals.setInputCloud(normals_copy_new);
                extract_normals.setIndices(inliers_front_plane);
                extract_normals.filter(*normals_copy_new);
              }
              else
              {
                if (cmd_out) ROS_INFO("Cannot find plane perpendicular to the ground plane");
                hasPlane = false;
              }
            } while(hasPlane);
            if (cmd_out) ROS_INFO("Done: %3.2f ms.", tt.toc());
            ratio_med /= count;
            ratio_med /= area_ratio;
            if (ratio_med > 0)
              isCylinder = ratio_med < 0.45? true: false;
          }

          if (isCylinder)
          {
            if (cmd_out) ROS_INFO("# Found CYLINDRICAL OBJECT:\n# Radius = %3.3f\n# height = %3.3f\n# Centroid: (%3.3f, %3.3f, %3.3f)", (dims[1]+dims[0]) / 4, *height, centroid(0), centroid(1), centroid(2));
            cylinder_data.radius = (dims[1]+dims[0]) / 4;
            cylinder_data.height = *height;
            cylinder_data.centroid = centroid_vector;
            pub_cylinder.publish(cylinder_data);
          }
          else
          {
            hasPlane = true;
            if (cmd_out) {
              ROS_INFO("Searching for a frontal plane... ");
              tt.tic();
            }
            // Searching for a frontal plane
            do
            {
              seg_norm.setModelType(SACMODEL_NORMAL_PLANE);
              seg_norm.setNormalDistanceWeight(dist_weight);
              seg_norm.setMethodType(SAC_RANSAC);
              seg_norm.setMaxIterations(iter2);
              seg_norm.setDistanceThreshold(thresh2);
              seg_norm.setInputCloud(cluster_copy);
              seg_norm.setInputNormals(normals_copy);
              seg_norm.segment(*inliers_front_plane, *coef_front_plane);
              extract.setInputCloud(cluster_copy);
              extract.setIndices(inliers_front_plane);
              extract.setNegative(false);
              extract.filter(*front_plane);
              if (front_plane->points.size() / cluster_size > 0.1)
              {
                front_plane_norm << coef_front_plane->values[0], coef_front_plane->values[1], coef_front_plane->values[2];
                if (this->isPerpendicularToPlane(false, front_plane_norm, z_vector, *frontAngle))
                {
                  hasPlane = false;
                  if (rviz_out) *front_plane_all += *front_plane;
                }
                else
                {
                  extract.setNegative(true);
                  extract.filter(*cluster_copy);
                  extract_normals.setNegative(true);
                  extract_normals.setInputCloud(normals_copy);
                  extract_normals.setIndices(inliers_front_plane);
                  extract_normals.filter(*normals_copy);
                }
              }
              else
              {
                if (cmd_out) ROS_INFO("No frontal plane found...");
                hasPlane = false;
              }

            } while (hasPlane);
            if (cmd_out) {
              ROS_INFO("Done: %3.2f ms.", tt.toc());
              ROS_INFO("Found BOX-LIKE OBJECT:\n# Length = %3.3f\n# Width = %3.3f\n# Height = %3.3f"
                       "\n# Frontal angle = %3.3f\n# Centroid: (%3.3f, %3.3f, %3.3f)",
                       dims[0], dims[1], *height, *frontAngle, centroid(0), centroid(1), centroid(2));
            }
            box_data.length = dims[0];
            box_data.width = dims[1];
            box_data.height = *height;
            box_data.angle = *frontAngle;
            box_data.centroid = centroid_vector;
            pub_box.publish(box_data);
          }
        }
        else
        {
          // Estimate object's profile
          vector<double> profile(2,0);
          Point p;
          profile += (x_dim, z_dim);
          dims = this->projectAndProcess(hasParallelPlaneToFloor, isCylinder, cluster_copy, coefficients_floor, top_plane_project_all, top_plane_project_flat_all, top_plane_convex_hull_all, centroid);
          computeCentroid(*cluster_copy, p);
          centroid << p.x, p.y, p.z;
          centroid = rot_glob_inv * (centroid - trans_glob.translation());
          centroid_vector = this->transformToBaseLink(transform_to_base_link, centroid, kinect_centroid);
          if (!isCylinder)
          {
            if (cmd_out) ROS_INFO("# Found UNDEFINED OBJECT:\n# Profile = (%3.3f, %3.3f)"
                                  "\n# Length = %3.3f\n# Width = %3.3f\n# Centroid: (%3.3f, %3.3f, %3.3f)",
                                  x_dim, z_dim, dims[0], dims[1], centroid(0), centroid(1), centroid(2));
            undefined_data.profile = profile;
            undefined_data.centroid = centroid_vector;
            undefined_data.length = dims[0];
            undefined_data.width = dims[1];
            pub_undefined.publish(undefined_data);
          }
          else
          {
            if (cmd_out) ROS_INFO("Found CYLINDRICAL OBJECT\n# Radius = %3.3f\n# Height = %3.3f"
                                  "\n# Centroid: (%3.3f, %3.3f, %3.3f)",
                                  (dims[1]+dims[0]) / 4, *height, centroid(0), centroid(1), centroid(2));
            cylinder_data.radius = (dims[1]+dims[0]) / 4;
            cylinder_data.height = z_dim;
            cylinder_data.centroid = centroid_vector;
            pub_cylinder.publish(cylinder_data);
          }
        }
      }
      else
      {
        Point p;
        cluster_dims = this->findMinMax3D(front_plane->makeShared());
        double x_dim = fabs(cluster_dims[3] - cluster_dims[0]);
        double y_dim = fabs(cluster_dims[4] - cluster_dims[1]);
        double z_dim = fabs(cluster_dims[5] - cluster_dims[2]);
        double length = *frontAngle > 45? y_dim / cos(fabs(90 - *frontAngle) * (M_PI / 180)):
                                          x_dim / cos(*frontAngle * (M_PI / 180));
        double width = x_dim < y_dim? x_dim: y_dim;
        computeCentroid(*front_plane, p);
        centroid << p.x, p.y, p.z;
        centroid = rot_glob_inv * (centroid - trans_glob.translation());
        centroid_vector = this->transformToBaseLink(transform_to_base_link, centroid, kinect_centroid);
        if (cmd_out) ROS_INFO("# Found WALL-LIKE OBJECT:\n# Length = %3.3f\n# Width = %3.3f\n# Height = %3.3f"
                              "\n# Frontal angle = %3.3f\n# Centroid: (%3.3f, %3.3f, %3.3f)",
                              length, width, z_dim, *frontAngle, centroid(0), centroid(1), centroid(2));
        wall_like_object_data.centroid = centroid_vector;
        wall_like_object_data.length = length;
        wall_like_object_data.width = width;
        wall_like_object_data.height = z_dim;
        wall_like_object_data.angle = *frontAngle;
        pub_wall_like_object.publish(wall_like_object_data);
      }
      *cloud_clusters += *cloud_cluster;
      if (cmd_out) cerr << "\n---------------------------------------\n";
    }
    if (rviz_out)
    {
      if (cmd_out) {
        ROS_INFO("Publishing additional topics...");
        tt.tic();
      }
      string topic = "kinect2_ir_optical_frame";
      toROSMsg(*cloud_clusters, *clusters_out);
      toROSMsg(*top_plane_project_all, *projected_out);
      toROSMsg(*top_plane_project_flat_all, *projected_flat_out);
      toROSMsg(*top_plane_convex_hull_all, *convex_hull_out);
      toROSMsg(*cloud_floor, *cloud_floor_out);
      toROSMsg(*cloud_top_plane_all, *top_planes_out);
      toROSMsg(*cloud_processed, *cloud_out);
      toROSMsg(*cloud_cluster_all, *backup_out);
      toROSMsg(*front_plane_all, *front_planes_out);

      cloud_floor_out->header.frame_id = topic;
      backup_out->header.frame_id = topic;
      clusters_out->header.frame_id = topic;
      top_planes_out->header.frame_id = topic;
      cloud_out->header.frame_id = topic;
      projected_out->header.frame_id = topic;
      projected_flat_out->header.frame_id = topic;
      convex_hull_out->header.frame_id = topic;
      front_planes_out->header.frame_id = topic;

      pub_floor.publish(cloud_floor_out);
      pub_backup.publish(backup_out);
      pub_cloud.publish(cloud_out);
      pub_clusters.publish(clusters_out);
      pub_top_planes.publish(top_planes_out);
      pub_hull.publish(convex_hull_out);
      pub_projected.publish(projected_out);
      pub_projected_flat.publish(projected_flat_out);
      pub_front_plane.publish(front_planes_out);
      if (cmd_out) {
        ROS_INFO("Done: %3.3f ms", tt.toc());
        cerr << "\n***************************************\n";
      }
    }
  }

};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "obstacle_processor_node");
  fstream f;
  string line;
  tf::TransformListener lr(ros::Duration(10));
  tf_listener = &lr;
  alfa = 0;
  f.open("cal_data.dat");
  if (!f.is_open())
  {
    cerr << "Error while reading a calibration data!\n";
    return -1;
  }
  getline(f, line);
  alfa = atof(line.c_str());
  f.close();
  if (alfa == 0) return -1;
  ros::NodeHandle nh;
  cerr << "Read successfully from cal data = " << alfa << "\n";
  ObstacleProcessor cs(nh);
  ros::spin();
}
