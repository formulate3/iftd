#include "include/IFTDesc.h"

// 计算皮尔逊相关系数的函数
double pearsonCorrelation(const std::array<int, 9>& x, const std::array<int, 9>& y) {
    if (x.size() != y.size() || x.size() == 0) {
        throw std::invalid_argument("Arrays must have the same non-zero size.");
    }

    // 计算均值
    double mean_x = std::accumulate(x.begin(), x.begin() + 8, 0.0) / 8.0;
    double mean_y = std::accumulate(y.begin(), y.begin() + 8, 0.0) / 8.0;

    // 计算协方差和标准差
    double covariance = 0.0;
    double variance_x = 0.0;
    double variance_y = 0.0;

    for (size_t i = 0; i < x.size() - 1; ++i) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;

        covariance += dx * dy;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }

    // 防止分母为零
    if (variance_x == 0 && variance_y == 0) {
//        throw std::runtime_error("One of the arrays has zero variance, correlation is undefined.");
        return 1;
    }
    else if(variance_x == 0){
        if(mean_y > 0.3) return 0;
        else if(mean_y > 0.1) return 0.5;
        else return 1;
    }
    else if(variance_y == 0){
        if(mean_x > 0.3) return 0;
        else if(mean_x > 0.1) return 0.5;
        else return 1;
    }
    else{
        // 计算皮尔逊相关系数
        return covariance / std::sqrt(variance_x * variance_y);
    }
}

double cosineSimilarity(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) {
    // Compute the norms of the vectors
    double norm_v1 = v1.norm();
    double norm_v2 = v2.norm();

    // Check for zero vectors to avoid division by zero
    if (norm_v1 == 0 || norm_v2 == 0) {
//        throw std::invalid_argument("One of the vectors has zero magnitude.");
        return 0;
    }

    // Compute the dot product
    double dot_product = v1.dot(v2);

    // Compute and return the cosine similarity
    return dot_product / (norm_v1 * norm_v2);
}


std::pair<double, double> computeGaussianParams(const std::array<int, 9>& array){
    double sum = 0.0, sum_sq = 0.0;
    int count = 0;

    for (int density : array) {
        if (density > 0) { // 忽略密度为0的格子
            sum += density;
            sum_sq += density * density;
            count++;
        }
    }

    if (count == 0) {
        return {-1.0, -1.0}; // 所有密度为0，返回错误值
    }

    double mean = sum / count;
    double variance = (sum_sq / count) - (mean * mean);
    variance = std::max(variance, 0.0); // 避免数值误差导致负方差

    return {mean, std::sqrt(variance)};
}

double computeWassersteinDistance(const std::array<int, 9>& array1, const std::array<int, 9>& array2){
    // 计算两个分布的均值和标准差
    auto [mean1, stddev1] = computeGaussianParams(array1);
    auto [mean2, stddev2] = computeGaussianParams(array2);

    if (mean1 == -1.0 || mean2 == -1.0) {
        return -1.0; // 某一个数组的密度值均为0
    }

    return std::sqrt((mean1 - mean2) * (mean1 - mean2) + (stddev1 - stddev2) * (stddev1 - stddev2));
}

void down_sampling_voxel(pcl::PointCloud<pcl::PointXYZI> &pl_feat,
                         double voxel_size) {
  int intensity = rand() % 255;
  if (voxel_size < 0.01) {
    return;
  }
  std::unordered_map<VOXEL_LOC, M_POINT> voxel_map;
  uint plsize = pl_feat.size();

  for (uint i = 0; i < plsize; i++) {
    pcl::PointXYZI &p_c = pl_feat[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      iter->second.xyz[0] += p_c.x;
      iter->second.xyz[1] += p_c.y;
      iter->second.xyz[2] += p_c.z;
      iter->second.intensity += p_c.intensity;
      iter->second.count++;
    } else {
      M_POINT anp;
      anp.xyz[0] = p_c.x;
      anp.xyz[1] = p_c.y;
      anp.xyz[2] = p_c.z;
      anp.intensity = p_c.intensity;
      anp.count = 1;
      voxel_map[position] = anp;
    }
  }
  plsize = voxel_map.size();
  pl_feat.clear();
  pl_feat.resize(plsize);

  uint i = 0;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter) {
    pl_feat[i].x = iter->second.xyz[0] / iter->second.count;
    pl_feat[i].y = iter->second.xyz[1] / iter->second.count;
    pl_feat[i].z = iter->second.xyz[2] / iter->second.count;
    pl_feat[i].intensity = iter->second.intensity / iter->second.count;
    i++;
  }
}
void read_parameters(ros::NodeHandle &nh, ConfigSetting &config_setting) {
  // pre-preocess
  nh.param<double>("ds_size", config_setting.ds_size_, 0.5);
  // read bin_file
  nh.param<int>("bin_timestamp", config_setting.bin_timestamp, 1);
  // BEV
  nh.param<int>("BEV_X_NUM", config_setting.BEV_X_NUM, 400);
  nh.param<int>("BEV_Y_NUM", config_setting.BEV_Y_NUM, 400);
  nh.param<int>("BEV_X_MAX", config_setting.BEV_X_MAX, 100);
  nh.param<int>("BEV_Y_MAX", config_setting.BEV_Y_MAX, 100);
  nh.param<double>("lidar_height", config_setting.lidar_height, 1.5);
  nh.param<double>("height_bin", config_setting.height_bin, 0.3);
  nh.param<double>("Density_resolution", config_setting.Density_resolution, 0.5);
  nh.param<double>("Density_threshold", config_setting.Density_threshold, 0.05);
  nh.param<bool>("use_global_bev", config_setting.use_global_bev, true);
  // feature extraction
  nh.param<double>("image_quartity", config_setting.image_quartity, 0.15);
  nh.param<int>("Hamming_distance", config_setting.Hamming_distance, 10);
  // std descriptor
  nh.param<int>("descriptor_near_num", config_setting.descriptor_near_num_, 15);
  nh.param<double>("descriptor_min_len", config_setting.descriptor_min_len_, 2);
  nh.param<double>("descriptor_max_len", config_setting.descriptor_max_len_, 50);
  nh.param<double>("std_side_resolution", config_setting.std_side_resolution_, 0.2);

  // candidate search
  nh.param<int>("skip_near_num", config_setting.skip_near_num_, 50);
  nh.param<int>("candidate_num", config_setting.candidate_num_, 50);
  nh.param<int>("sub_frame_num", config_setting.sub_frame_num_, 10);
  nh.param<double>("rough_dis_threshold", config_setting.rough_dis_threshold_,0.01);
  nh.param<double>("vertex_diff_threshold",config_setting.vertex_diff_threshold_, 0.5);
  nh.param<int>("image_reseize",config_setting.image_reseize_, 20);
  nh.param<double>("distance_threshold",config_setting.distance_threshold_, 15.0);
  nh.param<double>("image_threshold", config_setting.image_threshold_, 0.2);
  nh.param<double>("density_diff_threshold", config_setting.density_diff_threshold, 0.1);

  std::cout << "Sucessfully load parameters:" << std::endl;
  std::cout << "----------------Main Parameters-------------------" << std::endl;
  std::cout << "BEV_X_NUM: " << config_setting.BEV_X_NUM << std::endl;
  std::cout << "BEV_Y_NUM: " << config_setting.BEV_Y_NUM << std::endl;
  std::cout << "BEV_X_MAX: " << config_setting.BEV_X_MAX << std::endl;
  std::cout << "BEV_Y_MAX: " << config_setting.BEV_Y_MAX << std::endl;
  std::cout << "Density_resolution: " << config_setting.Density_resolution << std::endl;
  std::cout << "Density_threshold: " << config_setting.Density_threshold << std::endl;
  std::cout << "use_global_bev: " << config_setting.use_global_bev << std::endl;
  std::cout << "image_quartity: " << config_setting.image_quartity << std::endl;
  std::cout << "Hamming_distance: " << config_setting.Hamming_distance << std::endl;
  std::cout << "loop detection threshold: " << config_setting.image_threshold_ << std::endl;
  std::cout << "sub-frame number: " << config_setting.sub_frame_num_ << std::endl;
  std::cout << "candidate number: " << config_setting.candidate_num_ << std::endl;
  std::cout << "image_reseize: " << config_setting.image_reseize_ << std::endl;
  std::cout << "distance threshold: " << config_setting.distance_threshold_ << std::endl;
  std::cout << "descriptor_near_num: " << config_setting.descriptor_near_num_ << std::endl;
  std::cout << "density_diff_threshold: "<< config_setting.density_diff_threshold << std::endl;
}

void load_pose_with_time(
    const std::string &pose_file,
    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &poses_vec,
    std::vector<double> &times_vec) {
  times_vec.clear();
  poses_vec.clear();
  std::ifstream fin(pose_file);
  std::string line;
  Eigen::Matrix<double, 1, 7> temp_matrix;
  while (getline(fin, line)) {
    std::istringstream sin(line);
    std::vector<std::string> Waypoints;
    std::string info;
    int number = 0;
    while (getline(sin, info, ' ')) {
      if (number == 0) {
        double time;
        std::stringstream data;
        data << info;
        data >> time;
        times_vec.push_back(time);
        // std::cout.precision(9);
        // std::cout << std::fixed << "time: " << time;
        number++;
      } else {
        double p;
        std::stringstream data;
        data << info;
        data >> p;
        temp_matrix[number - 1] = p;
        if (number == 7) {
          Eigen::Vector3d translation(temp_matrix[0], temp_matrix[1],
                                      temp_matrix[2]);
          Eigen::Quaterniond q(temp_matrix[6], temp_matrix[3], temp_matrix[4],
                               temp_matrix[5]);

          std::pair<Eigen::Vector3d, Eigen::Matrix3d> single_pose;
          single_pose.first = translation;
          single_pose.second = q.toRotationMatrix();
          poses_vec.push_back(single_pose);
        }
        number++;
      }
    }
  }
}

int findClosestTimestamp(double target, const std::vector<double>& times_vec) {
    auto closest = std::min_element(times_vec.begin(), times_vec.end(),
        [target](double a, double b) {
            return std::abs(a - target) < std::abs(b - target);
        });
    return std::distance(times_vec.begin(), closest);
}

void pose_bin_timestamp_align(std::vector<std::string> bin_files,std::vector<std::string> &outbin_files,
  std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &outposes_vec,std::vector<double> &outtimes_vec,
  std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &inposes_vec,std::vector<double> &intimes_vec,
  int time_unit)
{
  outposes_vec.clear();
  outtimes_vec.clear();
  outbin_files.clear();
  std::cout.precision(9);
  std::cout << std::fixed;

  for (const auto &bin : bin_files)
  {
    double lidar_time = std::stod(bin) / time_unit; 

    int closest_index = findClosestTimestamp(lidar_time, intimes_vec);
    outbin_files.push_back(bin);
    outposes_vec.push_back(inposes_vec[closest_index]);
    outtimes_vec.push_back(intimes_vec[closest_index]);
  }
}

double time_inc(std::chrono::_V2::system_clock::time_point &t_end,
                std::chrono::_V2::system_clock::time_point &t_begin) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(t_end -
                                                                   t_begin)
             .count() *
         1000;
}

pcl::PointXYZI vec2point(const Eigen::Vector3d &vec) {
  pcl::PointXYZI pi;
  pi.x = vec[0];
  pi.y = vec[1];
  pi.z = vec[2];
  return pi;
}
Eigen::Vector3d point2vec(const pcl::PointXYZI &pi) {
  return Eigen::Vector3d(pi.x, pi.y, pi.z);
}

bool attach_greater_sort(std::pair<double, int> a, std::pair<double, int> b) {
  return (a.first > b.first);
}

void publish_std_pairs(
    const std::vector<std::pair<IFTDesc, IFTDesc>> &match_std_pairs,Eigen::Vector3d translation ,Eigen::Matrix3d rotation, 
    const ros::Publisher &std_publisher) {
  visualization_msgs::MarkerArray ma_line;
  visualization_msgs::Marker m_line;
  m_line.type = visualization_msgs::Marker::LINE_LIST;
  m_line.action = visualization_msgs::Marker::ADD;
  m_line.ns = "lines";
  // Don't forget to set the alpha!
  m_line.scale.x = 0.25;
  m_line.pose.orientation.w = 1.0;
  m_line.header.frame_id = "camera_init";
  m_line.id = 0;
  int max_pub_cnt = 1;
  for (auto var : match_std_pairs) {
    if (max_pub_cnt > 100) {
      break;
    }
    max_pub_cnt++;
    m_line.color.a = 0.8;
    m_line.points.clear();
    m_line.color.r = 138.0 / 255;
    m_line.color.g = 226.0 / 255;
    m_line.color.b = 52.0 / 255;
    geometry_msgs::Point p;
    p.x = var.second.vertex_A_[0];
    p.y = var.second.vertex_A_[1];
    p.z = var.second.vertex_A_[2];
    Eigen::Vector3d t_p;
    t_p << p.x, p.y, p.z;
    t_p = rotation * t_p + translation;
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    p.x = var.second.vertex_B_[0];
    p.y = var.second.vertex_B_[1];
    p.z = var.second.vertex_B_[2];
    t_p << p.x, p.y, p.z;
    t_p = rotation * t_p + translation;
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    p.x = var.second.vertex_C_[0];
    p.y = var.second.vertex_C_[1];
    p.z = var.second.vertex_C_[2];
    t_p << p.x, p.y, p.z;
    t_p = rotation * t_p + translation;
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    p.x = var.second.vertex_B_[0];
    p.y = var.second.vertex_B_[1];
    p.z = var.second.vertex_B_[2];
    t_p << p.x, p.y, p.z;
    t_p = rotation * t_p + translation;
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    p.x = var.second.vertex_C_[0];
    p.y = var.second.vertex_C_[1];
    p.z = var.second.vertex_C_[2];
    t_p << p.x, p.y, p.z;
    t_p = rotation * t_p + translation;
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    p.x = var.second.vertex_A_[0];
    p.y = var.second.vertex_A_[1];
    p.z = var.second.vertex_A_[2];
    t_p << p.x, p.y, p.z;
    t_p = rotation * t_p + translation;
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    // another
    m_line.points.clear();
    m_line.color.r = 1;
    m_line.color.g = 1;
    m_line.color.b = 1;
    p.x = var.first.vertex_A_[0];
    p.y = var.first.vertex_A_[1];
    p.z = var.first.vertex_A_[2];
    t_p << p.x, p.y, p.z;
    t_p = rotation * t_p + translation;
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    p.x = var.first.vertex_B_[0];
    p.y = var.first.vertex_B_[1];
    p.z = var.first.vertex_B_[2];
    t_p << p.x, p.y, p.z;
    t_p = rotation * t_p + translation;
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    p.x = var.first.vertex_C_[0];
    p.y = var.first.vertex_C_[1];
    p.z = var.first.vertex_C_[2];
    t_p << p.x, p.y, p.z;
    t_p = rotation * t_p + translation;
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    p.x = var.first.vertex_B_[0];
    p.y = var.first.vertex_B_[1];
    p.z = var.first.vertex_B_[2];
    t_p << p.x, p.y, p.z;
    t_p = rotation * t_p + translation;
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    p.x = var.first.vertex_C_[0];
    p.y = var.first.vertex_C_[1];
    p.z = var.first.vertex_C_[2];
    t_p << p.x, p.y, p.z;
    t_p = rotation * t_p + translation;
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    p.x = var.first.vertex_A_[0];
    p.y = var.first.vertex_A_[1];
    p.z = var.first.vertex_A_[2];
    t_p << p.x, p.y, p.z;
    t_p = rotation * t_p + translation;
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
  }
  for (int j = 0; j < 100 * 6; j++) {
    m_line.color.a = 0.00;
    ma_line.markers.push_back(m_line);
    m_line.id++;
  }
  std_publisher.publish(ma_line);
  m_line.id = 0;
  ma_line.markers.clear();
}

  void IFTDescManager::GenerateIFTDescs(pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                       std::vector<IFTDesc> &stds_vec, int num ,Eigen::Vector3d translation ,Eigen::Matrix3d rotation)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZI>); 
  pcl::copyPointCloud(*input_cloud, *temp_cloud);

  std::pair<Eigen::MatrixXd, Eigen::MatrixXi> result;
//  BEVResult bev_result;
  cv::Mat BEV_Image_Pixel;
  std::vector<std::vector<std::vector<Eigen::Vector3d>>> BEV_point_map;
//  std::vector<std::vector<double>> BEV_min_z_map(BEV_X_NUM, std::vector<double>(BEV_Y_NUM, std::numeric_limits<double>::max()));
  double min_z = 100;
  ///转为lidar系是为了使得方便使得点云在中间
  for (int i = 0; i < temp_cloud->size(); i++)
  {
      pcl::PointXYZI point = temp_cloud->points[i];
      point = vec2point(rotation.transpose() * (point2vec(point) - translation));

      temp_cloud->points[i] = point;
  }
    // Make BEV 按照有几个高度段有点为每个网格赋值
//  auto result = makeBEV(temp_cloud, translation, rotation);
      result = makeDensityBEV(temp_cloud, BEV_point_map, min_z);
      matrix_density = result.second;
      // BEV2Mat
      BEV_Image_Pixel = Eigen2Mat(result.second);

//      bev_result = makeDensityGlobalBEV(temp_cloud);
//      BEV_Image_Pixel = Eigen2Mat(bev_result.density_matrix);

  BEV_Image_Pixel.convertTo(BEV_Image_Pixel, CV_8UC1);
  cv::normalize(BEV_Image_Pixel, BEV_Image_Pixel, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::Mat BEV_Image_Pixel_copy = BEV_Image_Pixel.clone();
  //转为彩色的
  cv::cvtColor(BEV_Image_Pixel_copy, BEV_Image_Pixel_copy, cv::COLOR_GRAY2BGR);

  BEV_images.push_back(BEV_Image_Pixel);

  std::vector<cv::Point2f> corners;
  cv::goodFeaturesToTrack(BEV_Image_Pixel, corners, 100, image_quartity, Hamming_distance, cv::noArray(), 3, false, 0.04);
  std::cout << "corners size: " << corners.size() << std::endl;

  //增加描述信息
  std::vector<CornerDescriptor> corner_data;
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr corner_points(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  for (const auto &corner : corners){

      pcl::PointXYZINormal point;
      point.x = (corner.x - BEV_X_NUM / 2) * (BEV_X_MAX * 2 / BEV_X_NUM);
      point.y = (corner.y - BEV_Y_NUM / 2) * (BEV_Y_MAX * 2 / BEV_Y_NUM);
      point.z = result.first(corner.y, corner.x);
      point.intensity = result.second(corner.y, corner.x);
//      corner_points->push_back(point);

      //提取局部描述信息(顺时针)
      std::array<int, 9> neighborhood = {0};
      ///注意这里x和y的调换
      const int x = corner.y, y = corner.x;
      const auto &points_in_cell = BEV_point_map[x][y];
      //这里是要计算z值的分布
      double z_threshold = min_z;
      double z_sum = 0.0;
      std::vector<double> filtered_z;
      for (const auto &pt : points_in_cell) {
          if (pt.z() > z_threshold) {
              filtered_z.push_back(pt.z());
              z_sum += pt.z();
          }
      }
      double z_mean = z_sum / filtered_z.size();
      double z_variance = 0;
      if(filtered_z.size() > 0){
          for (double z : filtered_z) {
              z_variance += (z - z_mean) * (z - z_mean);
          }
          z_variance /= filtered_z.size();
      }
      double entropy = (1/2.0) * (1 + std::log(2 * M_PI)) + 0.5 * std::log(z_variance);
      double score = 0;
      if(filtered_z.size() > 0){
          for (double z : filtered_z) {
              double gaussian_value = -0.5 * (z - z_mean) * (z - z_mean) / z_variance;
              score += std::exp(gaussian_value);
          }
      }

      //下面是要存储周围3*3方格内的密度
      const int dy[] = {-1, -1, 0, 1, 1, 1, 0, -1}; // 左, 左上, 上, 右上, 右, 右下, 下, 左下
      const int dx[] = {0, -1, -1, -1, 0, 1, 1, 1};
      int density_count = result.second(x, y);
      for (int i = 0; i < 8; ++i) {
          int nx = x + dx[i];
          int ny = y + dy[i];
          //rows和cols可能反了
          if (nx >= 0 && nx < result.second.rows() && ny >= 0 && ny < result.second.cols()) {
              neighborhood[i] = result.second(nx, ny); // 使用 `result.second` 填充密度值
              density_count += neighborhood[i];
          } else {
              neighborhood[i] = 0.0; // 边界外填充为 0
          }
      }
      ///第9个元素存储中心值
      neighborhood[8] = result.second(x, y);

      corner_points->push_back(point);
      Eigen::Vector2i pos(x, y);
      //本来最后两个是z_mean和z_variance，现在分别换为概率密度得分和熵
//      CornerDescriptor descriptor(point, neighborhood, pos, z_mean, z_variance);
      CornerDescriptor descriptor(point, neighborhood, pos, score, entropy);
      corner_data.push_back(descriptor);

      ///增加用来可视化的
      cv::circle(BEV_Image_Pixel_copy, corner, 3, cv::Scalar(0, 0, 255), 2);
  }

  corner_cloud_vec_.push_back(corner_points);
  ///增加用来可视化
//  std::string bev_with_corners_filename = "/home/tkw/IFTD/src/iftd/img_density/"+ std::to_string(num) + "_corners.png";
//  cv::imwrite(bev_with_corners_filename, BEV_Image_Pixel_copy);

  // std::cout << "[Description] corners size:" << corner_points->size()
            // << std::endl;

  // step4, generate stable triangle descriptors
  stds_vec.clear();
  build_IFTDesc(corner_data, stds_vec);
  world_poses.push_back(std::make_pair(translation, rotation));

  std::cout << "[Description] stds size:" << stds_vec.size() << std::endl;
}

cv::Mat IFTDescManager::Eigen2Mat(Eigen::MatrixXi &matrix){
  cv::Mat image(matrix.rows(), matrix.cols(), CV_32SC1);
  for (int i = 0; i < matrix.rows(); i++)
    for (int j = 0; j < matrix.cols(); j++){
      image.at<int>(i, j) = matrix(i, j);
    }

  return image;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXi> IFTDescManager::makeBEV(pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud, Eigen::Vector3d translation, Eigen::Matrix3d rotation)
{
  Eigen::MatrixXd BEV_IMAGE = Eigen::MatrixXd::Zero(BEV_X_NUM, BEV_Y_NUM);
  Eigen::MatrixXi BEV_IMAGE_Pixel = Eigen::MatrixXi::Zero(BEV_X_NUM, BEV_Y_NUM);
  Eigen::MatrixXd BEV_IMAGE_density = Eigen::MatrixXd::Zero(BEV_X_NUM, BEV_Y_NUM);
  std::bitset<24> BEV_IMAGE_Temp[BEV_X_NUM][BEV_Y_NUM] = {};

  double BEV_X_GAP = BEV_X_MAX * 2 / BEV_X_NUM;
  double BEV_Y_GAP = BEV_Y_MAX * 2 / BEV_Y_NUM;

  for (int pt_idx = 0; pt_idx < input_cloud->size(); pt_idx++)
  {
    Eigen::Vector3d point_ori(input_cloud->points[pt_idx].x, input_cloud->points[pt_idx].y, input_cloud->points[pt_idx].z);

    int idx = point_ori.x() / BEV_X_GAP + BEV_X_NUM / 2;
    int idy = point_ori.y() / BEV_Y_GAP + BEV_Y_NUM / 2;
    double point_height = point_ori.z() + lidar_height;

    if (point_height < 0.3 || idx < 0 || idx >= BEV_X_NUM || idy < 0 || idy >= BEV_Y_NUM)
      continue;
    BEV_IMAGE_density(idx, idy)++;
    int point_height_ = int(point_height / 0.3);
    if (point_height > BEV_IMAGE(idx, idy))
      BEV_IMAGE(idx, idy) = point_height;
    BEV_IMAGE_Temp[idx][idy] |= (1 << point_height_);
  }

  for (int i = 0; i < BEV_IMAGE_Pixel.rows(); i++)
    for (int j = 0; j < BEV_IMAGE_Pixel.cols(); j++)
    {
      int count = BEV_IMAGE_Temp[i][j].count();
      BEV_IMAGE_Pixel(i, j) = count;
    }

  return std::make_pair(BEV_IMAGE_density, BEV_IMAGE_Pixel);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXi> IFTDescManager::makeDensityBEV(pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                                                                           std::vector<std::vector<std::vector<Eigen::Vector3d>>> &BEV_point_map,
                                                                           double &min_z){
    Eigen::MatrixXi BEV_IMAGE_density = Eigen::MatrixXi::Zero(BEV_X_NUM, BEV_Y_NUM);
    Eigen::MatrixXd BEV_IMAGE_avg_z = Eigen::MatrixXd::Zero(BEV_X_NUM, BEV_Y_NUM);
    Eigen::MatrixXd BEV_IMAGE_z_sum = Eigen::MatrixXd::Zero(BEV_X_NUM, BEV_Y_NUM);
    double BEV_X_GAP = BEV_X_MAX * 2 / BEV_X_NUM;
    double BEV_Y_GAP = BEV_Y_MAX * 2 / BEV_Y_NUM;

    // 初始化栅格点云映射
    BEV_point_map.resize(BEV_X_NUM, std::vector<std::vector<Eigen::Vector3d>>(BEV_Y_NUM));

    for (int pt_idx = 0; pt_idx < input_cloud->size(); pt_idx++)
    {
        Eigen::Vector3d point_ori(input_cloud->points[pt_idx].x, input_cloud->points[pt_idx].y, input_cloud->points[pt_idx].z);

        int idx = point_ori.x() / BEV_X_GAP + BEV_X_NUM / 2;
        int idy = point_ori.y() / BEV_Y_GAP + BEV_Y_NUM / 2;

        if (idx < 0 || idx >= BEV_X_NUM || idy < 0 || idy >= BEV_Y_NUM)
            continue;
        BEV_IMAGE_density(idx, idy)++;
        BEV_IMAGE_z_sum(idx, idy) += point_ori.z();

        // 将点存入栅格
        BEV_point_map[idx][idy].emplace_back(point_ori);
        min_z = std::min(min_z, point_ori.z());
    }

    // 找到点数最小值和最大值
    int min_density = std::numeric_limits<int>::max();
    int max_density = std::numeric_limits<int>::min();

    for (int i = 0; i < BEV_X_NUM; i++)
    {
        for (int j = 0; j < BEV_Y_NUM; j++)
        {
            if (BEV_IMAGE_density(i, j) > 0)
            {
                min_density = std::min(min_density, BEV_IMAGE_density(i, j));
                max_density = std::max(max_density, BEV_IMAGE_density(i, j));
            }
        }
    }

    for (int i = 0; i < BEV_X_NUM; i++)
    {
        for (int j = 0; j < BEV_Y_NUM; j++)
        {
            if (BEV_IMAGE_density(i, j) > 0)
            {
                if(BEV_IMAGE_density(i, j) < Density_threshold * (max_density - min_density)){
                    BEV_IMAGE_density(i, j) = 0;
                    BEV_IMAGE_avg_z(i, j) = 0;
                }
                else{
                    BEV_IMAGE_avg_z(i, j) = BEV_IMAGE_z_sum(i, j) / BEV_IMAGE_density(i, j);
                }
            }
            else
            {
                BEV_IMAGE_avg_z(i, j) = 0; // 如果栅格没有点，平均 z 设为 0
            }
        }
    }

    return std::make_pair(BEV_IMAGE_avg_z, BEV_IMAGE_density);
}

BEVResult IFTDescManager::makeDensityGlobalBEV(
        pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud)
{

    struct GridData{
        int count = 0;
        double z_sum = 0.0;
    };
    std::map<Eigen::Array2i, GridData, ComparePixels> grid_data;
    double max_points = std::numeric_limits<double>::min();
    double min_points = std::numeric_limits<double>::max();
    Eigen::Array2i lower_bound_coordinates = Eigen::Array2i::Constant(std::numeric_limits<int>::max());
    Eigen::Array2i upper_bound_coordinates = Eigen::Array2i::Constant(std::numeric_limits<int>::min());
    double resolution = Density_resolution;

    // 离散化函数
    auto discretize = [resolution](const Eigen::Vector3d &point) -> Eigen::Array2i {
        return (point.head<2>() / resolution).array().floor().cast<int>();
    };

    // 遍历点云
    for (const auto &point : input_cloud->points)
    {
        Eigen::Vector3d point_vec(point.x, point.y, point.z);
        const auto pixel = discretize(point_vec);

        // 统计点数
        auto &data = grid_data[pixel];
        data.count++;
        data.z_sum += point.z;

        // 更新边界
        lower_bound_coordinates = lower_bound_coordinates.min(pixel);
        upper_bound_coordinates = upper_bound_coordinates.max(pixel);
    }

    // 计算矩阵大小
    const auto rows_and_columns = upper_bound_coordinates - lower_bound_coordinates + Eigen::Array2i::Constant(1);
    const int n_rows = rows_and_columns.x();
    const int n_cols = rows_and_columns.y();

    // 初始化结果矩阵
    Eigen::MatrixXi BEV_IMAGE_density = Eigen::MatrixXi::Zero(n_rows, n_cols);
    Eigen::MatrixXd BEV_IMAGE_avg_z = Eigen::MatrixXd::Zero(n_rows, n_cols);

    // 填充结果矩阵
    for (const auto &[pixel, data] : grid_data)
    {
        const auto pixel_idx = pixel - lower_bound_coordinates;

        // 填充密度
        BEV_IMAGE_density(pixel_idx.x(), pixel_idx.y()) = data.count;
        BEV_IMAGE_avg_z(pixel_idx.x(), pixel_idx.y()) = data.z_sum / data.count;
    }

    // 返回密度和平均 z 值矩阵
    return BEVResult{
        .avg_z_matrix = BEV_IMAGE_avg_z,
        .density_matrix = BEV_IMAGE_density,
        .lower_bound = lower_bound_coordinates,
        .upper_bound = upper_bound_coordinates,
    };

}


void IFTDescManager::SearchLoop(
    const std::vector<IFTDesc> &stds_vec, std::pair<int, double> &loop_result,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &loop_transform,
    std::vector<std::pair<IFTDesc, IFTDesc>> &loop_std_pair) {

  if (stds_vec.size() == 0) {
    ROS_ERROR_STREAM("No IFTDescs!");
    loop_result = std::pair<int, double>(-1, 0);
    return;
  }
  // step1, select candidates, default number 50
  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<STDMatchList> candidate_matcher_vec;

  auto t_candidate_selector_begin = std::chrono::high_resolution_clock::now();
  candidate_selector(stds_vec, candidate_matcher_vec); //粗匹配
  auto t_candidate_selector_end = std::chrono::high_resolution_clock::now();
  candidate_selector_time.push_back(time_inc(t_candidate_selector_end, t_candidate_selector_begin));

  // std::cout << "candidate_matcher_vec size: " << candidate_matcher_vec.size() << std::endl;

  auto t2 = std::chrono::high_resolution_clock::now();

  // step2, select best candidates from rough candidates
  double best_score = 0;
  unsigned int best_candidate_id = -1;
  unsigned int triggle_candidate = -1;
  std::pair<Eigen::Vector3d, Eigen::Matrix3d> best_transform;
  std::vector<std::pair<IFTDesc, IFTDesc>> best_sucess_match_vec;

  auto t_candidate_verify_begin = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < candidate_matcher_vec.size(); i++) 
  {
    double verify_score = -1;
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> relative_pose;
    std::vector<std::pair<IFTDesc, IFTDesc>> sucess_match_vec;
    candidate_verify(candidate_matcher_vec[i], verify_score, relative_pose,   //细匹配
                     sucess_match_vec);
 
    if (verify_score > best_score) {
      best_score = verify_score;
      best_candidate_id = candidate_matcher_vec[i].match_id_.second;
      best_transform = relative_pose;
      best_sucess_match_vec = sucess_match_vec;
      triggle_candidate = i;
    }
  }

  auto t_candidate_verify_end = std::chrono::high_resolution_clock::now();
  candidate_verify_time.push_back(time_inc(t_candidate_verify_end, t_candidate_verify_begin));

    double mean_candidate_selector =
      std::accumulate(candidate_selector_time.begin(), candidate_selector_time.end(), 0) * 1.0 /
      candidate_selector_time.size();
    double mean_candidate_verify =
      std::accumulate(candidate_verify_time.begin(), candidate_verify_time.end(), 0) * 1.0 /
      candidate_verify_time.size();
    // std::cout << "-----------Mean time for candidate selector: " << mean_candidate_selector
              // << "ms, candidate verify: " << mean_candidate_verify << "ms" << std::endl;

  auto t3 = std::chrono::high_resolution_clock::now();

  // std::cout << "[Time] candidate selector: " << time_inc(t2, t1)
  //           << " ms, candidate verify: " << time_inc(t3, t2) << "ms"
  //           << std::endl;

  if (best_score > config_setting_.image_threshold_)
  {
    loop_result = std::pair<int, double>(best_candidate_id, best_score);
    loop_transform = best_transform;
    loop_std_pair = best_sucess_match_vec;

    std::pair<int , int> loop_pair = std::pair<int , int>(current_frame_id_ , best_candidate_id);
    loop_pairs.push_back(loop_pair);
    relative_poses.push_back(best_transform);

    return;
  }
  
  ///一个trick，依赖于上一帧正确回环上的情景
  else if(loop_pairs.size() > 0 && loop_pairs.back().second != -1)
  {
    int history_frame_id = loop_pairs.back().second;

    //Define the transformation matrix for the camera and radar coordinate systems.
    Eigen::Matrix3d lidar2image;
    lidar2image << 1,0,0,0,1,0,0,0,1;

    Eigen::Vector3d current_t_ = world_poses[current_frame_id_].first;
    Eigen::Matrix3d current_rot_ = world_poses[current_frame_id_].second;
    Eigen::Vector3d last_t_ = world_poses[current_frame_id_ - 1].first;
    Eigen::Matrix3d last_rot_ = world_poses[current_frame_id_ - 1].second;

    Eigen::Vector3d temp_translation = last_rot_.transpose() * (current_t_ - last_t_);
    Eigen::Matrix3d temp_rotation = last_rot_.transpose() * current_rot_;
    Eigen::Matrix3d xy2uv_rotation;
    xy2uv_rotation << 0,1,0,1,0,0,0,0,1;
    temp_translation = xy2uv_rotation * temp_translation;

    Eigen::Vector3d current_relative_t = Eigen::Vector3d::Zero();
    current_relative_t.head(2) = temp_translation.head(2);
    Eigen::Matrix3d current_relative_rot = Eigen::Matrix3d::Identity();
    Eigen::Matrix2d temp_rotation_2d = temp_rotation.topLeftCorner(2,2);
    temp_rotation_2d.col(0).normalize();
    temp_rotation_2d.col(1).normalize();
    current_relative_rot.topLeftCorner(2,2) = temp_rotation_2d;

    Eigen::Vector3d last_relative_t = relative_poses.back().first;
    Eigen::Matrix3d last_relative_rot = relative_poses.back().second;

    Eigen::Vector3d relative_t = last_relative_rot * current_relative_t + last_relative_t;

    Eigen::Matrix3d relative_rot = last_relative_rot * current_relative_rot;

    double verify_score = image_similarity_verify(current_frame_id_, history_frame_id, relative_rot, relative_t);

    // if (verify_score > config_setting_.image_threshold_)
    if (verify_score > config_setting_.image_threshold_ && relative_t.norm() < config_setting_.distance_threshold_) //distance threshold
    {
      loop_result = std::pair<int, double>(history_frame_id, verify_score);
      loop_transform = std::pair<Eigen::Vector3d, Eigen::Matrix3d>(relative_t, relative_rot);
      // loop_std_pair = best_sucess_match_vec;

      std::pair<int, int> loop_pair = std::pair<int, int>(current_frame_id_, history_frame_id);
      loop_pairs.push_back(loop_pair);
      relative_poses.push_back(loop_transform);
      return;
    }
    else
    {
      std::pair<int, int> loop_pair = std::pair<int, int>(current_frame_id_, -1);
      loop_pairs.push_back(loop_pair);
      relative_poses.push_back(best_transform);
      loop_result = std::pair<int, double>(-1, 0);
      return;
    }
  }

  else
  {
    std::pair<int , int> loop_pair = std::pair<int , int>(current_frame_id_ , -1);
    loop_pairs.push_back(loop_pair);
    relative_poses.push_back(best_transform);
    loop_result = std::pair<int, double>(-1, 0);
    return;
  }
}

void IFTDescManager::AddIFTDescs(const std::vector<IFTDesc> &stds_vec) {
  // update frame id
  current_frame_id_++;
  for (auto single_std : stds_vec) {
    // calculate the position of single std
    IFTDesc_LOC position;
    position.x = (int)(single_std.side_length_[0] + 0.5);
    position.y = (int)(single_std.side_length_[1] + 0.5);
    position.z = (int)(single_std.side_length_[2] + 0.5);
    position.a = (int)(single_std.angle_[0]);
    position.b = (int)(single_std.angle_[1]);
    position.c = (int)(single_std.angle_[2]);
    auto iter = data_base_.find(position);
    if (iter != data_base_.end()) {
      data_base_[position].push_back(single_std);
    } else {
      std::vector<IFTDesc> descriptor_vec;
      descriptor_vec.push_back(single_std);
      data_base_[position] = descriptor_vec;
    }
  }
  return;
}

void IFTDescManager::build_IFTDesc(
    const std::vector<CornerDescriptor> &corner_data,
    std::vector<IFTDesc> &stds_vec) {
  stds_vec.clear();
  double scale = 1.0 / config_setting_.std_side_resolution_;
  int near_num = config_setting_.descriptor_near_num_;
  double max_dis_threshold = config_setting_.descriptor_max_len_;
  double min_dis_threshold = config_setting_.descriptor_min_len_;

  std::unordered_map<VOXEL_LOC, bool> feat_map;
  // 转换数据为 point cloud 以支持 KdTree
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr corner_points(new pcl::PointCloud<pcl::PointXYZINormal>());
  for (const auto &data : corner_data) {
      corner_points->push_back(data.point);
  }

  pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZINormal>);
  kd_tree->setInputCloud(corner_points);
  std::vector<int> pointIdxNKNSearch(near_num);
  std::vector<float> pointNKNSquaredDistance(near_num);
  // Search N nearest corner points to form stds.
  for (size_t i = 0; i < corner_data.size(); i++) {

    const auto &searchPoint = corner_data[i].point;
    const auto &searchDensity = corner_data[i].neighborhood_density;  // 当前点的密度值描述
    const auto &searchPos = corner_data[i].position;
    Eigen::Vector2d searchZ(corner_data[i].mean_z, corner_data[i].var_z);
    if (kd_tree->nearestKSearch(searchPoint, near_num, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      for (int m = 1; m < near_num - 1; m++) {
        for (int n = m + 1; n < near_num; n++) {
          pcl::PointXYZINormal p1 = searchPoint;
          pcl::PointXYZINormal p2 = corner_data[pointIdxNKNSearch[m]].point;
          pcl::PointXYZINormal p3 = corner_data[pointIdxNKNSearch[n]].point;
          std::array<int, 9> d1 = searchDensity;
          std::array<int, 9> d2 = corner_data[pointIdxNKNSearch[m]].neighborhood_density;
          std::array<int, 9> d3 = corner_data[pointIdxNKNSearch[n]].neighborhood_density;
          Eigen::Vector2i pos1 = searchPos;
          Eigen::Vector2i pos2 = corner_data[pointIdxNKNSearch[m]].position;
          Eigen::Vector2i pos3 = corner_data[pointIdxNKNSearch[n]].position;
          Eigen::Vector2d z1 = searchZ;
          Eigen::Vector2d z2(corner_data[pointIdxNKNSearch[m]].mean_z, corner_data[pointIdxNKNSearch[m]].var_z);
          Eigen::Vector2d z3(corner_data[pointIdxNKNSearch[n]].mean_z, corner_data[pointIdxNKNSearch[n]].var_z);


            double a = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) +
                          pow(p1.z - p2.z, 2));
          double b = sqrt(pow(p1.x - p3.x, 2) + pow(p1.y - p3.y, 2) +
                          pow(p1.z - p3.z, 2));
          double c = sqrt(pow(p3.x - p2.x, 2) + pow(p3.y - p2.y, 2) +
                          pow(p3.z - p2.z, 2));
          double angle_a = std::acos((b * b + c * c - a * a) / (2 * c * b)) * 180.0 / M_PI;
          double angle_b = std::acos((a * a + c * c - b * b) / (2 * a * c)) * 180.0 / M_PI;
          double angle_c = std::acos((a * a + b * b - c * c) / (2 * a * b)) * 180.0 / M_PI;
          if (a > max_dis_threshold || b > max_dis_threshold ||
              c > max_dis_threshold || a < min_dis_threshold ||
              b < min_dis_threshold || c < min_dis_threshold) {
            continue;
          }
          if (angle_a > 175 || angle_b > 175 || angle_c > 175
              || angle_a < 5 || angle_b < 5 || angle_c < 5) {
            continue;
          }
          // re-range the vertex by the side length
          double temp;
          double temp_angle;
          Eigen::Vector3d A, B, C;
          Eigen::Vector3i l1, l2, l3;
          Eigen::Vector3i l_temp;
          l1 << 1, 2, 0;
          l2 << 1, 0, 3;
          l3 << 0, 2, 3;
          ///这样后排序为c>b>a
          if (a > b) {
            temp = a;
            a = b;
            b = temp;
            temp_angle = angle_a;
            angle_a = angle_b;
            angle_b = temp_angle;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          if (b > c) {
            temp = b;
            b = c;
            c = temp;
            temp_angle = angle_b;
            angle_b = angle_c;
            angle_c = temp_angle;
            l_temp = l2;
            l2 = l3;
            l3 = l_temp;
          }
          if (a > b) {
            temp = a;
            a = b;
            b = temp;
            temp_angle = angle_a;
            angle_a = angle_b;
            angle_b = temp_angle;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          // check augnmentation
          pcl::PointXYZ d_p;
          d_p.x = a * 1000;
          d_p.y = b * 1000;
          d_p.z = c * 1000;
          VOXEL_LOC position((int64_t)d_p.x, (int64_t)d_p.y, (int64_t)d_p.z);
          auto iter = feat_map.find(position);
          Eigen::Vector3d normal_1, normal_2, normal_3;
          std::array<int, 9> density_A = {0};
          std::array<int, 9> density_B = {0};
          std::array<int, 9> density_C = {0};
          Eigen::Vector2d mean_var_A(0, 0);
          Eigen::Vector2d mean_var_B(0, 0);
          Eigen::Vector2d mean_var_C(0, 0);
          if (iter == feat_map.end()) {
            Eigen::Vector3d vertex_attached;
            if (l1[0] == l2[0]) {
              A << p1.x, p1.y, p1.z;
              // normal_1 << p1.normal_x, p1.normal_y, p1.normal_z;
              vertex_attached[0] = p1.intensity;
              density_A = d1;
              mean_var_A = z1;
            } else if (l1[1] == l2[1]) {
              A << p2.x, p2.y, p2.z;
              // normal_1 << p2.normal_x, p2.normal_y, p2.normal_z;
              vertex_attached[0] = p2.intensity;
              density_A = d2;
              mean_var_A = z2;
            } else {
              A << p3.x, p3.y, p3.z;
              // normal_1 << p3.normal_x, p3.normal_y, p3.normal_z;
              vertex_attached[0] = p3.intensity;
              density_A = d3;
              mean_var_A = z3;
            }
            if (l1[0] == l3[0]) {
              B << p1.x, p1.y, p1.z;
              // normal_2 << p1.normal_x, p1.normal_y, p1.normal_z;
              vertex_attached[1] = p1.intensity;
              density_B = d1;
              mean_var_B = z1;
            } else if (l1[1] == l3[1]) {
              B << p2.x, p2.y, p2.z;
              // normal_2 << p2.normal_x, p2.normal_y, p2.normal_z;
              vertex_attached[1] = p2.intensity;
              density_B = d2;
              mean_var_B = z2;
            } else {
              B << p3.x, p3.y, p3.z;
              // normal_2 << p3.normal_x, p3.normal_y, p3.normal_z;
              vertex_attached[1] = p3.intensity;
              density_B = d3;
              mean_var_B = z3;
            }
            if (l2[0] == l3[0]) {
              C << p1.x, p1.y, p1.z;
              // normal_3 << p1.normal_x, p1.normal_y, p1.normal_z;
              vertex_attached[2] = p1.intensity;
              density_C = d1;
              mean_var_C = z1;
            } else if (l2[1] == l3[1]) {
              C << p2.x, p2.y, p2.z;
              // normal_3 << p2.normal_x, p2.normal_y, p2.normal_z;
              vertex_attached[2] = p2.intensity;
              density_C = d2;
              mean_var_C = z2;
            } else {
              C << p3.x, p3.y, p3.z;
              // normal_3 << p3.normal_x, p3.normal_y, p3.normal_z;
              vertex_attached[2] = p3.intensity;
              density_C = d3;
              mean_var_C = z3;
            }
            IFTDesc single_descriptor;
            single_descriptor.vertex_A_ = A;
            single_descriptor.vertex_B_ = B;
            single_descriptor.vertex_C_ = C;
            single_descriptor.center_ = (A + B + C) / 3;
            single_descriptor.vertex_attached_ = vertex_attached; //gaidong
            single_descriptor.side_length_ << scale * a, scale * b, scale * c;
            // single_descriptor.angle_[0] = fabs(5 * normal_1.dot(normal_2)); //gaidong
            // single_descriptor.angle_[1] = fabs(5 * normal_1.dot(normal_3));
            // single_descriptor.angle_[2] = fabs(5 * normal_3.dot(normal_2));
            single_descriptor.angle_[0] = angle_a;
            single_descriptor.angle_[1] = angle_b;
            single_descriptor.angle_[2] = angle_c;

            single_descriptor.density_A = density_A;
            single_descriptor.density_B = density_B;
            single_descriptor.density_C = density_C;

            single_descriptor.density_mean[0] = std::accumulate(density_A.begin(), density_A.end(), 0.0) / density_A.size();
            single_descriptor.density_mean[1] = std::accumulate(density_B.begin(), density_B.end(), 0.0) / density_B.size();
            single_descriptor.density_mean[2] = std::accumulate(density_C.begin(), density_C.end(), 0.0) / density_C.size();

            int x_center = static_cast<int>(std::round((pos1.x() + pos2.x() + pos3.x())/3.0));
            int y_center = static_cast<int>(std::round((pos1.y() + pos2.y() + pos3.y())/3.0));
            single_descriptor.density_center = matrix_density(x_center, y_center);

            single_descriptor.mean_var_A = mean_var_A;
            single_descriptor.mean_var_B = mean_var_B;
            single_descriptor.mean_var_C = mean_var_C;

            // single_descriptor.angle << 0, 0, 0;
            single_descriptor.frame_id_ = current_frame_id_;
            // Eigen::Matrix3d triangle_positon;
            feat_map[position] = true;
            stds_vec.push_back(single_descriptor);
          }
        }
      }
    }
  }
};

void IFTDescManager::candidate_selector(
    const std::vector<IFTDesc> &stds_vec,
    std::vector<STDMatchList> &candidate_matcher_vec) {
  double match_array[MAX_FRAME_N] = {0};
  std::vector<std::pair<IFTDesc, IFTDesc>> match_vec;
  std::vector<int> match_index_vec;
  std::vector<Eigen::Vector3i> voxel_round;
  // voxel_round.push_back(Eigen::Vector3i(0, 0, 0));
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        Eigen::Vector3i voxel_inc(x, y, z);
        voxel_round.push_back(voxel_inc);
      }
    }
  }

  std::vector<bool> useful_match(stds_vec.size());
  std::vector<std::vector<size_t>> useful_match_index(stds_vec.size());
  std::vector<std::vector<IFTDesc_LOC>> useful_match_position(stds_vec.size());
  std::vector<size_t> index(stds_vec.size());
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
    useful_match[i] = false;
  }
  // speed up matching  
  int dis_match_cnt = 0;
  int final_match_cnt = 0;
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif 
  for (size_t i = 0; i < stds_vec.size(); i++) { 
    IFTDesc src_std = stds_vec[i];
    IFTDesc_LOC position;
    int best_index = 0;
    IFTDesc_LOC best_position;
    double dis_threshold =
        src_std.side_length_.norm() * config_setting_.rough_dis_threshold_;
    for (auto voxel_inc : voxel_round) {
      position.x = (int)(src_std.side_length_[0] + voxel_inc[0]);
      position.y = (int)(src_std.side_length_[1] + voxel_inc[1]);
      position.z = (int)(src_std.side_length_[2] + voxel_inc[2]);
      Eigen::Vector3d voxel_center((double)position.x + 0.5,
                                   (double)position.y + 0.5,
                                   (double)position.z + 0.5);
      if ((src_std.side_length_ - voxel_center).norm() < 1.5) { 
        auto iter = data_base_.find(position);
        if (iter != data_base_.end()) {
            ///找到对应的哈希键后，遍历这里的每一帧，当帧数间隔足够且三角形边长差较小，且强度或者密度判断
          for (size_t j = 0; j < data_base_[position].size(); j++) {
            if ((src_std.frame_id_ - data_base_[position][j].frame_id_) >
                config_setting_.skip_near_num_) {
              double dis =
                  (src_std.side_length_ - data_base_[position][j].side_length_)
                      .norm();
              // rough filter with side lengths
              if (dis < dis_threshold) {
                dis_match_cnt++;

                ///my way
//                double vertex_attach_diff = pearsonCorrelation(src_std.density_A, data_base_[position][j].density_A)
//                        + pearsonCorrelation(src_std.density_B, data_base_[position][j].density_B)
//                        + pearsonCorrelation(src_std.density_C, data_base_[position][j].density_C);
//                //相当于平均相关度>0.5
//                if(vertex_attach_diff >1.5){
//                    final_match_cnt++;
//                    useful_match[i] = true;
//                    useful_match_position[i].push_back(position);
//                    useful_match_index[i].push_back(j);
//                }
                ///old way
//                  double vertex_attach_diff =
//                          2.0 *
//                          (src_std.vertex_attached_ -
//                           data_base_[position][j].vertex_attached_)
//                                  .norm() /
//                          (src_std.vertex_attached_ +
//                           data_base_[position][j].vertex_attached_)
//                                  .norm();
                  ///some change
                  //1、平均z值或是概率密度分布
//                  Eigen::Vector3d mean_z_now(src_std.mean_var_A.x(), src_std.mean_var_B.x(), src_std.mean_var_C.x());
//                  Eigen::Vector3d mean_z_old(data_base_[position][j].mean_var_A.x(), data_base_[position][j].mean_var_B.x(), data_base_[position][j].mean_var_C.x());
//                  double vertex_attach_diff =
//                          2.0 *
//                          (mean_z_now - mean_z_old).norm()
//                          /(mean_z_now + mean_z_old).norm();
                  //2.1、z的方差或是熵
//                  Eigen::Vector3d mean_z_now(src_std.mean_var_A.y(), src_std.mean_var_B.y(), src_std.mean_var_C.y());
//                  Eigen::Vector3d mean_z_old(data_base_[position][j].mean_var_A.y(), data_base_[position][j].mean_var_B.y(), data_base_[position][j].mean_var_C.y());
//                  double vertex_attach_diff =
//                          2.0 *
//                          std::abs((mean_z_now.x() + mean_z_now.y() + mean_z_now.z()) - (mean_z_old.x() + mean_z_old.y() + mean_z_old.z()))
//                          /((mean_z_now.x() + mean_z_now.y() + mean_z_now.z()) + (mean_z_old.x() + mean_z_old.y() + mean_z_old.z()));
                  //2.2、熵的余弦相似度
//                  Eigen::Vector3d mean_z_now(src_std.mean_var_A.y(), src_std.mean_var_B.y(), src_std.mean_var_C.y());
//                  Eigen::Vector3d mean_z_old(data_base_[position][j].mean_var_A.y(), data_base_[position][j].mean_var_B.y(), data_base_[position][j].mean_var_C.y());
//                  double vertex_attach_diff = cosineSimilarity(mean_z_now, mean_z_old);
                  //3.1 score
//                  Eigen::Vector3d mean_z_now(src_std.mean_var_A.x(), src_std.mean_var_B.x(), src_std.mean_var_C.x());
//                  Eigen::Vector3d mean_z_old(data_base_[position][j].mean_var_A.x(), data_base_[position][j].mean_var_B.x(), data_base_[position][j].mean_var_C.x());
//                  double vertex_attach_diff =
//                          2.0 *
//                          (mean_z_now - mean_z_old).norm()
//                          /(mean_z_now + mean_z_old).norm();
                  //3.2 score的余弦相似度
                  Eigen::Vector3d mean_z_now(src_std.mean_var_A.x(), src_std.mean_var_B.x(), src_std.mean_var_C.x());
                  Eigen::Vector3d mean_z_old(data_base_[position][j].mean_var_A.x(), data_base_[position][j].mean_var_B.x(), data_base_[position][j].mean_var_C.x());
                  double vertex_attach_diff = cosineSimilarity(mean_z_now, mean_z_old);

//                  double vertex_attach_diff =
//                          2.0 *
//                          abs(src_std.density_center -
//                           data_base_[position][j].density_center)/
//                          (src_std.density_center +
//                           data_base_[position][j].density_center);
//                  double vertex_attach_diff =
//                          2.0 *
//                          (src_std.density_mean -
//                           data_base_[position][j].density_mean)
//                                  .norm() /
//                          (src_std.density_mean +
//                           data_base_[position][j].density_mean)
//                                  .norm();

                  if (vertex_attach_diff >
                      config_setting_.vertex_diff_threshold_) {

                      final_match_cnt++;
                      useful_match[i] = true;
                      useful_match_position[i].push_back(position);
                      useful_match_index[i].push_back(j);
                  }

//                if(src_std.density_mean.norm() < 2 || data_base_[position][j].density_mean.norm() < 2){
//
//                }
//                else{
//                    double density_attach_diff =
//                            2.0 *
//                            (src_std.density_mean -
//                             data_base_[position][j].density_mean)
//                                    .norm() /
//                            (src_std.density_mean +
//                             data_base_[position][j].density_mean)
//                                    .norm();
//
//                    if (density_attach_diff <
//                        config_setting_.density_diff_threshold) {
//
//                        final_match_cnt++;
//                        useful_match[i] = true;
//                        useful_match_position[i].push_back(position);
//                        useful_match_index[i].push_back(j);
//                    }
//                }
              }
            }
          }
        }
      }
    }
  }
  // std::cout << "dis match num:" << dis_match_cnt
  //           << ", final match num:" << final_match_cnt << std::endl;

  // record match index
  std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i>>  
      index_recorder;
  for (size_t i = 0; i < useful_match.size(); i++) {
    if (useful_match[i]) {  
      for (size_t j = 0; j < useful_match_index[i].size(); j++) {  
        match_array[data_base_[useful_match_position[i][j]] 
                              [useful_match_index[i][j]]
                                  .frame_id_] += 1;  
        Eigen::Vector2i match_index(i, j);
        index_recorder.push_back(match_index); 
        match_index_vec.push_back(
            data_base_[useful_match_position[i][j]][useful_match_index[i][j]]
                .frame_id_);  
      }
    }
  }

  // select candidate according to the matching score
  for (int cnt = 0; cnt < config_setting_.candidate_num_; cnt++) { 
    double max_vote = 1;
    int max_vote_index = -1;
    for (int i = 0; i < MAX_FRAME_N; i++) { 
      if (match_array[i] > max_vote) {
        max_vote = match_array[i];
        max_vote_index = i;
      }
    }
    STDMatchList match_triangle_list;
    if (max_vote_index >= 0 && max_vote >= 10) {
      match_array[max_vote_index] = 0;
      match_triangle_list.match_id_.first = current_frame_id_;
      match_triangle_list.match_id_.second = max_vote_index; 
      for (size_t i = 0; i < index_recorder.size(); i++) {
        if (match_index_vec[i] == max_vote_index) { 
          std::pair<IFTDesc, IFTDesc> single_match_pair; 
          single_match_pair.first = stds_vec[index_recorder[i][0]];
          single_match_pair.second =
              data_base_[useful_match_position[index_recorder[i][0]]
                                              [index_recorder[i][1]]]
                        [useful_match_index[index_recorder[i][0]]
                                           [index_recorder[i][1]]];
          match_triangle_list.match_list_.push_back(single_match_pair);
        }
      }
      candidate_matcher_vec.push_back(match_triangle_list);
    } else {
      break;
    }
  }
}

// Get the best candidate frame by geometry check
void IFTDescManager::candidate_verify(
    const STDMatchList &candidate_matcher, double &verify_score,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &relative_pose,
    std::vector<std::pair<IFTDesc, IFTDesc>> &sucess_match_vec) {

  sucess_match_vec.clear();

  int skip_len = 1;
  int use_size = candidate_matcher.match_list_.size() / skip_len;
  double dis_threshold = 3.0;
  std::vector<size_t> index(use_size);
  std::vector<int> vote_list(use_size);
  std::vector<std::vector<Eigen::Vector3d>> vertexs_list(use_size);
  for (size_t i = 0; i < index.size(); i++) {
    index[i] = i;
  }
  std::mutex mylock;

#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (size_t i = 0; i < use_size; i++) {
    auto single_pair = candidate_matcher.match_list_[i * skip_len];
    int vote = 0;
    Eigen::Matrix3d test_rot;
    Eigen::Vector3d test_t;
    triangle_solver(single_pair, test_t, test_rot); 

    for (size_t j = 0; j < candidate_matcher.match_list_.size(); j++) { 
      auto verify_pair = candidate_matcher.match_list_[j];
      Eigen::Vector3d A = verify_pair.first.vertex_A_;
      Eigen::Vector3d A_transform = test_rot * A + test_t;
      Eigen::Vector3d B = verify_pair.first.vertex_B_;
      Eigen::Vector3d B_transform = test_rot * B + test_t;
      Eigen::Vector3d C = verify_pair.first.vertex_C_;
      Eigen::Vector3d C_transform = test_rot * C + test_t;
      double dis_A = (A_transform - verify_pair.second.vertex_A_).norm();
      double dis_B = (B_transform - verify_pair.second.vertex_B_).norm();
      double dis_C = (C_transform - verify_pair.second.vertex_C_).norm();
      if (dis_A < dis_threshold && dis_B < dis_threshold &&
          dis_C < dis_threshold) {
        vertex_test(A,vertexs_list[i]);
        vertex_test(B,vertexs_list[i]);
        vertex_test(C,vertexs_list[i]);
        vote++;
      }
    }
    mylock.lock();
    vote_list[i] = vote;
    mylock.unlock();
  }

  int max_vote_index = 0;
  int max_vote = 0;
  int vertex_num = 0;
  for (size_t i = 0; i < vote_list.size(); i++) {
    if (max_vote < vote_list[i]) {
      max_vote_index = i;
      max_vote = vote_list[i];
      vertex_num = vertexs_list[i].size();
    }
  }

  if (max_vote >= 5 && vertex_num >= 6 ) 
  {
    auto best_pair = candidate_matcher.match_list_[max_vote_index * skip_len];
    int vote = 0;
    Eigen::Matrix3d best_rot;
    Eigen::Vector3d best_t;
    triangle_solver(best_pair, best_t, best_rot);  

    if(best_t.norm() > config_setting_.distance_threshold_){  
      verify_score = -1;
      return;
    }

    relative_pose.first = best_t;
    relative_pose.second = best_rot;
    int num_success = 0;
    for (size_t j = 0; j < candidate_matcher.match_list_.size(); j++) { 
      auto verify_pair = candidate_matcher.match_list_[j];
      Eigen::Vector3d A = verify_pair.first.vertex_A_;
      Eigen::Vector3d A_transform = best_rot * A + best_t;
      Eigen::Vector3d B = verify_pair.first.vertex_B_;
      Eigen::Vector3d B_transform = best_rot * B + best_t;
      Eigen::Vector3d C = verify_pair.first.vertex_C_;
      Eigen::Vector3d C_transform = best_rot * C + best_t;
      double dis_A = (A_transform - verify_pair.second.vertex_A_).norm();
      double dis_B = (B_transform - verify_pair.second.vertex_B_).norm();
      double dis_C = (C_transform - verify_pair.second.vertex_C_).norm();
      if (dis_A < dis_threshold && dis_B < dis_threshold && dis_C < dis_threshold)
      {
        sucess_match_vec.push_back(verify_pair);
        num_success += 1;
      }
    }

//    verify_score = num_success/(1.0 * candidate_matcher.match_list_.size());
    verify_score = image_similarity_verify(candidate_matcher.match_id_.first, candidate_matcher.match_id_.second,best_rot, best_t);
  } 
  else 
  {
    verify_score = -1;
  }

}

void IFTDescManager::vertex_test(Eigen::Vector3d A,std::vector<Eigen::Vector3d> &vertexs)
{
  if (vertexs.size() == 0){
    vertexs.push_back(A);
    return;
  }
  for (const auto &vertex : vertexs)
    if ( (vertex - A).norm() < 1)
      return;
  vertexs.push_back(A);
}

double IFTDescManager::image_similarity_verify(int source_frame_id, int target_frame_id, const Eigen::Matrix3d &rot,const Eigen::Vector3d &t) 
{

  cv::Mat source_image = BEV_images[source_frame_id];
  cv::Mat target_image = BEV_images[target_frame_id];
  cv::Mat transform_matrix = (cv::Mat_<double>(2, 3) << rot(0, 0), rot(0, 1), t(0),rot(1, 0), rot(1, 1), t(1));

  cv::Mat transformed_image = image_transformer(source_image, transform_matrix);

  cv::Mat resized_transformed_image, resized_target_image;
  cv::resize(transformed_image, resized_transformed_image, cv::Size(config_setting_.image_reseize_, config_setting_.image_reseize_));
  cv::resize(target_image, resized_target_image, cv::Size(config_setting_.image_reseize_, config_setting_.image_reseize_));

  Eigen::VectorXd  resized_transformed_image_dhash = dhash(resized_transformed_image);
  Eigen::VectorXd  resized_target_image_dhash = dhash(resized_target_image);
    
  if(resized_transformed_image_dhash.norm() == 0 || resized_target_image_dhash.norm() == 0)
    return -1;
  double score = resized_transformed_image_dhash.dot(resized_target_image_dhash) / (resized_transformed_image_dhash.norm() * resized_target_image_dhash.norm());

  return score;
}

Eigen::VectorXd IFTDescManager::dhash(cv::Mat image)
{
  Eigen::VectorXd dhash(image.rows * (image.cols - 1));
  int index = 0;
  for (int i = 0; i < image.rows; i++){
    for (int j = 0; j < image.cols - 1; j++){
      if (image.at<uchar>(i, j) > image.at<uchar>(i, j + 1)){
        dhash[index] = 1.0;
      }
      else{
        dhash[index] = 0.0;
      }
      index++;
    }
  }
  return dhash;
}
  
cv::Mat IFTDescManager::image_transformer(cv::Mat &src, cv::Mat &transform_matrix)
{
  cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, src.type());
  double a = transform_matrix.at<double>(0, 0);
  double rotation_angle = acos(a) * 180 / CV_PI;
  
  cv::Point2f center(src.cols / 2.0, src.rows / 2.0);
  cv::Mat rot_mat = cv::getRotationMatrix2D(center, rotation_angle, 1.0);
  rot_mat.at<double>(0, 2) += transform_matrix.at<double>(0, 2) / (BEV_X_MAX * 2 / BEV_X_NUM);
  rot_mat.at<double>(1, 2) += transform_matrix.at<double>(1, 2) / (BEV_Y_MAX * 2 / BEV_Y_NUM);
  cv::warpAffine(src, dst, rot_mat, cv::Size(src.cols, src.rows));

  return dst;
}
void IFTDescManager::triangle_solver(std::pair<IFTDesc, IFTDesc> &std_pair,
                                    Eigen::Vector3d &t, Eigen::Matrix3d &rot) {
  Eigen::Matrix3d src = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d ref = Eigen::Matrix3d::Zero();
  src.col(0) = std_pair.first.vertex_A_ - std_pair.first.center_;
  src.col(1) = std_pair.first.vertex_B_ - std_pair.first.center_;
  src.col(2) = std_pair.first.vertex_C_ - std_pair.first.center_;
  ref.col(0) = std_pair.second.vertex_A_ - std_pair.second.center_;
  ref.col(1) = std_pair.second.vertex_B_ - std_pair.second.center_;
  ref.col(2) = std_pair.second.vertex_C_ - std_pair.second.center_;
  Eigen::Matrix3d covariance = src * ref.transpose();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(covariance, Eigen::ComputeThinU |
                                                        Eigen::ComputeThinV);
  Eigen::Matrix3d V = svd.matrixV();
  Eigen::Matrix3d U = svd.matrixU();
  rot = V * U.transpose();
  if (rot.determinant() < 0) {
    Eigen::Matrix3d K;
    K << 1, 0, 0, 0, 1, 0, 0, 0, -1;
    rot = V * K * U.transpose();
  }
  t = -rot * std_pair.first.center_ + std_pair.second.center_;
}