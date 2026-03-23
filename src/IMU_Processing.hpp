#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3Stamped.h>
#include "use-ikfom.hpp"

/// *************Preconfiguration

#define MAX_INI_COUNT (20)

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4,4) &T);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  void set_enu_init(bool enable);
  void set_gnss_ext(const V3D &t_gnss_in_imu, const M3D &R_imu_from_gnss);
  void gps_euler_cbk(const geometry_msgs::Vector3Stamped::ConstPtr &msg);
  bool   enu_inited = false;
  double enu_init_roll  = 0.0;
  double enu_init_pitch = 0.0;
  double enu_init_yaw   = 0.0;
  V3D    enu_init_pos   = V3D::Zero();
  Eigen::Matrix<double, 12, 12> Q;
  void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);

  ofstream fout_imu;
  V3D cov_acc; //vector
  V3D cov_gyr;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  double first_lidar_time;

 private:
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);

  PointCloudXYZI::Ptr cur_pcl_un_;
  sensor_msgs::ImuConstPtr last_imu_;
  deque<sensor_msgs::ImuConstPtr> v_imu_;
  vector<Pose6D> IMUpose; // //(时间，加速度，角速度，速度，位置，旋转矩阵）
  vector<M3D>    v_rot_pcl_;
  M3D Lidar_R_wrt_IMU;
  V3D Lidar_T_wrt_IMU;
  V3D mean_acc; //初始化时得到的加速度均值
  V3D mean_gyr;
  V3D angvel_last;
  V3D acc_s_last;
  double start_timestamp_;
  double last_lidar_end_time_;
  int    init_iter_num = 1; // 初始化时，imu数据迭代的次数
  bool   b_first_frame_ = true;
  bool   imu_need_init_ = true;
  bool   enu_init_en_ = false;
  bool   gps_euler_received_ = false;
  V3D    gps_euler_ned_ = V3D::Zero();    // GPS原始NED欧拉角 (roll_NED, pitch_NED, heading_NED) [rad]
  V3D    gnss_T_wrt_IMU_ = V3D::Zero();  // GNSS天线在IMU坐标系下的位置（杆臂）
  M3D    gnss_R_to_IMU_ = M3D::Identity();  // 从GNSS坐标系到IMU坐标系的旋转
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 1;
  Q = process_noise_cov();
  cov_acc       = V3D(0.1, 0.1, 0.1);
  cov_gyr       = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last     = Zero3d;
  Lidar_T_wrt_IMU = Zero3d;
  Lidar_R_wrt_IMU = Eye3d;
  last_imu_.reset(new sensor_msgs::Imu());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() 
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_    = true;
  start_timestamp_  = -1;
  init_iter_num     = 1;
  v_imu_.clear();
  IMUpose.clear();
  last_imu_.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

void ImuProcess::set_enu_init(bool enable)
{
  enu_init_en_ = enable;
}

void ImuProcess::set_gnss_ext(const V3D &t_gnss_in_imu, const M3D &R_imu_from_gnss)
{
  gnss_T_wrt_IMU_  = t_gnss_in_imu;
  gnss_R_to_IMU_   = R_imu_from_gnss;
}

void ImuProcess::gps_euler_cbk(const geometry_msgs::Vector3Stamped::ConstPtr &msg)
{
  // 存储原始NED欧拉角 (x=roll_NED, y=pitch_NED, z=heading_NED) [rad]
  gps_euler_ned_ = V3D(msg->vector.x, msg->vector.y, msg->vector.z);
  gps_euler_received_ = true;
}

void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance    初始化 重力(b系下) 、 gyro_bias 、 bgn 、 ban 
   ** 2. normalize the acceleration measurenments to unit gravity **/   //  归一化重力
  
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_)     //   提取第一帧IMU数据
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z; //记录最早加速度为加速度均值
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z; //记录最早角速度为角速度均值
    first_lidar_time = meas.lidar_beg_time; //记录lidar起始时间（绝对）
  }

  //读取所有imu信息
  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;//记录当前加速度
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;//记录当前角速度

    mean_acc      += (cur_acc - mean_acc) / N;//更新加速度均值
    mean_gyr      += (cur_gyr - mean_gyr) / N;//更新角速度均值

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N); //更新加速度协方差（向量）
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N); //更新角速度协方差（向量）

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

    N ++;
  }
  state_ikfom init_state = kf_state.get_x(); // 初始状态量

  if (enu_init_en_ && gps_euler_received_)
  {
    // ---- 1. 从 GPS euler (NED) 构建 GNSS 帧在 ENU 下的旋转 R_enu_gnss ----
    double roll_enu_g  = gps_euler_ned_.y();                   // pitch_NED → roll_ENU
    double pitch_enu_g = gps_euler_ned_.x();                   // roll_NED  → pitch_ENU
    double yaw_enu_g   = M_PI / 2.0 - gps_euler_ned_.z();     // heading_NED → yaw_ENU
    M3D R_enu_gnss = (Eigen::AngleAxisd(yaw_enu_g,   V3D::UnitZ())
                    * Eigen::AngleAxisd(pitch_enu_g, V3D::UnitY())
                    * Eigen::AngleAxisd(roll_enu_g,  V3D::UnitX())).toRotationMatrix();

    // ---- 2. 通过外参将 GNSS 方向转换到 IMU 方向 ----
    // R_imu_gnss = Lidar_R_wrt_IMU * Gnss_R_wrt_Lidar (GNSS帧→IMU帧)
    // R_enu_imu  = R_enu_gnss * R_gnss_imu = R_enu_gnss * R_imu_gnss^T
    M3D R_enu_imu = R_enu_gnss * gnss_R_to_IMU_.transpose();

    // ---- 3. 从 R_enu_imu 提取 IMU 真实 yaw ----
    double yaw_imu = atan2(R_enu_imu(1, 0), R_enu_imu(0, 0));

    // ---- 4. 用 mean_acc 计算 IMU 自身 roll/pitch（重力方向更准） ----
    // 静止时 mean_acc ≈ R_enu_imu^T * [0, 0, G]^T
    double roll_from_acc  = atan2(mean_acc.y(), mean_acc.z());
    double pitch_from_acc = -asin(mean_acc.x() / mean_acc.norm());

    // ---- 5. 最终 R_init: mean_acc 的 roll/pitch + 外参校正后的 yaw ----
    M3D R_init = (Eigen::AngleAxisd(yaw_imu,        V3D::UnitZ())
                * Eigen::AngleAxisd(pitch_from_acc, V3D::UnitY())
                * Eigen::AngleAxisd(roll_from_acc,  V3D::UnitX())).toRotationMatrix();

    init_state.rot  = SO3(R_init);
    init_state.grav = S2(0, 0, -G_m_s2);  // ENU 坐标系下重力始终为 [0, 0, -G]

    // ---- 6. 初始位置: GNSS天线在ENU原点，补偿杆臂到IMU ----
    // P_gnss_enu = P_imu_enu + R_enu_imu * T_gnss_in_imu = 0
    // => P_imu_enu = -R_enu_imu * T_gnss_in_imu
    init_state.pos = -R_init * gnss_T_wrt_IMU_;

    enu_init_roll  = roll_from_acc;
    enu_init_pitch = pitch_from_acc;
    enu_init_yaw   = yaw_imu;
    enu_init_pos   = init_state.pos;
    enu_inited     = true;

    ROS_INFO("ENU Init: roll=%.4f, pitch=%.4f, yaw=%.4f (deg), pos=(%.4f, %.4f, %.4f)",
             roll_from_acc * 180.0 / M_PI, pitch_from_acc * 180.0 / M_PI, yaw_imu * 180.0 / M_PI,
             init_state.pos.x(), init_state.pos.y(), init_state.pos.z());
    ROS_INFO("  (GPS euler NED: roll=%.2f, pitch=%.2f, heading=%.2f deg -> IMU yaw=%.2f deg)",
             gps_euler_ned_.x() * 180.0 / M_PI, gps_euler_ned_.y() * 180.0 / M_PI,
             gps_euler_ned_.z() * 180.0 / M_PI, yaw_imu * 180.0 / M_PI);
  }
  else
  {
    init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2); // 默认：根据加速度均值估计重力方向
  }

  init_state.bg  = mean_gyr; // 初始化陀螺仪bias为平均角速度
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;        //   t_lidar_imu  translate外参
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;       //   R_lidar_imu rotation 外参
  kf_state.change_x(init_state);                                  //    初始化状态量

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P(); // 获取协方差矩阵模版
  init_P.setIdentity(); //以下初始化协方差
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001; //外参r
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;//外参t
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;//bias g
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;//bias a
  init_P(21,21) = init_P(22,22) = 0.00001; // S2   重力记为S2流形
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back(); //记录最后一个imu
}

// IMU 后向传播，点云去畸变
void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu; // imu数据序列
  v_imu.push_front(last_imu_); //从头插入上一个imu数据
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec(); //imu序列起始时间
  const double &imu_end_time = v_imu.back()->header.stamp.toSec(); //imu序列结束时间
  const double &pcl_beg_time = meas.lidar_beg_time; //点云起始时间
  const double &pcl_end_time = meas.lidar_end_time; //点云结束时间
  
  /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list); //点按时间从小到大排序(从旧到新）
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  /*** Initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  //设定初始时刻相对状态(相对于imu积分初始状态的时间，上一加速度，上一角速度，速度，位置，旋转矩阵）
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

  /*** forward propagation at each imu point   前向传播***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;

  input_ikfom in;//输入状态量：记录加速度和角速度
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;

    //加速度和角速度均值，j与j+1的均值
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;

    //根据初始化得到的加速度均值，将加速度归算到重力尺度下
    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

    if(head->header.stamp.toSec() < last_lidar_end_time_) //前一IMU时间小于上一lidar结束时间
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_; //记录后一IMU时间与上一lidar结束时间之差
      // dt = tail->header.stamp.toSec() - pcl_beg_time;
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();//相邻imu时刻时间差
    }

    //记录输入量的均值和协方差，记录的是测量值
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    //向前传播
    kf_state.predict(dt, Q, in);//根据输入数据向前传播(两imu数据间隔时间差，协方差矩阵，输入数据） todo

    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg; // w = w - bg
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);// a(world) = r * ( a(body) - ba)
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];// a(world) + g(负值)
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;//m+1时刻与lidar起始时刻时间差
    //保存帧内imu数据 m+1 时刻的pose、前一时刻与当前时刻imu数据的均值（w系）、v、p、r、
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);// lidar终点与imu终点时间差
  kf_state.predict(dt, Q, in); // in 此时为最后一个imu数据，传播到lidar终点时刻与imu终点时刻较大者
  
  imu_state = kf_state.get_x();//lidar终点时刻与imu终点时刻较大者，imu位姿
  last_imu_ = meas.imu.back(); //记录最后一个imu数据为上一imu数据，用于下一次imu前向传播的开头
  last_lidar_end_time_ = pcl_end_time;//记录点云结束时间为上一帧结束时间

  /*** undistort each lidar point (backward propagation) ***/
  auto it_pcl = pcl_out.points.end() - 1;//点云从后向前遍历（时间大到小），此前点云已经按照时间从小到大排序
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--) //imu pose从后向前遍历
  {
    auto head = it_kp - 1;// 前一imu位姿，j-1
    auto tail = it_kp;// 后一imu位姿，j
    //j-1时刻imu的p、v、r
    R_imu<<MAT_FROM_ARRAY(head->rot);
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);
    pos_imu<<VEC_FROM_ARRAY(head->pos);

    //j时刻的加速度和角速度
    acc_imu<<VEC_FROM_ARRAY(tail->acc);
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);

    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --) //head->offset_time为j-1时刻，imu相对于传播起始时刻的时间
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;//相对于前一imu时刻的时间差

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));//R(global <-- i)
      // global <-- head <-- i
      
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);//lidar系下索引i点的坐标
      //imu_state:lidar终点时刻与imu终点时刻较大者，imu位姿
      //Ti - Te
      //T_ei 从imu终点位置指向索引i时刻imu位置的平移向量，w系, 即T_i - T_e
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);//T(i <-- end)
      // 从imu终点位置指向索引i时刻imu位置的平移向量

      //imu_state.offset_R_L_I:终点时刻imu和lidar的外参

      /***  变换： 终点时刻lidar系 <-- 终点时刻imu（body系） <-- 畸变纠正到global系 <-- 终点时刻imu（body系）<-- 终点时刻lidar系
       * 在imu（body）系下进行畸变纠正到w系，可能不太准确
       * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) 将i点，按照imu终点时刻的外参，从lidar系转换到终点时刻的imu系下
       * (R_i * ( P ) + T_i)  畸变纠正，将i点纠正到global系下
       * (imu_state.rot.conjugate() * ( P - T_e)  将i点旋转到终点时刻的，imu系下
       * T_i - T_e = T_ei
       * imu_state.offset_R_L_I.conjugate() * （P - imu_state.offset_T_L_I) 变换到终点时刻的lidar系下
       ***/
      V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * \
      (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);// not accurate!
      
      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}

void ImuProcess::Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();

  if(meas.imu.empty()) {return;};
  ROS_ASSERT(meas.lidar != nullptr);

  //imu 初始化，初始化完成后才做畸变纠正
  if (imu_need_init_)
  {
      /** 1. initializing the gravity, gyro bias, acc and gyro covariance
        * 2. normalize the acceleration measurenments to unit gravity **/
    /// The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back(); // 记录最后一个imu数据

    state_ikfom imu_state = kf_state.get_x(); // 获取状态量
    if (init_iter_num > MAX_INI_COUNT) //初始化imu数据量大于阈值
    {
      if (enu_init_en_ && !gps_euler_received_)
      {
        ROS_WARN_THROTTLE(1.0, "ENU init enabled, waiting for GPS euler data on /gps/euler ...");
        return;
      }

      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
      // ROS_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
      //          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return;
  }

  //畸变纠正
  UndistortPcl(meas, kf_state, *cur_pcl_un_);

  t2 = omp_get_wtime();
  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}
