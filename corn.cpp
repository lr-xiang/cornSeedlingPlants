#include <iostream>
#include <fstream>
#include <string>
#include <math.h> 

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
//#include <opencv2/aruco.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_types_conversion.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <boost/thread/thread.hpp>
#include <boost/filesystem.hpp> 

#include <pcl/registration/icp.h>
#include <pcl/registration/joint_icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/correspondence_rejection_median_distance.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
//#include <pcl/visualization/pcl_plotter.h>
#include <pcl/console/time.h>   // TicToc

#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/normal_3d.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/radius_outlier_removal.h>

#include <pcl/segmentation/extract_clusters.h>  //euclidean cluster
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/lccp_segmentation.h>
#include <pcl/segmentation/cpc_segmentation.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/min_cut_segmentation.h>

#include <pcl/surface/mls.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/gp3.h>

#include <pcl/features/organized_edge_detection.h>
#include <pcl/features/normal_3d.h>

#include <Eigen/Dense>  //using vector
#include <vector>

#include <pcl/ModelCoefficients.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <vtkPolyLine.h>

#include <pcl/common/pca.h>

#include <utility>

#include <pcl/filters/passthrough.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/bc_clustering.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/copy.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <boost/config.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/two_bit_color_map.hpp>
#include <boost/graph/named_function_params.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/connected_components.hpp>

#include <ctime>
#include <chrono>

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace boost;

#define PI 3.14159265
#define visualize
//#define mergeSeg  //step1
//#define leafSeg //step 3
#define featureExtraction //step2
//#define paperVisual //the last step
//#define leafsegvisual //the last step

int to_merge_th = 2000;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;

typedef pcl::SupervoxelClustering<PointT>::VoxelAdjacencyList Graph;
typedef boost::graph_traits<Graph>::vertex_iterator supervoxel_iterator;
typedef boost::graph_traits<Graph>::edge_iterator supervoxel_edge_iter;
typedef pcl::SupervoxelClustering<PointT>::VoxelID Voxel;
typedef pcl::SupervoxelClustering<PointT>::EdgeID Edge;
typedef pcl::SupervoxelClustering<PointT>::VoxelAdjacencyList::adjacency_iterator supervoxel_adjacency_iterator;

struct SVEdgeProperty
{
	float weight;
};

struct SVVertexProperty
{
	uint32_t supervoxel_label;
	pcl::Supervoxel<PointT>::Ptr supervoxel;
	uint32_t index;
	float max_width;
	int vertex;
	bool near_junction = false;
	bool start_point = false;
	std::vector<uint32_t> children;
	std::vector<uint32_t> parents;
	std::vector<int> cluster_indices;
};

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, SVVertexProperty, SVEdgeProperty> sv_graph_t;

typedef boost::graph_traits<sv_graph_t>::vertex_descriptor sv_vertex_t;

typedef boost::graph_traits<sv_graph_t>::edge_descriptor sv_edge_t;

struct Leaf {

    float node;

    float length = 0.f;

    float angle = -1;

    float area = -1;

    float angle_2d = -1;

    float node_height=-1;

    bool right_side = false;

    float curve = -1;

    float curvature = -1;

    bool is_tiller = false;

	float angle_vertical = -1;

	int leaf_num = -1;

    std::vector<sv_vertex_t> node_vec;

};

cv::Mat kinect_rgb_camera_matrix_cv_, kinect_rgb_dist_coeffs_cv_;
cv::Mat image;

PointT pot_center;

using namespace std;

using namespace pcl;
using namespace pcl::io;

string date = "06182018";

//string current_exp_folder = "/media/lietang/easystore/RoAD/cornValidation/13days_validation/";
string current_exp_folder = "/media/lietang/easystore/RoAD/cornValidation/corn2nd/"+date+"/";
//string current_exp_folder = "/media/lietang/easystore/RoAD/cornValidation/maize04132019/";

string merged_path;

string holistic_measurement_path = current_exp_folder + "traits_holistic_"+date+".csv";

string leaf_measurement_path = current_exp_folder + "individual_leaf_traits"+date +".csv";

bool lcompare(const Leaf &l1, const Leaf &l2){ 

	if(l1.node_height == l2.node_height)
		return l1.angle>l2.angle;
	else
		return l1.node_height < l2.node_height;

}

float pre_x = 0, pre_y = 0, pre_z = 0, Dist = 0;
void pp_callback(const pcl::visualization::PointPickingEvent& event);

int loadData(std::string& id, std::vector<PointCloudT::Ptr, Eigen::aligned_allocator <PointCloudT::Ptr>>& data);

inline void filter(PointCloudT::Ptr cloud_in,PointCloudT::Ptr cloud_out);

inline void cloudRegistration(std::vector<PointCloudT::Ptr, Eigen::aligned_allocator <PointCloudT::Ptr >> &data, PointCloudT::Ptr cloud_out);


inline void downsample(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_out, float leaf_size);

int ICPregistrationMatrix(std::vector<PointCloudT::Ptr, Eigen::aligned_allocator<PointCloudT::Ptr> > &data,
PointCloudT::Ptr cloud_out);

inline void getPlant(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_out);

int skeleton(PointCloudT::Ptr cloud_in, float ratio);

double soilDetection(PointCloudT::Ptr cloud);

string current_pot_id;

double getMeshArea(PointCloudT::Ptr cloud, bool projected);

void potDetection(PointCloudT::Ptr cloud);

ofstream outFile;

float soil_z = -1;

inline void alignCloud(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_out);

double projectedArea(PointCloudT::Ptr cloud, int cor, bool convex);

double convexhull(PointCloudT::Ptr cloud, bool area);

inline void projectCloudy(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_out);

int
main(int argc,
	char* argv[])
{

	cout<<current_exp_folder<<endl;

	merged_path = current_exp_folder + "merged";

	boost::filesystem::path dir_3d_images(current_exp_folder + "leafSegResults");

	if (boost::filesystem::create_directory(dir_3d_images))
	{
		std::cerr << "Directory Created for 3d images\n";
	}


	boost::filesystem::path merged_3d_images(current_exp_folder + "merged");

	if (boost::filesystem::create_directory(merged_3d_images))
	{
		std::cerr << "Directory Created for 3d images\n";
	}


#ifndef visualize
	outFile.open(holistic_measurement_path, ios::app);

	outFile << "plant_id,sample_date,plant_height,plant_length,plant_width,";

	outFile << "total_area,convex_hull_area,convex_hull_volumn,LAI,aspect_ratio,";

	outFile << "projected_area_x,projected_area_y,projected_area_z,";

	outFile << "projected_convex_area_x,projected_convex_area_y,projected_convex_area_z,";

	outFile << "solidity,solidity_x,solidity_y,solidity_z\n";

	outFile.close();



	outFile.open(leaf_measurement_path, ios::app);

	outFile << "plant_id,sample_date,leaf_num,leaf_area,leaf_length,leaf_curvature,node_height,leaf_angle,leaf_angle_2d,leaf_angle_vertical\n";
	
	outFile.close();

#endif


	std::ifstream label_input(current_exp_folder + "label.txt");  


	std::vector<std::string> pot_labels;
	if (label_input.is_open())
	{
		int line_num = 0;
		for (std::string line; std::getline(label_input, line); line_num++){
			boost::replace_all(line, "\r","");
			boost::replace_all(line, "\n","");
			pot_labels.push_back(line);
			cout<<"line "<<line_num<< ": "<<line<<endl;
		}		
	}

	cout<<"label size: "<<pot_labels.size()<<endl;

	for (int i = 0;i < pot_labels.size();i++){ //pot_labels.size()

		current_pot_id = pot_labels[i];

#ifdef visualize
		cout<<"input id index\n";

		current_pot_id = "";

		//cin>>current_pot_id;	
		cin>>i;

		current_pot_id = pot_labels[i];
#endif	
		cout <<"current_pot_id: "<<current_pot_id<<endl;


#ifdef mergeSeg
//load data
		std::vector<PointCloudT::Ptr, Eigen::aligned_allocator<PointCloudT::Ptr> > data;

		if (loadData(current_pot_id, data) < 0)
				continue;

		PointCloudT::Ptr cloud_merged(new PointCloudT);

		ICPregistrationMatrix(data, cloud_merged);

		//cloudRegistration(data, cloud_merged);

		PointCloudT::Ptr cloud_plant(new PointCloudT);

		getPlant(cloud_merged, cloud_plant);

//registration

#endif

#ifdef featureExtraction

	string path = current_exp_folder + "merged/" + current_pot_id +".pcd";

	PointCloudT::Ptr cloud_plant(new PointCloudT);

	if (pcl::io::loadPCDFile<PointT>(path, *cloud_plant) == -1) //* load the file
	{

		PCL_ERROR("Couldn't read file test_pcd.pcd \n");
		cout << path<<endl;
		return -1;
	}

	float ratio;

	if (i>32) 
		ratio = 0.5f;
	else
		ratio = 1.f;
	skeleton(cloud_plant, ratio);
#endif
	}


#ifdef paperVisual

	std::vector<string> controls = {"BRI1-RNAi_Control_plant_23","BRI1-RNAi_Control_plant_34","BRI1-RNAi_Control_plant_27","BRI1-RNAi_Control_plant_32","BRI1-RNAi_Control_plant_12","BRI1-RNAi_Control_plant_18"};


	std::vector<string> PCZs = {"BRI1-RNAi_PCZ-500_plant_2","BRI1-RNAi_PCZ-500_plant_3","BRI1-RNAi_PCZ-500_plant_4","BRI1-RNAi_PCZ-500_plant_10","BRI1-RNAi_PCZ-500_plant_30","BRI1-RNAi_PCZ-500_plant_35"};


	PointCloudT::Ptr final_cloud(new PointCloudT);

	int count_i = 0;

	PointT pt;

	float x_shift = 0.1f;

	uint32_t rgb;

    rgb = 0 << 16 | 0 << 8 | 255;

	for (auto &i:controls){

		PointCloudT::Ptr cloud(new PointCloudT);

		//string path = current_exp_folder + "merged/" + i+".pcd";

		string path = current_exp_folder + "leafSegResults/" + i+".pcd";

		if (pcl::io::loadPCDFile<PointT>(path, *cloud) == -1) //* load the file

		{

			PCL_ERROR("Couldn't read file test_pcd.pcd \n");

			cout << path<<endl;

			continue;

		}

		//alignCloud(cloud,cloud);

		for (auto &p:cloud->points){
			pt = p;

#ifdef leafsegvisual
			pt.x = p.x + x_shift*count_i;

			if (count_i==1)
				pt.y = p.y -0.05f;

			if (count_i==1)
				pt.x = p.x + 0.08f*count_i;
#else
			pt.x = p.x + x_shift*count_i;

			if (count_i==4)
				pt.x = p.x + 0.09f*count_i;

			pt.rgb = *reinterpret_cast<float*>(&rgb);// comment for merge leaf seg
#endif

			final_cloud->points.push_back(pt);
		}
		count_i++;
	}  //i:controls


	x_shift = 0.1f;
	count_i = 0;
    rgb = 255 << 16 | 0 << 8 | 0;

	for (auto &i:PCZs){

		PointCloudT::Ptr cloud(new PointCloudT);

		//string path = current_exp_folder + "merged/" + i+".pcd";

		string path = current_exp_folder + "leafSegResults/" + i+".pcd";

		if (pcl::io::loadPCDFile<PointT>(path, *cloud) == -1) //* load the file
		{

			PCL_ERROR("Couldn't read file test_pcd.pcd \n");
			cout << path<<endl;
			continue;
		}

		//alignCloud(cloud,cloud);

		float y_shift = 0.2f;

		for (auto &p:cloud->points){
			pt = p;
#ifdef leafsegvisual
			pt.y = p.y - y_shift;
			pt.x = p.x + x_shift*count_i;
			if (count_i==3){
				pt.x = p.x + 0.08f*count_i;
				pt.y = pt.y - 0.03f;}

			if (count_i==4 ){
				pt.x = p.x + 0.05f*count_i;
				pt.y = pt.y - 0.06f;}

			if (count_i==5 ){
				pt.x = p.x + 0.097f*count_i;}
#else
			pt.y = p.y - y_shift;
			pt.x = p.x + x_shift*count_i;

			if (count_i==3)
				pt.x = p.x + 0.095f*count_i;
			pt.rgb = *reinterpret_cast<float*>(&rgb); // comment for merge leaf seg
#endif
			final_cloud->points.push_back(pt);
		}
		count_i++;
	}  //i:controls

	final_cloud->width = 1;
	final_cloud->height = final_cloud->points.size();

	//pcl::io::savePCDFile(current_exp_folder + "merged/" + "controlpczs_cloud_colored.pcd", *final_cloud);

	pcl::io::savePCDFile(current_exp_folder + "leafSegResults/" + "controlpczs_cloud_colored.pcd", *final_cloud);

#endif

	cout << "All finished!!\n";
	getchar();
	return 0;
}



void pp_callback(const pcl::visualization::PointPickingEvent& event)
{
	float x, y, z;
	event.getPoint(x, y, z);
	Dist = sqrt(pow(x - pre_x, 2) + pow(y - pre_y, 2) + pow(z - pre_z, 2));
	//	Eigen::Vector3f dir(pre_x-x, pre_y-y, pre_z-z);
	//	dir.normalize();	
	pre_x = x;
	pre_y = y;
	pre_z = z;
	std::cout << "x:" << x << " y:" << y << " z:" << z << " distance:" << Dist/*<<" nx:"<<dir(0)<<" ny:"<<dir(1)<<" nz:"<<dir(2)*/ << std::endl;

}


int loadData(std::string& id, std::vector<PointCloudT::Ptr, Eigen::aligned_allocator <PointCloudT::Ptr>>& data) {

	cout << "loading " << id << endl;

	//PointCloudT::Ptr cloud(new PointCloudT);

	string path = current_exp_folder +id;

	std::vector<string> id_vec = {"_scan_3.pcd","_scan_0.pcd","_scan_4.pcd","_scan_2.pcd"}; //  "_scan_2.pcd"

	int count = 0;

	for(auto &i:id_vec){

		std::string temp_path = path + i;

		PointCloudT::Ptr cloud(new PointCloudT);

		if (pcl::io::loadPCDFile<PointT>(temp_path, *cloud) == -1) //* load the file
		{

			PCL_ERROR("Couldn't read file test_pcd.pcd \n");
			cout << temp_path<<endl;
			return -1;
		}

		filter(cloud, cloud);

		if(count ==3)
			getPlant(cloud,cloud);

		data.push_back(cloud);
		cout << "count: "<<count<<" data : " <<i<< " size: "<< cloud->size() << endl;

		count++;

	}

	
/*
	for (int i = 4;i >0;i--) {

		temp_path = path + "_scan_" + to_string(i) + ".pcd";

		PointCloudT::Ptr cloud_source(new PointCloudT);

		if (pcl::io::loadPCDFile<PointT>(temp_path, *cloud_source) == -1) //* load the file
		{
			PCL_ERROR("Couldn't read file \n");
			std::cout << temp_path << std::endl;
			return -1;
		}
		
		data.push_back(cloud_source);

		cout << "data " << 5 - i << " : " << data.at(5 - i)->size() << endl;

	}
*/
	return 1;
}


inline void filter(PointCloudT::Ptr cloud_in,PointCloudT::Ptr cloud_out){
	
	PointCloudT::Ptr cloud_backup(new PointCloudT);
	PointCloudT::Ptr tmp(new PointCloudT);
        *cloud_backup += *cloud_in;
        cloud_out->clear();

	int r, g, b;

	for(auto &i:cloud_backup->points){
		r = (int)i.r;
		g = (int)i.g;
		b = (int)i.b;
		
		if(i.z<0.022f )  //i.z<0.025f 
			continue;

		if(i.y>0.51f)
			continue;

		if(i.x>0.1f && i.z < 0.04f)
			continue;

		tmp->points.push_back(i);

	}

	float radius =0.002f;
	int num =50;
	//cin >>radius>>num;

	pcl::RadiusOutlierRemoval<PointT> outrem;
	outrem.setInputCloud(tmp);
	outrem.setRadiusSearch(radius);
	outrem.setMinNeighborsInRadius(num);
	// apply filter
	outrem.filter(*cloud_out);

	cout<<"before filtering: "<<tmp->size()<<" after: "<<cloud_out->size()<<endl;

#ifdef visualize
	pcl::visualization::PCLVisualizer viewer("ICP demo");

	int v1(0);
	int v2(1);
	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

	viewer.registerPointPickingCallback(&pp_callback);

	viewer.addPointCloud(cloud_backup, to_string(cv::getTickCount()), v1);
	viewer.addPointCloud(cloud_out, to_string(cv::getTickCount()), v2);

	viewer.spin();
#endif

}


inline void cloudRegistration(std::vector<PointCloudT::Ptr, Eigen::aligned_allocator <PointCloudT::Ptr >> &data, PointCloudT::Ptr cloud_out) {
	
			PointCloudT::Ptr colored_cloud(new PointCloudT);
			uint32_t rgb;


			int a, b, c;
	
			string cloud_id;

            for (int i = 0; i < data.size(); i++) {

				PointCloudT::Ptr cloud(new PointCloudT);

                *cloud = *data.at(i);
		
				filter(cloud, cloud);

				*cloud_out+=*cloud;

                cout << "cloud " << i << " : " << cloud_out->size() << "\n" << endl;
#if 1			
				srand(time(NULL));

				a = (double)(rand() % 256);
				b = (double)(rand() % 256);
				c = (double)(rand() % 256);

        		rgb = a << 16 | b << 8 | c;



		    	for (auto &j : cloud->points){
					j.rgb = *reinterpret_cast<float*> (&rgb); 
					colored_cloud->points.push_back(j);}	

#endif             
            }

#if 1
 
            pcl::visualization::PCLVisualizer viewer("ICP demo");


            int v1(0);

            int v2(1);

            viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);

            viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);


			viewer.registerPointPickingCallback(&pp_callback);

 			viewer.addPointCloud(colored_cloud, to_string(cv::getTickCount()), v1);
		    viewer.addPointCloud(cloud_out, to_string(cv::getTickCount()), v2);

            viewer.spin();

		colored_cloud->width = 1;
		colored_cloud->height = colored_cloud->points.size();

		cloud_out->width = 1;
		cloud_out->height = cloud_out->points.size();

		pcl::io::savePCDFile(current_exp_folder + "merged/" + current_pot_id +"_colored_cloud.pcd", *colored_cloud);
		pcl::io::savePCDFile(current_exp_folder + "merged/" + current_pot_id +"_cloud_out.pcd", *cloud_out);

#endif		
            cout << "registration done! \n" << endl;
        }

int ICPregistrationMatrix(std::vector<PointCloudT::Ptr, Eigen::aligned_allocator<PointCloudT::Ptr> > &data,
PointCloudT::Ptr cloud_out) {

	cout << "ready for registration\n" << endl;

	PointCloudT::Ptr cloud_target(new PointCloudT);  // 

	PointCloudT::Ptr cloud_icp(new PointCloudT);  // ICP output point cloud

	PointCloudT::Ptr cloud(new PointCloudT);

	PointCloudT::Ptr cloud_plant(new PointCloudT);

	Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();

	*cloud_target += *data.at(0);

	cout << "cloud_target original cloud:  " << cloud_target->size() << endl;

	//removePotOutlier(cloud_target, cloud_target);

	//filter(cloud_target, cloud_target);

	pcl::console::TicToc time;

	pcl::IterativeClosestPoint<PointT, PointT> icp;

	icp.setMaxCorrespondenceDistance(0.01); //0.01

	icp.setTransformationEpsilon(1e-15);

	icp.setEuclideanFitnessEpsilon(1e-15);

	icp.setMaximumIterations(100);//100

	cloud->clear();

	*cloud += *cloud_target;


		downsample(cloud_target, cloud, 0.0005);

		for (int i = 1;i <data.size();i++) {

			PointCloudT::Ptr cloud_source(new PointCloudT);
			PointCloudT::Ptr cloud_icp_plant(new PointCloudT);
			PointCloudT::Ptr cloud_source_downsample(new PointCloudT);

			cloud_icp->clear();

			*cloud_source += *data.at(i);

			cout << "target cloud: " << cloud->size() << " points." << endl;

			cout << "pcd file " << i << " (source original):" << cloud_source->size() << " points." << endl;

			//filter(cloud_source, cloud_source);

			cout << "source cloud: " << cloud_source->size() << " points." << endl;

			time.tic();

			downsample(cloud_source, cloud_source_downsample, 0.0005);

			icp.setInputTarget(cloud);

			icp.setInputSource(cloud_source_downsample);

			icp.align(*cloud_icp);

			cout << "round " << i << " finished registration in " << time.toc() << " ms.\n" << endl;;

			//transformation_matrix = icp.getFinalTransformation().cast<double>();

			//print4x4Matrix(transformation_matrix);

			//for visualize
#ifdef visualize
			pcl::visualization::PCLVisualizer viewer("ICP demo");

			int v1(0);
			int v2(1);
			viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
			viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

			viewer.registerPointPickingCallback(&pp_callback);

			viewer.addPointCloud(cloud, to_string(cv::getTickCount()), v1);

			pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_target_color_h(cloud_icp, 180, 20, 20);
			viewer.addPointCloud(cloud_icp, cloud_target_color_h, to_string(cv::getTickCount()), v1);

			pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_source_color_h(cloud_source_downsample, 180, 20, 20);
			viewer.addPointCloud(cloud_source_downsample, cloud_source_color_h, to_string(cv::getTickCount()), v2);

			viewer.addPointCloud(cloud, to_string(cv::getTickCount()), v2);

			viewer.spin();
#endif

			*cloud += *cloud_icp;
		}
	
	downsample(cloud, cloud_out, 0.0005);

	cout << "registration done!\n" << endl;

	cout << "cloud plant final: " << cloud_out->size() << "\n" << endl;

	//getchar();

	return 1;
}


inline void downsample(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_out, float leaf_size) {

	cout << "before downsampling: " << cloud->size() << endl;

	PointCloudT::Ptr cloud_backup(new PointCloudT);
	*cloud_backup += *cloud;

	cloud_out->clear();

	//downsample


	pcl::VoxelGrid<PointT> sor;
	sor.setInputCloud(cloud_backup);
	sor.setLeafSize(leaf_size, leaf_size, leaf_size);
	sor.filter(*cloud_out);

#ifdef visualize
	cout << "leaf_size: " << leaf_size << endl;
	cout << "cloud after 0.0004 downsampling: " << cloud_out->size() << endl;
#endif

}

inline void getPlant(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_out){

	PointCloudT::Ptr cloud_backup(new PointCloudT);

	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    pcl::PassThrough<PointT>pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimitsNegative(false);

	pass.setFilterLimits(0.036f, 10.f);

    pass.filter(*cloud_backup);

	std::vector<pcl::PointIndices> cluster_indices;

	pcl::EuclideanClusterExtraction<PointT> ec;

	pcl::search::KdTree<PointT>::Ptr ectree(new pcl::search::KdTree<PointT>);

	ectree->setInputCloud(cloud_backup);

	ec.setClusterTolerance(0.005); // 0.001

	ec.setMinClusterSize(0);

	ec.setMaxClusterSize(250000);

	ec.setSearchMethod(ectree);

	ec.setInputCloud(cloud_backup);

	ec.extract(cluster_indices);

	PointCloudT::Ptr tmp(new PointCloudT);

	pcl::PCA<PointT> pca;

	Eigen::Vector3f eigen_values, major_vector;

	Eigen::Vector3f line_z(0,0,1);

	float linearity, angle, abs_cos;

	cloud_out->clear();

	for (int i = 0;i < cluster_indices.size();i++) {

		if (cluster_indices[i].indices.size() > 1000) {

			PointCloudT::Ptr leaf(new PointCloudT);

			copyPointCloud(*cloud_backup, cluster_indices[i].indices, *tmp);

			pca.setInputCloud(tmp);

			eigen_values = pca.getEigenValues();

			major_vector = pca.getEigenVectors().col(0);

            abs_cos = std::abs(major_vector.dot(line_z));

            angle = acos(abs_cos) / PI *180.f;

			linearity = (eigen_values(0) - eigen_values(1))/eigen_values(0);

			cout << "i: " << i << ", " << cluster_indices[i].indices.size() <<" linearity: "<<linearity<<" angle: "<<angle<<endl;

			if(linearity>0.99 && angle > 30) continue;

			*cloud_out+=*tmp;

			
		}

		cout << "i: " << i << ", " << cluster_indices[i].indices.size() << endl;
	}

	cout<<"cloud_backup: "<<cloud_backup->size()<<" cloud_out: "<<cloud_out->size()<<endl;

#ifdef visualize
			pcl::visualization::PCLVisualizer viewer("ICP demo");

			int v1(0);
			int v2(1);
			viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
			viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

			viewer.registerPointPickingCallback(&pp_callback);

			viewer.addPointCloud(cloud_backup, to_string(cv::getTickCount()), v1);


			viewer.addPointCloud(cloud_out, to_string(cv::getTickCount()), v2);

			viewer.spin();
#endif

//save 


	std::stringstream file_name;
	file_name << merged_path << "/" << current_pot_id << ".pcd";

	pcl::io::savePCDFile(file_name.str(), *cloud_out);
}


int skeleton(PointCloudT::Ptr cloud_in, float ratio){

//if big plants, rotate point cloud to align the widest dimension along with x axis

	//if(ratio>0.5f)
	alignCloud(cloud_in,cloud_in);

	soil_z = 0.036f;

	cout<<"soil_z: "<<soil_z<< " ratio: " <<ratio<<endl;

	PointCloudT::Ptr cloud_backup(new PointCloudT);

    *cloud_backup += *cloud_in;  //will rotate cloud?

	PointCloudT::Ptr cloud(new PointCloudT);

    *cloud += *cloud_in;  //will rotate cloud?

    float stem_radius = 0.01f;

	float slice_thickness = stem_radius/2.f;

    //skeleton
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    pcl::PassThrough<PointT>pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimitsNegative(false);

    pcl::PassThrough<PointT> pass_in_cluster;
    pass_in_cluster.setFilterFieldName("z");
    pass_in_cluster.setInputCloud(cloud);

    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(slice_thickness*ratio); // 1cm
    ec.setMinClusterSize(5);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);

    pcl::KdTreeFLANN<PointT> kdtree;
    pcl::KdTreeFLANN<PointT> kdtree_cluster;
    kdtree.setInputCloud(cloud);

    sv_graph_t skeleton_graph;

    uint32_t rgb;

    int cnt = 0;

    PointCloudT::Ptr cloud_temp(new PointCloudT);

    //make leaf skeleton

	Eigen::Vector4f min_pt, max_pt;

    getMinMax3D(*cloud, min_pt, max_pt);

    bool first_slice = true;

    pcl::PointIndices::Ptr pre_indices(new pcl::PointIndices);

	for (float x = min_pt(2); x <= max_pt(2); x += slice_thickness) {
    //for (float x = soil_z + slice_thickness; x <= max_pt(2); x += slice_thickness) {

        pass.setFilterLimits(x, x + slice_thickness);

        pcl::PointIndices::Ptr slice_indices(new pcl::PointIndices);

        pass.filter(slice_indices->indices);

        std::vector<int> tmp_indices;

        tmp_indices = slice_indices->indices;

        // append previous slice
        slice_indices->indices.insert(slice_indices->indices.end(), pre_indices->indices.begin(), pre_indices->indices.end());

        pre_indices->indices = tmp_indices;

        ec.setIndices(slice_indices);

        std::vector<pcl::PointIndices> cluster_indices;

        ec.extract(cluster_indices);

        if (first_slice) {

            first_slice = false;

            for (int j = 0; j < cluster_indices.size(); j++) {

                Eigen::Vector4f centroid;

                pcl::compute3DCentroid(*cloud, cluster_indices[j], centroid);

                PointT p; p.getVector3fMap() = centroid.head(3);

                std::vector<int> k_indices; std::vector<float> k_sqr_distances;

                if (kdtree.nearestKSearch(p, 1, k_indices, k_sqr_distances) == 1) {

                    SVVertexProperty vp;

                    vp.index = k_indices[0];

                    sv_vertex_t v = boost::add_vertex(vp, skeleton_graph);
                }
            }

            continue;
        }

        pass_in_cluster.setFilterLimits(x, x + slice_thickness);

        for (auto & two_layer_cluster : cluster_indices) {

            pcl::PointIndices::Ptr lower_slice_inliers_in_cluster(new pcl::PointIndices);

            pcl::PointIndices::Ptr upper_slice_inliers_in_cluster(new pcl::PointIndices);

            boost::shared_ptr<std::vector<int>> two_layer_cluster_ptr(new std::vector<int>(two_layer_cluster.indices));

            pass_in_cluster.setIndices(two_layer_cluster_ptr);

            pass_in_cluster.setFilterLimitsNegative(false);

            pass_in_cluster.filter(lower_slice_inliers_in_cluster->indices);

            pass_in_cluster.setFilterLimitsNegative(true);

            pass_in_cluster.filter(upper_slice_inliers_in_cluster->indices);

            std::vector<pcl::PointIndices> lower_clusters_indices_in_cluster;

            std::vector<pcl::PointIndices> upper_clusters_indices_in_cluster;

            ec.setIndices(lower_slice_inliers_in_cluster);

            ec.extract(lower_clusters_indices_in_cluster);

            ec.setIndices(upper_slice_inliers_in_cluster);

            ec.extract(upper_clusters_indices_in_cluster);

            for (auto & cl : lower_clusters_indices_in_cluster)
            {
                Eigen::Vector4f lower_centroid;

                pcl::compute3DCentroid(*cloud, cl, lower_centroid);

                PointT p; p.getVector3fMap() = lower_centroid.head(3);

                std::vector<int> k_indices; std::vector<float> k_sqr_distances;

                if (kdtree.nearestKSearch(p, 1, k_indices, k_sqr_distances) == 1) {

                    SVVertexProperty vp;

                    vp.index = k_indices[0];

                    vp.cluster_indices = cl.indices;

                    sv_vertex_t new_vertex = boost::add_vertex(vp, skeleton_graph);

                    for (auto & cu : upper_clusters_indices_in_cluster) {

                        Eigen::Vector4f upper_centroid;

                        pcl::compute3DCentroid(*cloud, cu, upper_centroid);

                        PointT p_u; p_u.getVector3fMap() = upper_centroid.head(3);

                        k_indices.clear(); k_sqr_distances.clear();

                        if (kdtree.nearestKSearch(p_u, 1, k_indices, k_sqr_distances) == 1)
                            for (int old_vertex = boost::num_vertices(skeleton_graph) - 1; old_vertex >= 0; old_vertex--)
                                if (skeleton_graph[old_vertex].index == k_indices[0]
                                    && (new_vertex != old_vertex)) {

                                    sv_edge_t edge; bool edge_added;

                                    boost::tie(edge, edge_added) = boost::add_edge(old_vertex, new_vertex, skeleton_graph);

                                    float edge_weight = (lower_centroid - upper_centroid).norm();;

                                    skeleton_graph[edge].weight = edge_weight;

                                    break;
                                }
                    }
                }
            }
        }

        uint32_t r = cnt % 2 == 0 ? 255 : 0;

        rgb = r << 16 | 0 << 8 | 255;

#ifdef visualize
        for (auto & m : pre_indices->indices)
            cloud->points[m].rgb = *reinterpret_cast<float*>(&rgb);
#endif
        cnt++;
    }

    cout << "\n\ninitial skeleton done!\n" << endl;
	

	// remove loops in graph with MST
    std::vector<sv_edge_t> spanning_tree;

    boost::kruskal_minimum_spanning_tree(skeleton_graph, std::back_inserter(spanning_tree), boost::weight_map(boost::get(&SVEdgeProperty::weight, skeleton_graph)));

    sv_graph_t mst(boost::num_vertices(skeleton_graph));

    BGL_FORALL_VERTICES(v, skeleton_graph, sv_graph_t)
        mst[v] = skeleton_graph[v];

    for (auto & e : spanning_tree)
    {
        sv_vertex_t s = boost::source(e, skeleton_graph);

        sv_vertex_t t = boost::target(e, skeleton_graph);

        sv_edge_t new_e;

        bool edge_added;

        boost::tie(new_e, edge_added) = boost::add_edge(s, t, mst);

        mst[new_e].weight = skeleton_graph[e].weight;

    }

    cout << "mst done\n";


	int min_branch_size = 3;
	//prune short branches
    std::set<sv_vertex_t> vertices_to_remove;

    BGL_FORALL_VERTICES(v, mst, sv_graph_t) {

            // process leaf node
            if (boost::out_degree(v, mst) != 1) continue;

			//close to bottom

			if (cloud->points[mst[v].index].z - min_pt(2) < 2.f*slice_thickness) continue;

            sv_vertex_t cur_v = *(adjacent_vertices(v, mst).first);

            std::vector<sv_vertex_t> visited_vertices;

            visited_vertices.push_back(v);

            while (true) {

                const int num_neighbors = boost::out_degree(cur_v, mst);

                if (num_neighbors == 1) {    // leaf node

                    visited_vertices.push_back(cur_v);
                    break;
                }
                else if (num_neighbors == 2) { // can continue

                    BGL_FORALL_ADJ(cur_v, adj, mst, sv_graph_t) {

                        if (adj != visited_vertices.back()) {

                            visited_vertices.push_back(cur_v);
                            cur_v = adj;
                            break;
                        }
                    }

                    continue;
                }
                else     //intersection 
                    break;
            }

            if (visited_vertices.size() < min_branch_size)
                for (auto & visited_vertex : visited_vertices)
                    vertices_to_remove.insert(visited_vertex);
    }
	

	//clear vertices

	cout<<"vertices size : "<<vertices_to_remove.size()<<endl;

 	Eigen::Vector3f p0, p1;


	float temp_dis;

	sv_vertex_t nearest_v;

    for (auto iter = vertices_to_remove.begin(); iter != vertices_to_remove.end(); ++iter){

		getMinMax3D(*cloud, mst[*iter].cluster_indices, min_pt, max_pt);

		if(max_pt[0]-min_pt[0] > stem_radius){ //0.01f
			
			p0 = cloud->points[mst[*iter].index].getVector3fMap();

			//find the nearest v

			float dis =100.f;

			BGL_FORALL_VERTICES(v, mst, sv_graph_t){
			
				if (boost::out_degree(v, mst) < 1) continue;

				p1 = cloud->points[mst[v].index].getVector3fMap();

				//cout<<"(p0-p1).norm(): "<<(p0-p1).norm()<<endl;

				temp_dis = (p0-p1).norm();

				if(temp_dis!=0 && temp_dis < dis ){

					nearest_v = v;
					dis = temp_dis;
				
				}

			}

			//cout<<"dis: "<<dis<<endl; 
			rgb = 255 << 16 | 255 << 8 | 0;

			mst[nearest_v].cluster_indices.insert(mst[nearest_v].cluster_indices.end(), mst[*iter].cluster_indices.begin(), mst[*iter].cluster_indices.end());
			
			//for (auto & i : mst[nearest_v].cluster_indices)
                   //cloud->points[i].rgb = *reinterpret_cast<float*>(&rgb);	
		}
	
		boost::clear_vertex(*iter, mst);
	}
		

	cout<<"prune done\n";

	//refine
#if 1
	//find parents and children
    BGL_FORALL_VERTICES(v, mst, sv_graph_t) {

        if (boost::out_degree(v, mst) == 0) continue;

        float v_x = cloud->points[mst[v].index].z;

        BGL_FORALL_ADJ(v, adj, mst, sv_graph_t) {

            float adj_x = cloud->points[mst[adj].index].z;

			if (boost::out_degree(adj, mst) == 0) continue;

            if (adj_x > v_x) {

                mst[v].parents.push_back(adj);

            }
            else if (adj_x < v_x) {

                mst[v].children.push_back(adj);
            }

        }

    }

	//refine branching point 

    const float stop_branching_dist = 0.006f;

    while (true)
    {
        bool m = false;

        int lowest_branch = - 1;
        bool first = 1;
        for (int v = boost::num_vertices(mst) - 1; v > 0; v--) {
            
            if (boost::out_degree(v, mst) != 3 || mst[v].children.size() != 1 || mst[v].parents.size() != 2) continue;

            if (mst[mst[v].children[0]].parents.size() > 1) continue;
        
			if (mst[mst[v].children[0]].children.size() <1) continue;

            Eigen::Vector3f pp1 = cloud->points[mst[mst[v].parents[0]].index].getVector3fMap();

            Eigen::Vector3f pp2 = cloud->points[mst[mst[v].parents[1]].index].getVector3fMap();

            float dist = (pp1 - pp2).norm();

            if (dist < stop_branching_dist) continue;

            if (first) {
                lowest_branch = v; first = 0;
            }
            else if (cloud->points[mst[v].index].x < cloud->points[mst[lowest_branch].index].x)
                lowest_branch = v;

        }

        if (lowest_branch < 0) break;

        int v = lowest_branch;

        std::vector<int> indices = mst[v].cluster_indices;

        std::vector<cv::Point3f> cv_points(indices.size());

        cv::Mat labels;

        for (int i = 0; i < cv_points.size(); i++) {

            cv::Point3f point;
            point.x = cloud->points[indices[i]].x;
            point.y = cloud->points[indices[i]].y;
            point.z = cloud->points[indices[i]].z;
            cv_points[i] = point;
        }

        cv::Mat object_centers;

        cv::kmeans(cv_points, 2, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001), 10, cv::KMEANS_PP_CENTERS, object_centers);

        cv::Vec3f & center1 = object_centers.at<cv::Vec3f>(0, 0);

        cv::Vec3f & center2 = object_centers.at<cv::Vec3f>(1, 0);

        //float dist = std::sqrt(pow(center1[0] - center2[0], 2.f) + pow(center1[1] - center2[1], 2.f) + pow(center1[2] - center2[2], 2.f));

        //if (dist < stop_branching_dist) continue;

        PointT p1, p2;

        p1.x = center1[0]; p1.y = center1[1]; p1.z = center1[2];
        p2.x = center2[0]; p2.y = center2[1]; p2.z = center2[2];

        std::vector<PointT> point_pair(2);

        point_pair[0].x = center1[0]; point_pair[0].y = center1[1]; point_pair[0].z = center1[2];
        point_pair[1].x = center2[0]; point_pair[1].y = center2[1]; point_pair[1].z = center2[2];

        uint32_t rgb = 255 << 16 | 255 << 8 | 255;

        int p1_parent_idx = -1;

        if ((p1.getVector3fMap() - p2.getVector3fMap()).dot(
            cloud->points[mst[mst[v].parents[0]].index].getVector3fMap()
            - cloud->points[mst[mst[v].parents[1]].index].getVector3fMap())
        > 0.f)
            p1_parent_idx = 0;
        else
            p1_parent_idx = 1;

        mst[mst[v].children[0]].parents.clear();

        for (int p = 0; p < 2; p++) {

            std::vector<int> k_indices; std::vector<float> k_sqr_distances;

            if (kdtree.nearestKSearch(point_pair[p], 1, k_indices, k_sqr_distances) == 1) {

                SVVertexProperty vp;

                vp.index = k_indices[0];

                for (int i = 0; i < cv_points.size(); i++)
                    if (labels.at<int>(i, 0) == p)
                        vp.cluster_indices.push_back(indices[i]);

                uint32_t rgb_p = (255 * (1 - p) << 16 | 255 * p << 8 | 0);
#ifndef merged
                //for (auto & i : vp.cluster_indices)
                    //cloud->points[i].rgb = *reinterpret_cast<float*>(&rgb_p);
#endif
                vp.children = mst[v].children;

                vp.parents = mst[v].parents;

                sv_vertex_t new_vertex = boost::add_vertex(vp, mst);

                const int p_parent_idx = p == 0 ? p1_parent_idx : 1 - p1_parent_idx;

                boost::add_edge(new_vertex, mst[v].parents[p_parent_idx], mst);

                boost::add_edge(new_vertex, mst[v].children[0], mst);

                mst[mst[v].parents[p_parent_idx]].children[0] = new_vertex;

                mst[mst[v].children[0]].parents.push_back(new_vertex);

            }
            else cout << "kdtree for p fail\n";
        }

        boost::clear_vertex(v, mst);


    }//while


    cout << "refine branching done!" << endl;

#endif
#ifdef visualize


	pcl::visualization::PCLVisualizer viewer("ICP demo");

	int v1(0);
	int v2(1);

	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

	//viewer.addPointCloud(cloud_backup, to_string(cv::getTickCount()), v1);
	//viewer.addPointCloud(cloud, to_string(cv::getTickCount()), v2);

/*
	string temp0;
	BGL_FORALL_EDGES(e, skeleton_graph, sv_graph_t){
		temp0 = to_string(cv::getTickCount());
		viewer.addLine(cloud->points[skeleton_graph[boost::source(e, skeleton_graph)].index], cloud->points[skeleton_graph[boost::target(e, skeleton_graph)].index], 1, 1, 0, temp0, v1);
		viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 6, temp0, v1);
	}


	string temp1;
	BGL_FORALL_EDGES(e, mst, sv_graph_t){
		temp1 = to_string(cv::getTickCount());
		viewer.addLine(cloud->points[mst[boost::source(e, mst)].index], cloud->points[mst[boost::target(e, mst)].index], 1, 1, 0, temp1, v2);
		viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 6, temp1, v2);
	}
*/
	//viewer.spin();

#endif
	

/*********************holistic traits*****************************/


  //  	Eigen::Vector4f min_pt, max_pt;

		getMinMax3D(*cloud, min_pt, max_pt);

		float plant_height = (max_pt[2] - soil_z)*100.f;
	
		float plant_width = (max_pt[1] - min_pt[1])*100.f;

		float plant_length = (max_pt[0] - min_pt[0])*100.f;

		float total_area = getMeshArea(cloud_backup, false);

		cout<<"current_pot_id: "<<current_pot_id<<endl;


		//OOBB
		float delta_x, delta_y;
		float projected_area_x,projected_area_y,projected_area_z;
		float projected_convex_area_x,projected_convex_area_y,projected_convex_area_z;
		float convex_hull_area,convex_hull_volumn;
		float solidity, solidity_x, solidity_y, solidity_z;


		convex_hull_area = convexhull(cloud_backup, 1);
		convex_hull_volumn = convexhull(cloud_backup, 0);

		projected_area_x = projectedArea(cloud_backup, 0, 0);
		projected_area_y = projectedArea(cloud_backup, 1, 0);
		projected_area_z = projectedArea(cloud_backup, 2, 0);

		projected_convex_area_x = projectedArea(cloud_backup, 0, 1);
		projected_convex_area_y = projectedArea(cloud_backup, 1, 1);
		projected_convex_area_z = projectedArea(cloud_backup, 2, 1);

		solidity = total_area/convex_hull_area;
		solidity_x = projected_area_x/projected_convex_area_x;
		solidity_y = projected_area_x/projected_convex_area_y;
		solidity_z = projected_area_x/projected_convex_area_z;

		float LAI = total_area/projected_area_z;

		float aspect_ratio = plant_length/plant_width;


		cout<<"plant_height: "<<plant_height<<" plant_width: "<<plant_width<<" plant_length: "<<plant_length<<endl;

		cout<<"aspect_ratio: "<<aspect_ratio<<" solidity: "<<solidity<<" solidity_y: "<<solidity_y<<endl;


		//write to file 

#ifndef visualize


		outFile.open(holistic_measurement_path, ios::app);

		outFile << current_pot_id <<"," <<date<<","<< plant_height<<","<<plant_length<<","<<plant_width <<",";
		outFile <<total_area<<","<<convex_hull_area<<","<<convex_hull_volumn<<","<<LAI<<","<<aspect_ratio<<",";
		outFile<<projected_area_x<<","<<projected_area_y<<","<<projected_area_z<<",";
		outFile<<projected_convex_area_x<<","<<projected_convex_area_y<<","<<projected_convex_area_z<<",";
		outFile << solidity<<","<<solidity_x<<","<<solidity_y<<","<<solidity_z<<endl;
	
		outFile.close();

#endif
	
	//partial segment

	//leaf segment
	std::vector<std::vector<sv_vertex_t>> leaf_segment_vec;
	std::vector<std::vector<sv_vertex_t>> partial_segment_vec;

	std::vector<Leaf> leaf_vec;

	std::vector<sv_vertex_t> stem_vec;

    std::vector<bool> visited_map(boost::num_vertices(mst), false);

    bool leaf_candidate = false;

	PointCloudT::Ptr stem_centroid(new PointCloudT);

	pcl::PointIndices::Ptr plant_indices(new pcl::PointIndices); 

    BGL_FORALL_VERTICES(v, mst, sv_graph_t) {

        const int degree = boost::out_degree(v, mst);

        if (degree != 1 ) continue;
		//if (degree !=1 && degree !=3)  continue;

		//if (cloud->points[mst[v].index].z - min_pt(2) < 2.f*slice_thickness) continue;  //close to soil

        if (visited_map[v] == true) continue;

        leaf_candidate = false;

        visited_map[v] = true;

        std::vector<sv_vertex_t> segment;

        segment.push_back(v);

        sv_vertex_t cur_v = v;

        bool first_time = true;

        while (true) {

            BGL_FORALL_ADJ(cur_v, adj, mst, sv_graph_t) {

                visited_map[adj] = true;

                if (first_time) {

                    segment.push_back(adj);

                    cur_v = adj;

                    first_time = false;

                    continue;
                }


                if (adj == segment.at(segment.size() - 2))  continue;

                segment.push_back(adj);

                cur_v = adj;

                break;
            }

            if (boost::out_degree(cur_v, mst) == 1){ //usually noise
				partial_segment_vec.push_back(segment);
				for(auto &i:segment)
					boost::clear_vertex(i, mst);
				break;
			} 

            if (boost::out_degree(cur_v, mst) >= 3){

				//if (mst[cur_v].parents.size() ==2 ){

					if(cloud->points[mst[segment.front()].index].z - cloud->points[mst[segment.back()].index].z > -1.f*slice_thickness){ //
					
						mst[cur_v].start_point = true;
				
						leaf_candidate = true;
						
						stem_vec.push_back(cur_v);

					}
					else{
						for(auto &i:segment)
								stem_vec.push_back(i);
					}

                break;
            } 
        }

		if(leaf_candidate){

			leaf_segment_vec.push_back(segment); 
			
			Leaf leaf;
		
			if(segment.size()<4) continue;

			//cout<<"segment.size(): "<<segment.size()<<endl;

			for (int i = 0; i < segment.size(); i++)
                leaf.node_vec.push_back(segment.at(i));

			leaf_vec.push_back(leaf);

		}
    }

	cout << "leaf_segment_vec size: " << leaf_segment_vec.size() << endl;
	cout << "leaf_vec size: " << leaf_vec.size() << endl;
	cout << "stem_vec size: " << stem_vec.size() << endl;

	if(leaf_vec.size()<1){ 
	
		cout<<" the plant is too small and no leaf candidate\n";

		return -1;
	}
	
	BGL_FORALL_VERTICES(v, mst, sv_graph_t) {

        const int degree = boost::out_degree(v, mst);

 		if (visited_map[v] == true) continue;

		if(degree==0) continue;
		
		stem_vec.push_back(v); 

		plant_indices->indices.insert(plant_indices->indices.end(), mst[v].cluster_indices.begin(), mst[v].cluster_indices.end());

	}
	
	cout << "stem_vec size: " << stem_vec.size() << endl;

	//refine leaf end
	
	for (auto &i:leaf_vec){

		float endx, deltax;	

		//getMinMax3D(*cloud, mst[i.node_vec.front()].cluster_indices, min_pt, max_pt);

		endx= cloud->points[mst[i.node_vec.front()].index].x;
		
		int head;

		if(endx > cloud->points[mst[i.node_vec.back()].index].x){  //right hand
			deltax = -100.f;
			for(auto &j:mst[i.node_vec.front()].cluster_indices){
				if(cloud->points[j].x - endx>deltax){  //find the closest one
					head = j;
					deltax= cloud->points[j].x - endx;
					}
			}
		}
		else{
			deltax = 100.f;
			for(auto &j:mst[i.node_vec.front()].cluster_indices){
				if(cloud->points[j].x - endx < deltax){  //find the closest one
					head = j;
					deltax= cloud->points[j].x - endx;
					}
			}
		}
		
		
		SVVertexProperty vp;
		vp.index = head;
		sv_vertex_t new_vertex = boost::add_vertex(vp, mst);
		sv_edge_t edge; bool edge_added;

        boost::tie(edge, edge_added) = boost::add_edge(i.node_vec.front(), new_vertex, mst);

		i.node_vec.insert(i.node_vec.begin(), new_vertex);
#ifdef visualize
		//viewer.addText3D("tip", cloud->points[head], 0.005, 1.0, 1.0, 1.0,to_string(cv::getTickCount()), v2);
#endif
	}


	//stem detection
	Eigen::Vector3f line_p, line_d, major_vector;
	pcl::PCA<PointT> pca;

	for (auto &i:stem_vec) 
        stem_centroid->points.push_back(cloud->points[mst[i].index]);

	pca.setInputCloud(stem_centroid);

	line_d = -pca.getEigenVectors().col(0);

	Eigen::Vector4f center;

	pcl::compute3DCentroid(*stem_centroid, center);

	for(int i=0;i<3; i++)
		line_p[i] = center[i];

/********************************************individual leaf traits***************************/

	PointCloudT::Ptr cloud_projected(new PointCloudT);

	projectCloudy(cloud,cloud_projected);

	srand(time(NULL));
   	int a, b, c;
	PointT po;
	Eigen::Vector3f p2, p3;


	PointCloudT::Ptr colored_cloud(new PointCloudT);
	colored_cloud->clear();

	pca.setInputCloud(cloud);

	pcl::PCA<PointT> pca_projected;

	pca.setInputCloud(cloud_projected);

	Eigen::Vector3f eigen_values;

	//leaf area and length
   	for (int i = leaf_vec.size()-1; i >=0; i--){

		Leaf leaf = leaf_vec.at(i);

		int s = leaf.node_vec.size();
 
/*
        if (cloud->points[mst[leaf.node_vec.at(s-1)].index].z > cloud->points[mst[leaf.node_vec.at(s-2)].index].z) {
            cout << "erased: " << i <<", node: "<<leaf.node<< " junction node is higher than the neighbor\n";
            leaf_vec.erase(leaf_vec.begin() + i);
            continue;
        }*/
		
	  	pcl::PointIndices::Ptr leaf_area_indices(new pcl::PointIndices);  //for leaf area

        for (int j = 0; j < leaf.node_vec.size()-1; j++) {

			p2 = cloud->points[mst[leaf.node_vec[j]].index].getVector3fMap();

            p3 = cloud->points[mst[leaf.node_vec[j+1]].index].getVector3fMap();

			leaf_vec.at(i).length += (p2 - p3).norm();

            sv_vertex_t v_temp = leaf.node_vec.at(j);

            leaf_area_indices->indices.insert(leaf_area_indices->indices.end(), mst[v_temp].cluster_indices.begin(), mst[v_temp].cluster_indices.end()); //for leaf area

        }	

		PointCloudT::Ptr leaf_cloud(new PointCloudT);

        pcl::copyPointCloud(*cloud, *leaf_area_indices, *leaf_cloud);

        leaf_vec.at(i).area = getMeshArea(leaf_cloud, 0);

		leaf_vec.at(i).node_height = cloud->points[mst[leaf.node_vec.back()].index].z - soil_z;

		a = (int)(rand() % 256);
        b = (int)(rand() % 256);
        c = (int)(rand() % 256);
        rgb = a << 16 | b << 8 | c;

		leaf_area_indices->indices.insert(leaf_area_indices->indices.end(), mst[leaf.node_vec.back()].cluster_indices.begin(), mst[leaf.node_vec.back()].cluster_indices.end()); 

        for (auto &j : leaf_area_indices->indices){
			po = cloud->points[j];	
			po.rgb = *reinterpret_cast<float*> (&rgb); 
			colored_cloud->points.push_back(po);
		}

		//for curvature

		//pca.setIndices(leaf_area_indices);
		//pca_projected.setIndices(leaf_area_indices);

		//eigen_values = pca.getEigenValues();

		//cout<<"eigenvalues: "<<eigen_values(0)<<", "<<eigen_values(1)<<", "<<eigen_values(2)<<endl;

		//eigen_values = pca_projected.getEigenValues();

		//cout<<"eigenvalues: "<<eigen_values(0)<<", "<<eigen_values(1)<<", "<<eigen_values(2)<<endl;

	}


	//stem inliers

	getMinMax3D(*stem_centroid, min_pt, max_pt);

	Eigen::Vector3f p, d;

	a = (int)(rand() % 256);
    b = (int)(rand() % 256);
    c = (int)(rand() % 256);
    rgb = a << 16 | b << 8 | c;
	
	for(int i=0;i<cloud->points.size(); i++){

		po = cloud->points[i];

		if(po.z < min_pt[2] || po.z > (max_pt[2] + 2.f*slice_thickness))
			continue;

        p = po.getVector3fMap();

        d = line_d.cross(line_p - p);

        if(d.norm() / line_d.norm() < stem_radius/2.f){

			po.rgb = *reinterpret_cast<float*> (&rgb); 
			colored_cloud->points.push_back(po);

		}
	}	


	for(int i=0;i<colored_cloud->points.size(); i++){

		po = colored_cloud->points[i];

		if(po.z < min_pt[2] || po.z > (max_pt[2] + 2.f*slice_thickness))
			continue;

        p = po.getVector3fMap();

        d = line_d.cross(line_p - p);

        if(d.norm() / line_d.norm() < stem_radius/2.f){

			colored_cloud->points[i].rgb = *reinterpret_cast<float*> (&rgb); 
		}
	}
//leaf angle and node_distance

	PointCloudT::Ptr cloud_node(new PointCloudT);

	pca.setInputCloud(cloud);

   	for (int i = leaf_vec.size()-1; i >=0; i--){

		Leaf leaf = leaf_vec.at(i);

		int s = leaf.node_vec.size();

		//leaf angle
		pcl::PointIndices::Ptr leaf_angle_indices(new pcl::PointIndices);

		pcl::PointIndices::Ptr leaf_curvature_indices(new pcl::PointIndices);

        for (int j = leaf.node_vec.size()-2; j >=0; j--) {

			leaf_curvature_indices->indices.push_back(mst[leaf.node_vec[j]].index);
		}

        for (int j = leaf.node_vec.size()-2; j >=0; j--) {


			leaf_angle_indices->indices.push_back(mst[leaf.node_vec[j]].index);

			if(leaf_angle_indices->indices.size()>=5 || leaf_angle_indices->indices.size()>=s/2) break;

		}

		if (leaf_angle_indices->indices.size() < 3) {
            cout << "erased: " << i << ", node: " << leaf.node << " too less indices for leaf angle"<< endl;
            leaf_vec.erase(leaf_vec.begin() + i);
            continue;
        }

		pca.setIndices(leaf_angle_indices);

        major_vector = pca.getEigenVectors().col(0);

        float abs_cos = std::abs(major_vector.dot(line_d));

        leaf_vec.at(i).angle = acos(abs_cos) / PI *180.f;
		
		Eigen::Vector3f line_d_2d(line_d(0), 0, line_d(2));

        Eigen::Vector3f major_vector_2d(major_vector(0), 0, major_vector(2));

        abs_cos = std::abs(major_vector_2d.dot(line_d_2d) / major_vector_2d.norm() / line_d_2d.norm());

        leaf_vec.at(i).angle_2d = acos(abs_cos) / PI *180.f;

		rgb = 255 << 16 | 255 << 8 | 255;
	
        for (auto & i : leaf_angle_indices->indices) {
            po = cloud->points[i];
            po.rgb = *reinterpret_cast<float*> (&rgb);
            cloud_node->points.push_back(po);
        }


		pca.setIndices(leaf_curvature_indices);

		eigen_values = pca.getEigenValues();

		cout<<"eig: "<<eigen_values(0)<<", "<<eigen_values(1)<<", "<<eigen_values(2)<<" curvature: "<<eigen_values(1)/eigen_values(0)<<endl;

		leaf_vec.at(i).curvature = eigen_values(1)/eigen_values(0);
		
		Eigen::Vector3f Zaxis(0,0,1.f);
	
		abs_cos = std::abs(major_vector.dot(Zaxis));

		leaf_vec.at(i).angle_vertical = acos(abs_cos) / PI *180.f;

		//angle_vertical

		//cout<<"i: "<<i<< "leaf_angle_indices: "<<leaf_angle_indices->indices.size()<<endl;
		
	}

//sort leaf_vec by node distance

	std::sort(leaf_vec.begin(),leaf_vec.end(),lcompare);

	int leaf_number = 1;


   	for (int j = 0; j<leaf_vec.size(); j++){

		leaf_vec.at(j).leaf_num = leaf_number;
		
		Leaf i = leaf_vec.at(j);

		leaf_number++;

	}

#ifndef visualize
	outFile.open(leaf_measurement_path, ios::app);

   	for (int j = 0; j<leaf_vec.size(); j++){

		Leaf i = leaf_vec.at(j);

		outFile << current_pot_id <<","<<date<<","<< i.leaf_num<<","<<i.area<<","<<(i.length)*100.f<<"," <<i.curvature <<","<<(i.node_height)*100.f<<","<<i.angle<<","<<i.angle_2d<<","<<i.angle_vertical<<"\n";

	}

	outFile.close();

	
	colored_cloud->width = 1;
	colored_cloud->height = colored_cloud->points.size();

	pcl::io::savePCDFile(current_exp_folder+"leafSegResults/" + current_pot_id +".pcd", *colored_cloud);
#endif

//visualize

#ifdef visualize	


   	for (int i = leaf_vec.size()-1; i >=0; i--){	
		Leaf leaf = leaf_vec.at(i);		
		//viewer.addText3D("tip", cloud->points[head], 0.005, 1.0, 1.0, 1.0,to_string(cv::getTickCount()), v2);
		string textstr = "num: "+to_string(leaf.leaf_num)+"\narea: "+to_string(leaf.area) + "\nlength: "+to_string(leaf.length)+ "\nangle: "+to_string(leaf.angle)+ "\ncurvature: "+to_string(leaf.curvature);
		viewer.addText3D(textstr, cloud->points[mst[leaf.node_vec[0]].index], 0.003, 1.0, 1.0, 1.0,to_string(cv::getTickCount()), v1);
	}


    for (auto &i:leaf_vec) {

        rgb = 0 << 16 | 255 << 8 | 0;
        
        po = cloud->points[mst[i.node_vec.front()].index];

        po.rgb = *reinterpret_cast<float*> (&rgb);

        cloud_node->points.push_back(po);

        rgb = 0 << 16 | 0 << 8 | 0;

        po = cloud->points[mst[i.node_vec.back()].index];

        po.rgb = *reinterpret_cast<float*> (&rgb);

        cloud_node->points.push_back(po);

    }

	//viewer.addPointCloud<PointT>(cloud_node, "node", v2);
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "node", v2);

	downsample(cloud_backup, cloud_backup, 0.0007);
	viewer.addPointCloud(cloud_backup, to_string(cv::getTickCount()), v2);

	string temp1;

	BGL_FORALL_EDGES(e, mst, sv_graph_t){
		temp1 = to_string(cv::getTickCount());
		viewer.addLine(cloud->points[mst[boost::source(e, mst)].index], cloud->points[mst[boost::target(e, mst)].index], 1, 1, 0, temp1, v2);
		viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 6, temp1, v2);
	}

	BGL_FORALL_EDGES(e, mst, sv_graph_t){
		temp1 = to_string(cv::getTickCount());
		viewer.addLine(cloud->points[mst[boost::source(e, mst)].index], cloud->points[mst[boost::target(e, mst)].index], 1, 1, 0, temp1, v1);
		viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 6, temp1, v1);
	}

	viewer.addPointCloud(colored_cloud, to_string(cv::getTickCount()), v1);

	viewer.registerPointPickingCallback(&pp_callback);


//stem

    pcl::visualization::PointCloudColorHandlerCustom<PointT> red(stem_centroid, 255, 0, 0);
    viewer.addPointCloud<PointT>(stem_centroid, red, "stem_centroid", v1);	
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "stem_centroid", v1);


	pcl::ModelCoefficients coefficients;

    coefficients.values.resize(6);

    coefficients.values[0] = line_p(0);

    coefficients.values[1] = line_p(1);

    coefficients.values[2] = line_p(2);



    coefficients.values[3] = -0.1f*line_d(0);

    coefficients.values[4] = -0.1f*line_d(1);

    coefficients.values[5] = -0.1f*line_d(2);

    viewer.addLine(coefficients, to_string(cv::getTickCount()), v1);

	pcl::ModelCoefficients coefficients2;

    coefficients2.values.resize(6);

    coefficients2.values[0] = line_p(0);

    coefficients2.values[1] = line_p(1);

    coefficients2.values[2] = line_p(2);



    coefficients2.values[3] = 0.05f*line_d(0);

    coefficients2.values[4] = 0.05f*line_d(1);

    coefficients2.values[5] = 0.05f*line_d(2);
    viewer.addLine(coefficients2, to_string(cv::getTickCount()), v1);

	viewer.spin();
#endif


	return 1;
}

inline void projectCloudy(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_out){

        	PointCloudT::Ptr cloud_backup(new PointCloudT);

            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
			//z=0 after projected
            coefficients->values.resize(4);
            coefficients->values[0] = 0;
            coefficients->values[1] = 1;
            coefficients->values[2] = 0;
            coefficients->values[3] = 0;

            pcl::ProjectInliers<PointT> proj;
            proj.setModelType(pcl::SACMODEL_PLANE);
            proj.setInputCloud(cloud);
            proj.setModelCoefficients(coefficients);
            proj.filter(*cloud_backup);

			cloud_out->clear();

			*cloud_out+=*cloud_backup;

}


inline void alignCloud(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_out){


//project cloud
        	PointCloudT::Ptr cloud_backup(new PointCloudT);

            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
			//z=0 after projected
            coefficients->values.resize(4);
            coefficients->values[0] = 0;
            coefficients->values[1] = 0;
            coefficients->values[2] = 1;
            coefficients->values[3] = 0;

            pcl::ProjectInliers<PointT> proj;
            proj.setModelType(pcl::SACMODEL_PLANE);
            proj.setInputCloud(cloud);
            proj.setModelCoefficients(coefficients);
            proj.filter(*cloud_backup);
  
//get direction      
	pcl::PCA<PointT> pca;
	pca.setInputCloud(cloud_backup);

	Eigen::Vector3f major_vector = pca.getEigenVectors().col(0); //major_vector

	Eigen::Vector3f unity(1,0,0);


    float abs_cos = abs(major_vector.dot(unity) / major_vector.norm());

	float theta = acos(abs_cos) / PI *180.f;

	float theta_rad;

	if(major_vector(0)*major_vector(1)>0)
		theta_rad = -1.f *acos(abs_cos);
	else
		theta_rad = acos(abs_cos);

	cout<<"major_vector: "<<major_vector<<"\nunity: "<<unity<<"\ntheta: "<<theta<<endl;

	Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();

	  //float theta = M_PI/4; // The angle of rotation in radians
	  transform_1 (0,0) = cos (theta_rad);
	  transform_1 (0,1) = -sin(theta_rad);
	  transform_1 (1,0) = sin (theta_rad);
	  transform_1 (1,1) = cos (theta_rad);

	  // Define a translation of 0 meters on the x axis.
	  transform_1 (0,3) = 0;

	  // Print the transformation
	  printf ("Method #1: using a Matrix4f\n");
	  std::cout << transform_1 << std::endl;

    PointCloudT::Ptr transformed_cloud(new PointCloudT);

  	pcl::transformPointCloud (*cloud, *transformed_cloud, transform_1);


#ifdef visualize
	pcl::visualization::PCLVisualizer viewer("ICP demo");

	int v1(0);
	int v2(1);

	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);


	viewer.addPointCloud(cloud, to_string(cv::getTickCount()), v1);

	viewer.addPointCloud(cloud_backup, to_string(cv::getTickCount()), v2);

	viewer.addPointCloud(transformed_cloud, to_string(cv::getTickCount()), v1);

	viewer.addCoordinateSystem (1.0);

	viewer.registerPointPickingCallback(&pp_callback);

	viewer.spin();
#endif

	cloud_out->clear();

	*cloud_out+=*transformed_cloud;

}

double soilDetection(PointCloudT::Ptr cloud){


			PointCloudT::Ptr ground(new PointCloudT);
			*ground+= *cloud;
			PointCloudT::Ptr cloud_soil(new PointCloudT);

			pcl::PointIndices::Ptr indices(new pcl::PointIndices);
			pcl::PassThrough<PointT> pass_for_hough;
			pass_for_hough.setInputCloud(cloud);
			pass_for_hough.setFilterFieldName("z");
			pass_for_hough.setFilterLimitsNegative(false);
			pass_for_hough.setFilterLimits(-10.f, 0.03f);
			pass_for_hough.filter(indices->indices);
			pass_for_hough.filter(*cloud_soil);

			PointIndices::Ptr inliers(new PointIndices);

		potDetection(cloud_soil);

            SACSegmentation<PointT> seg;

            ModelCoefficients::Ptr coeff(new ModelCoefficients);

            seg.setOptimizeCoefficients(true);

            seg.setModelType(SACMODEL_PERPENDICULAR_PLANE);

            seg.setMethodType(SAC_RANSAC);

            seg.setDistanceThreshold(0.002); //0.002

            seg.setAxis(Eigen::Vector3f::UnitZ());

            seg.setEpsAngle(5. / 180 * M_PI);

            //seg.setInputCloud(ground);

			seg.setInputCloud(cloud);

			seg.setIndices(indices);

            seg.segment(*inliers, *coeff);

            Eigen::Vector4f centroid;

			double ground_z; 

            if (inliers->indices.size() == 0) {

                return -1.f;

            }

            else {

                pcl::compute3DCentroid(*cloud, *inliers, centroid);

                ground_z = centroid[2];

            }

		uint32_t rgb;
		rgb = 255 << 16 | 0 << 8 | 0;

		for(auto &i:indices->indices)
			cloud->points[i].rgb = *reinterpret_cast<float*>(&rgb);

		cout<<"ground_z: "<<ground_z<<endl;

#if 0
	PointCloudT::Ptr temp(new PointCloudT);
	pcl::copyPointCloud(*cloud, *inliers, *temp);


	pcl::visualization::PCLVisualizer viewer("ICP demo");

	int v1(0);
	int v2(1);
	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

	viewer.registerPointPickingCallback(&pp_callback);

	viewer.addPointCloud(cloud, to_string(cv::getTickCount()), v1);

    pcl::visualization::PointCloudColorHandlerCustom<PointT> red(temp, 255, 0, 0);
    viewer.addPointCloud<PointT>(temp, red, "pot", v1);


	viewer.addPointCloud(ground, to_string(cv::getTickCount()), v2);

	viewer.spin();
#endif		

	return ground_z;
}

void potDetection(PointCloudT::Ptr cloud){

	PointCloudT::Ptr cloud_pot(new PointCloudT);

	PointCloudT::Ptr cloud_icp(new PointCloudT);

	float center_x = -0.06f;

	float center_y = 0.43f;

	float center_z = 0.029f;

	float rad = 0.045;

	for (int i = 0;i < 360;++i) {

		float s = sin(i*3.1415926 / 180);

		float c = cos(i*3.1415926 / 180);

		for (float m = 0.0f;m < 0.001f;m += 0.0001f) {

			PointT point;

			point.x = center_x + (rad - m)*s;

			point.y = center_y + (rad - m)*c;

			point.z = center_z;

			cloud_pot->points.push_back(point);

		}
	}

	pcl::IterativeClosestPoint<PointT, PointT> icp;

	icp.setInputTarget(cloud);

	icp.setInputSource(cloud_pot);

	icp.align(*cloud_icp);

	computeCentroid(*cloud_icp, pot_center);

	cout << "pot_center: " << pot_center.x << ", " << pot_center.y << ", " << pot_center.z << endl;
#if 0

	pcl::visualization::PCLVisualizer viewer("ICP demo");


	int v1(0);

	int v2(1);

	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);

	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

	viewer.registerPointPickingCallback(&pp_callback);


	viewer.addPointCloud(cloud, to_string(cv::getTickCount()), v1);
	viewer.addPointCloud(cloud, to_string(cv::getTickCount()), v2);

       pcl::visualization::PointCloudColorHandlerCustom<PointT> red(cloud_icp, 255, 0, 0);

       viewer.addPointCloud<PointT>(cloud_icp, red, "pot", v1);

       pcl::visualization::PointCloudColorHandlerCustom<PointT> red2(cloud_pot, 255, 0, 0);
       viewer.addPointCloud<PointT>(cloud_pot, red2, "pot2", v2);

	viewer.spin();
#endif		

}

double getMeshArea(PointCloudT::Ptr cloud, bool projected) {

        PointCloudT::Ptr cloud_backup(new PointCloudT);

        if (projected) {
            
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
            coefficients->values.resize(4);
            coefficients->values[0] = 0;
            coefficients->values[1] = 1.0;
            coefficients->values[2] = 0;
            coefficients->values[3] = 0;

            pcl::ProjectInliers<PointT> proj;
            proj.setModelType(pcl::SACMODEL_PLANE);
            proj.setInputCloud(cloud);
            proj.setModelCoefficients(coefficients);
            proj.filter(*cloud_backup);
        
        }
        else {
            *cloud_backup += *cloud;
        }
        

        PointCloudT::Ptr cloud_temp(new PointCloudT);

        PointCloudT::Ptr cloud_temp0(new PointCloudT);

		PointCloudT::Ptr cloud_temp1(new PointCloudT);

        pcl::VoxelGrid<PointT> sor;
        //sor.setInputCloud(cloud_temp0);
        sor.setInputCloud(cloud_backup);
        sor.setLeafSize(0.003f, 0.003f, 0.003f); //0.002
        sor.filter(*cloud_temp1);  //cloud_temp0

        sor.setInputCloud(cloud_temp1);
        sor.setLeafSize(0.002f, 0.002f, 0.002f); //0.002
        sor.filter(*cloud_temp0);  //half size


        MovingLeastSquares<PointT, PointT> mls;
        mls.setInputCloud(cloud_temp0);
        mls.setSearchRadius(0.005);
        mls.setPolynomialFit(true);
        mls.process(*cloud_temp);                                


        pcl::NormalEstimation<PointT, pcl::Normal> n;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
        tree->setInputCloud(cloud_temp);
        n.setInputCloud(cloud_temp);
        n.setSearchMethod(tree); //
		//n.setKSearch(20);
        n.setRadiusSearch(0.01f);
        n.compute(*normals);

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::concatenateFields(*cloud_temp, *normals, *cloud_with_normals);

        pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;
        pcl::PolygonMesh triangles;
        pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);

        gp3.setSearchRadius(0.01f); //0.01

        gp3.setMu(2.5);
        gp3.setMaximumNearestNeighbors(100);
        gp3.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees
        gp3.setMinimumAngle(M_PI / 18); // 10 degrees
        gp3.setMaximumAngle(2 * M_PI / 3); // 120 degrees
        gp3.setNormalConsistency(false);

        // Get result
        gp3.setInputCloud(cloud_with_normals);
        gp3.setSearchMethod(tree2);
        gp3.reconstruct(triangles);


#if 0
        pcl::visualization::PCLVisualizer viewer("ICP demo");

        int v1(0);
        int v2(1);
        viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
        viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

        //viewer.addPointCloud(cloud_temp0, to_string(cv::getTickCount()), v1);

        viewer.addPointCloud(cloud_temp, to_string(cv::getTickCount()), v1);

	//	viewer.addPointCloud(cloud_backup, to_string(cv::getTickCount()), v1);

        viewer.addPolygonMesh(triangles,to_string(cv::getTickCount()),v2);

        viewer.registerPointPickingCallback(&pp_callback);

        viewer.spin();

		//return 0;

#endif

        double area = 0;
        size_t a, b, c;
        Eigen::Vector3d A, B, C, AB, AC, M;
        for (size_t i = 0; i < triangles.polygons.size(); ++i) {
            a = triangles.polygons[i].vertices[0];
            b = triangles.polygons[i].vertices[1];
            c = triangles.polygons[i].vertices[2];
            A(0) = cloud_temp->points[a].x;
            A(1) = cloud_temp->points[a].y;
            A(2) = cloud_temp->points[a].z;
            B(0) = cloud_temp->points[b].x;
            B(1) = cloud_temp->points[b].y;
            B(2) = cloud_temp->points[b].z;
            C(0) = cloud_temp->points[c].x;
            C(1) = cloud_temp->points[c].y;
            C(2) = cloud_temp->points[c].z;
            AB = A - B;
            AC = A - C;
            M = AB.cross(AC);
            area += 0.5*(M.norm());
        }
		cout<<"area: "<<area * 10000.f<<endl;

        return area * 10000.f;
    }


double projectedArea(PointCloudT::Ptr cloud, int cor, bool convex){

//project cloud
			cout<<"axis: "<<cor<<" (0: x=0, 1: y=0, 2: z=0)\n";

        	PointCloudT::Ptr cloud_projected(new PointCloudT);

            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
			//z=0 after projected
            coefficients->values.resize(4);

			for (int i=0;i<4;i++){
				if(i==cor)
					coefficients->values[i] = 1;
				else
					coefficients->values[i] = 0;
			}

            pcl::ProjectInliers<PointT> proj;
            proj.setModelType(pcl::SACMODEL_PLANE);
            proj.setInputCloud(cloud);
            proj.setModelCoefficients(coefficients);
            proj.filter(*cloud_projected);
  
			double area = getMeshArea(cloud_projected, 0);

			if(convex){
				double convex_area = convexhull(cloud_projected, 1);
				return convex_area;
			}

//convexhull

			return area;

}


double convexhull(PointCloudT::Ptr cloud, bool area){

			std::vector<pcl::Vertices> vertices_;

			ConvexHull<PointT> chull;
			PointCloudT::Ptr cloud_chull(new PointCloudT);
			chull.setInputCloud(cloud);
			chull.setComputeAreaVolume(true);
			chull.reconstruct(*cloud_chull, vertices_);

			float convex_area, convex_hull_volume;

			convex_area = chull.getTotalArea() * pow(100.f, 2);
			convex_hull_volume = chull.getTotalVolume()*pow(100.f, 3);


#if 0

			cout<<"convex_area: "<<convex_area<<" convex_hull_volume: "<<convex_hull_volume<<endl;

	pcl::visualization::PCLVisualizer viewer("ICP demo");


	int v1(0);
	int v2(1);

	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);


	viewer.registerPointPickingCallback(&pp_callback);

	viewer.addPointCloud(cloud, to_string(cv::getTickCount()), v1);
	viewer.addPointCloud(cloud, to_string(cv::getTickCount()), v2);

	//pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_target_color_h(cloud_chull, 180, 20, 20);
	//viewer.addPointCloud(cloud_chull, cloud_target_color_h, "convex", v2);

	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "convex", v2);


    viewer.addPolygonMesh<PointT> (cloud_chull, vertices_, "hull", v2);

	viewer.spin();


#endif

			if(area)
				return convex_area;
			else
				return convex_hull_volume;



}
