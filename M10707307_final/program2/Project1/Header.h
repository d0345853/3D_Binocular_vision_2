#include <iostream>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <fstream>				//文件儲存
#include <vector>				//動態矩陣
#include <cstring>
#include <string>				//字串
#include <algorithm>			//基本演算法
#include <omp.h>				//平行處理
#include <math.h>				//math符號運算
#include <sstream>
#include <iomanip>				//單位經度控制

using namespace std;
using namespace cv;


//***		Figure file			***//
string l_orig_pic = "L\\L_Color.JPG";								//左
string r_orig_pic = "R\\L_Color.JPG";								//右
Mat l_orig_img = imread(l_orig_pic);
Mat r_orig_img = imread(r_orig_pic);
int orig_rows = l_orig_img.rows, orig_cols= l_orig_img.cols;		//原始圖的長寬


//***		Figure file			***//
char L_image_path[16];												//300雷射圖片檔案
char R_image_path[16];												//300雷射圖片檔案
Mat L_scan;															//圖檔
Mat R_scan;
Mat L_scan_color;
Mat R_scan_color;
int r_color, g_color, b_color;										//取顏色


//***		XYZ file			***//
ofstream file_xyz;				
string file_name = "M10707307.txt";									//要改檔名要改這裡



//***		F point		***//		
int num = 11;														//特徵點個數   15個點
vector<long double> r_x = { 377,  230,  346,  363,  305,  462,	372,  219,  372,  438,  380,   242};	//特徵點分布
vector<long double> r_y = { 725,  534,  946,  333,  596,  678,  391,  705,  485,  776,	662,    614};
vector<long double> r_z = { 1,      1,	  1,	1,    1,    1,	  1,    1,    1,    1,	  1,      1};
vector<long double> l_x = { 427,  250,	357,  367,  353,  477,	384,  247,  401,  468,  428,    283};
vector<long double> l_y = { 713,  514,  939,  314,  580,  665,  374,  686,  469,  762,	650,     600};
vector<long double> l_z = { 1,	    1,	  1,    1,    1,    1,	  1,    1,    1,    1,	  1,       1};
//If you get more than 8 correspondences, least-square method or SVD may be used
//vector<long double> r_x = { 205,   95,   99,  156,  146,  325,	507,  345,  496,  386,  442,  445};	//特徵點分布
//vector<long double> r_y = {  57,  233,  270,  296,  210,  279,  296,  246,  211,  140,	90,    12};
//vector<long double> r_z = { 1,      1,	  1,	1,    1,    1,	  1,    1,    1,    1,	  1,    1};
//vector<long double> l_x = { 211,  252,	253,  362,  362,  514,	630,  551,  561,  437,  444,  445};
//vector<long double> l_y = { 99,  278,  320,  318,  223,  255,  238,  214,  185,  148,	105,   39};
//vector<long double> l_z = { 1,	    1,	  1,    1,    1,    1,	  1,    1,    1,    1,	  1,    1};

vector<long double> rn_x = r_x;	//特徵點分布
vector<long double> rn_y = r_y;
vector<long double> rn_z = r_z;
vector<long double> ln_x = l_x;
vector<long double> ln_y = l_y;
vector<long double> ln_z = l_z;


//***		F point		***//	
vector<long double> I_L;						//右邊投影到左邊的線
vector<long double> I_R;						//左邊投影到右邊的線
int x_3d;
int y_3d;
int z_3d;
//***		SVD參數		***//	
const int MAX_ITER = 200;						//迭代次數
const long double error_gate = 0.000000000001;	//最小誤差(如果太大  特徵值會遺失

int m = num;									//行
int n = 9;										//列	
int k = 9;										//rank=行
int X_lenght = 4;								//3D點長度

//***		SVD Matrix		***//
vector<vector<long double> > A;					//Input		A*h=0
vector<vector<long double> > F_hat;				//Output(轉移矩陣)
vector<vector<long double> > F;					//Output

vector<vector<long double> > U;					//U是一個(m,m)的矩陣
vector<long double> D;							//D是一個(m,n)的Diagonal Matrix（對角陣）  但只有取對角線的特徵值 
vector<vector<long double> > V;					//V是一個(n,n)的矩陣


//***		SVD Matrix		***//
vector<vector<long double> > B;					//Input		A*h=0
vector<long double> X_3D;						//Output

vector<vector<long double> > U2;				//U是一個(m,m)的矩陣
vector<long double> D2;							//D是一個(m,n)的Diagonal Matrix（對角陣）  但只有取對角線的特徵值 
vector<vector<long double> > V2;				//V是一個(n,n)的矩陣


//***		相機參數		***//	
vector<vector<long double> > Instrisic_L ;
vector<vector<long double> > Instrisic_R ;
vector<vector<long double> > Extrinsic_L ;
vector<vector<long double> > Extrinsic_R ;
vector<vector<long double> > P_L;
vector<vector<long double> > P_R;

//***		正規化參數		***//
//長度
vector<long double> l_lenght = l_x;
vector<long double> r_lenght = r_x;

//取平均
long double mean_r_x, mean_r_y;
long double mean_l_x, mean_l_y;

//取總平均
long double mean_r;
long double mean_l;

//尺寸轉換矩陣
vector<vector<long double> > l_T;
vector<vector<long double> > r_T;

//R 向量正規劃
long double r1_norm;
long double r2_norm;

double show_u;
double show_v;
double show_w;
	
vector<int> L_x_max;			//左圖片最亮點的X
vector<int> L_y_max;			//左圖片最亮點的X	
vector<int> L_val;
vector<long double> R_x_max;	//左圖片最亮點的X
vector<long double> R_y_max;	//左圖片最亮點的X	
vector<long double> R_val;
long double Rx = 0, Ry = 0;

int light_gate = 30;			//亮度閥值

//平行處理
int tid;
double R01 = 0, R02 = 0;



//***		測試輸入與輸出		***//
vector<long double>	 in_point = { 0,  0,  0};		//轉換前
vector<long double> out_point = { 0,  0,  0};		//轉換後