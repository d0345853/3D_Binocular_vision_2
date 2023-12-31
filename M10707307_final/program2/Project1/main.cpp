#include "Header.h"



//***		標準化		***//	
inline long double normalize(long double *x, int n) {//在 function 原型前面加上 inline 即可，如此前置處理器就會幫你將函式主體 copy 到呼叫函式的地方
	double r = 0, sq_r = 0;
	for (int i = 0; i < n; i++)					// 每個長度相加
		r += x[i] * x[i];
	sq_r = sqrt(r);								// 統一取開耕號的r

	if (sq_r < error_gate)						//當x的總體長度<error_gate時	
		return 0;
	for (int i = 0; i < n; i++)					//當x的總體長度>error_gate時，將x[i]/|x|，將向量x進行 nirmalizet成為单位向量
		x[i] /= sq_r;
	return sq_r;
}
inline long double normalize2(vector<long double> a) {
	long double sum = 0;
	int n = a.size();
	for (int i = 0; i < n; i++)
		sum += a[i];
	return sum / n;
}

//***		XY距離		***//	
inline std::vector<long double> get_lenght(vector<long double> x, vector<long double> y, long double x_mean, long double y_mean) {
	vector<long double> ans = x;
	int n = x.size();
	for (int i = 0; i < n; i++)
		ans[i] = sqrt(pow(x[i] - x_mean, 2) + pow(y[i] - y_mean, 2));		 //開更號(X^2+Y^2)  
	return ans;
}

//***		正交		***//	
inline void orthogonal(long double *a, long double *b, int n) {	//正交，讓|a|=1
	double r = 0;
	//**  product a*b  **//
	for (int i = 0; i < n; i++)
		r += a[i] * b[i];

	//**  orthogonal vector  **//
	for (int i = 0; i < n; i++)
		b[i] -= r * a[i];
}

//***		SVD		***//	
bool Used_SVD(vector<vector<long double> > A_SVD, int K_SVD, int M_SVD, vector<vector<long double> > &U_SVD, vector<long double> &D_SVD, vector<vector<long double> > &V_SVD) {
	int M = M_SVD;
	int N = K_SVD;

	//左右向量定義 (左向量計算U、右向量計算V)
	long double *Alpha = new long double[M];
	long double *Beta = new long double[N];
	long double *temp_Alpha = new long double[M];	 //temp_Alpha向量用于迭帶運算
	long double *temp_Beta = new long double[N];
	int col = 0;

	#pragma omp parallel for  schedule(static)
	for (int col = 0; col < K_SVD; col++) {
		long double diff = 1;
		long double r = -1;

		//生成向量 α  (比對之後疊加到U矩陣中)
		while (1) {
			for (int i = 0; i < M; i++)
				Alpha[i] = (long double)rand();						//隨機生成一个向量 α(1×M)  並用迭帶一直嘗試逼近正確值
			if (normalize(Alpha, M) > error_gate)				//當向量的絕對長度>error_gate，則完成left_vector向量，跳出loop；
				break;
		}

		//進入迭代
		for (int iter = 0; diff >= error_gate && iter < MAX_ITER; iter++) {

			//分配內存給  左迭代向量α  和右迭代向量β
			memset(temp_Alpha, 0, sizeof(long double)*M);
			memset(temp_Beta, 0, sizeof(long double)*N);

			/////////////////////////////  V  //////////////////////////////////////
			//生成右迭代向量β_next(1×N)
			for (int i = 0; i < M; i++)
				for (int j = 0; j < N; j++)
					temp_Beta[j] += Alpha[i] * A_SVD[i][j];		//向量β_next(1×N)=向量α(1×M)×A(M×N)矩阵疊加，透過隨機向量來取得
			r = normalize(temp_Beta, N);						//把β_next 進行標準化			
			if (r < error_gate)									//如果標準化<誤差則直接跳出 
				break;											//如果範圍太小(A矩陣或

			//β_next = V與β_next 進行正交運算
			for (int i = 0; i < col; i++)
				orthogonal(&V_SVD[i][0], temp_Beta, N);			//與右矩陣K X V正交，正交化得到β'_next
			normalize(temp_Beta, N);							//單位化β'_next(旋轉過後)

			////////////////////////////////  U  ///////////////////////////////////
			//生成右迭代向量α_next(1×M)			
			for (int i = 0; i < M; i++)
				for (int j = 0; j < N; j++)
					temp_Alpha[i] += temp_Beta[j] * A_SVD[i][j];//向量α_next(1×M)= A(M×N)×β_next'(N×1)疊加
			r = normalize(temp_Alpha, M);						//把α_next 進行標準化	
			if (r < error_gate)								    //如果標準化<誤差則直接跳出 
				break;

			//α_next = U與α_next 進行正交運算
			for (int i = 0; i < col; i++)
				orthogonal(&U_SVD[i][0], temp_Alpha, M);		//與左矩陣M X K正交，正交化得到α'_next
			normalize(temp_Alpha, M);							//單位化α'_next(旋轉過後)

			///////////////////////////////////////////////////////////////////
			//誤差判斷(用來決定是否終止疊代)
			diff = 0;
			for (int i = 0; i < M; i++) {
				long double d = temp_Alpha[i] - Alpha[i];			//計算前後兩個迭代向量的差距的平方(並疊加)
				diff += d * d;
			}

			//存放內存     
			memcpy(Alpha, temp_Alpha, sizeof(long double)*M);		//α=α'_next		U
			memcpy(Beta, temp_Beta, sizeof(long double)*N);			//β=β'_next		V
		}

		///////////////////////////  輸出  ////////////////////////////////////////
		//得到A矩陣之奇異值
		if (r >= error_gate) {										//若向量的模長(特徵向量對角線以外的值旁邊的數)>error_gate
			D_SVD[col] = r;											//D: r=A矩陣奇異值
			memcpy((char *)&U_SVD[col][0], Alpha, sizeof(long double)*M);//U
			memcpy((char *)&V_SVD[col][0], Beta, sizeof(long double)*N); //V  (我們只需要這一項)
		}
		else {
			break;
		}
	}

	//清空所有不必要的向量
	delete[] Alpha;
	delete[] temp_Alpha;
	delete[] Beta;
	delete[] temp_Beta;
	return 0;
}

/***************************************  主程式  ********************************************/
int main(int argc, char const *argv[]){

	//***		0.初始化		***///////////////////////////////////////////////////
	A.clear();
	F.clear();
	F_hat.clear();

	U.clear();
	D.clear();
	V.clear();

	Instrisic_L.clear();
	Instrisic_R.clear();
	Extrinsic_L.clear();
	Extrinsic_R.clear();

	B.clear();
	X_3D.clear();

	U2.clear();
	D2.clear();
	V2.clear();
	//***		1.矩陣大小定義		***///////////////////////////////////////////////
	// SVD參數
	A.resize(m);								//重新定義A大小
	for (int i = 0; i < m; i++)
		A[i].resize(n);

	F.resize(n / 3);							// 3 X 3
	for (int i = 0; i < n / 3; i++)
		F[i].resize(n / 3 , 0);

	F_hat.resize(n / 3);							// 3 X 3
	for (int i = 0; i < n / 3; i++)
		F_hat[i].resize(n / 3, 0);

	V.resize(k);								// 宣告V(K X N)矩陣
	for (int i = 0; i < k; i++)
		V[i].resize(n, 0);

	U.resize(k);								// M X K
	for (int i = 0; i < k; i++)
		U[i].resize(m, 0);

	D.resize(k, 0);								// K X K(只看對角線)

	// 正歸化參數
	l_T.resize(n / 3);								// 3 X 3	
	for (int i = 0; i < n / 3; i++)
		l_T[i].resize(n / 3, 0);

	r_T.resize(n / 3);								// 3 X 3
	for (int i = 0; i < n / 3; i++)
		r_T[i].resize(n/3, 0);

	//  投影直線參數
	I_L.resize(3, 0);
	I_R.resize(3, 0);

	// 3D參數
	B.resize(X_lenght);								//重新定義A大小
	for (int i = 0; i < X_lenght; i++)
		B[i].resize(X_lenght);


	V2.resize(X_lenght);								// 宣告V(K X N)矩陣
	for (int i = 0; i < X_lenght; i++)
		V2[i].resize(X_lenght, 0);

	U2.resize(X_lenght);								// M X K
	for (int i = 0; i < X_lenght; i++)
		U2[i].resize(X_lenght, 0);

	D2.resize(X_lenght, 0);								// K X K(只看對角線)

	////***		2.正規化		***//////////////////////////////////////////////
	//// 取平均
	mean_l_x = normalize2(l_x);
	mean_l_y = normalize2(l_y);
	mean_r_x = normalize2(r_x);
	mean_r_y = normalize2(r_y);

	////// 取絕對長度
	l_lenght = get_lenght(l_x, l_y, mean_l_x, mean_l_y);
	r_lenght = get_lenght(r_x, r_y, mean_r_x, mean_r_y);

	////// 總平均
	mean_l = normalize2(l_lenght);
	mean_r = normalize2(r_lenght);


	////// 右轉移矩陣
	r_T[0][0] = 1.414 / mean_r;				//scale
	r_T[1][1] = 1.414 / mean_r;
	r_T[2][0] = -1.414*mean_r_x / mean_r;	//shift
	r_T[2][1] = -1.414*mean_r_y / mean_r;
	r_T[2][2] = 1;

	////// 左轉移矩陣
	l_T[0][0] = 1.414 / mean_l;				//scale 
	l_T[1][1] = 1.414 / mean_l;
	l_T[0][2] = -1.414*mean_l_x / mean_l;	//shift
	l_T[1][2] = -1.414*mean_l_y / mean_l;
	l_T[2][2] = 1;

	//// T轉換之後的數值
	#pragma omp parallel for  schedule(static)
	for (int i = 0; i < num; i++) {							// L全部數值進行正規劃
		ln_x[i] = (l_x[i] - mean_l_x)*l_T[0][0];
		ln_y[i] = (l_y[i] - mean_l_y)*l_T[1][1];
		ln_z[i] = 1;
	}

	#pragma omp parallel for  schedule(static)				// R的轉制矩陣
	for (int i = 0; i < num; i++) {							// R全部數值進行正規劃
		rn_x[i] = (r_x[i] - mean_r_x) *r_T[0][0];
		rn_y[i] = (r_y[i] - mean_r_y) *r_T[1][1];
		rn_z[i] = 1;
	}


	////***		3.定義矩陣內容		***//////////////////////////////////////////////
	#pragma omp parallel for  schedule(static)					// 平行處理
	for (int i = 0; i < num; i++) {
		A[i][0] = rn_x[i] * ln_x[i];		A[i][1] = rn_x[i] * ln_y[i];		A[i][2] = rn_x[i] * ln_z[i];
		A[i][3] = rn_y[i] * ln_x[i];		A[i][4] = rn_y[i] * ln_y[i];		A[i][5] = rn_y[i] * ln_z[i];
		A[i][6] = rn_z[i] * ln_x[i];		A[i][7] = rn_z[i] * ln_y[i];		A[i][8] = rn_z[i] * ln_z[i];
	}

	//***		SVD			***//
	int ret = Used_SVD(A, k, m, U, D, V);

	//***	  輸出F矩陣	  ***//
	#pragma omp parallel for  schedule(static)							// 平行處理
	for (int j = 0; j < n/3; j++) {
		F[0][j] = V[V.size() - 1][j] ;
		F[1][j] = V[V.size() - 1][j + 3] ;
		F[2][j] = V[V.size() - 1][j + 6] ;
	}

	//***	  F矩陣(比例)還原	  ***//
	for (int i = 0; i < F.size(); i++) {
		for (int j = 0; j < F[0].size(); j++) {
			#pragma omp parallel for  schedule(static)			
			for (int T = 0; T < 3; T++)
				F_hat[i][j] += r_T[i][T] * F[T][j];					// (r_T)旋轉 * F_hat  = F_hat2
		}
	}

	for (int i = 0; i < F.size(); i++) {
		for (int j = 0; j < F[0].size(); j++) {
			F[i][j] = 0;
			#pragma omp parallel for  schedule(static)				
			for (int T = 0; T < 3; T++)
				F[i][j] += F_hat[i][T] * l_T[T][j];					//  F_hat2  * l_T = F
		}
	}
	for (int i = 0; i < F.size(); i++) {
		#pragma omp parallel for  schedule(static)				

		for (int j = 0; j < F[0].size(); j++) {
			F[i][j] = F[i][j] / F[2][2];							//  F_hat2  * l_T = F
		}
	}
	////// 清除不必要的矩陣	
	A.clear();
	F_hat.clear();

	U.clear();
	D.clear();
	V.clear();


	////***		4.開啟相機參數  Parameter.txt 		***//////////////////////////////////////////////
	ifstream in("Parameter.txt");			//打開文件
	string inputStr;
	vector<string> inputContent;			//每一行讀取  共N行
	while (getline(in, inputStr))			//讀取txt
		inputContent.push_back(inputStr);	//以行為單位
	in.close();								//關閉文件

	////***		讀取相機矩陣		***////
	Instrisic_L.resize(3);					// 3 X 3
	for (int i = 0; i < 3; i++)
		Instrisic_L[i].resize(3, 0);

	Instrisic_R.resize(3);					// 3 X 3
	for (int i = 0; i < 3; i++)
		Instrisic_R[i].resize(3, 0);

	Extrinsic_L.resize(3);					// 3 X 4
	for (int i = 0; i < 3; i++)
		Extrinsic_L[i].resize(4, 0);

	Extrinsic_R.resize(3);					// 3 X 4
	for (int i = 0; i < 3; i++)
		Extrinsic_R[i].resize(4, 0);

	P_L.resize(3);							// 3 X 4
	for (int i = 0; i < 3; i++)
		P_L[i].resize(4, 0);

	P_R.resize(3);							// 3 X 4
	for (int i = 0; i < 3; i++)
		P_R[i].resize(4, 0);



	////***		5.開啟Parameter.txt 		***//////////////////////////////////////////////
	double scan_num = 0;													//存入的double
	for (int i = 0; i < inputContent.size(); i++) {
		//cout << inputContent[i] << endl;									//印出String所有狀態
		char * cstr=0, * p=0;												//分割前 分割後
		if (inputContent[i] == "#Left Camera Intrinsic parameter") {		//前一行
			#pragma omp parallel for  schedule(static)						//平行處理
			for (int j = 0; j < 3; j++) {									//ROW
				cstr = new char[inputContent[i + 1 + j].length() + 1];		//分割前的char存在cstr
				std::strcpy(cstr, inputContent[i + 1 + j].c_str());			//分割
				p = std::strtok(cstr, " ");									//分割後的char存在p
				int count_P = 0;											//分割的次數
				while (p != 0)												//Column
				{
					scan_num = std::stod(p);			//String 轉換 double
					Instrisic_L[j][count_P] = scan_num;	//存入
					//std::cout << p << std::endl;	
					p = std::strtok(NULL, " ");			//結尾
					count_P++;
				}
			}
		}
		else if (inputContent[i] == "#Right Camera Intrinsic parameter") {
			#pragma omp parallel for  schedule(static)						
			for (int j = 0; j < 3; j++) {									
				cstr = new char[inputContent[i + 1 + j].length() + 1];	
				std::strcpy(cstr, inputContent[i + 1 + j].c_str());		
				p = std::strtok(cstr, " ");								
				int count_P = 0;											
				while (p != 0)										
				{
					scan_num = std::stod(p);			
					Instrisic_R[j][count_P] = scan_num;	
					p = std::strtok(NULL, " ");			
					count_P++;
				}
			}
		}
		else if (inputContent[i] == "#Left Camera Extrinsic parameter") {
			#pragma omp parallel for  schedule(static)					
			for (int j = 0; j < 3; j++) {								
				cstr = new char[inputContent[i + 1 + j].length() + 1];	
				std::strcpy(cstr, inputContent[i + 1 + j].c_str());		
				p = std::strtok(cstr, " ");								
				int count_P = 0;										
				while (p != 0)											
				{
					scan_num = std::stod(p);			
					Extrinsic_L[j][count_P] = scan_num;
					p = std::strtok(NULL, " ");			
					count_P++;
				}
			}
		}
		else if (inputContent[i] == "#Right Camera Extrinsic parameter") {
			#pragma omp parallel for  schedule(static)					
			for (int j = 0; j < 3; j++) {								
				cstr = new char[inputContent[i + 1 + j].length() + 1];		
				std::strcpy(cstr, inputContent[i + 1 + j].c_str());			
				p = std::strtok(cstr, " ");									
				int count_P = 0;									
				while (p != 0)											
				{
					scan_num = atof(p);					
					Extrinsic_R[j][count_P] = scan_num;					
					p = std::strtok(NULL, " ");			
					count_P++;
				}
			}
		}
		delete[] cstr, p;
	}

	////***		6.  P矩陣 		***//////////////////////////////////////////////
	for (int i = 0; i < 3; i++) {
	#pragma omp parallel for  schedule(static)							//平行處理
		for (int j = 0; j < 4; j++) {
			for (int T = 0; T < 3; T++) {
				P_L[i][j] += Instrisic_L[i][T] * Extrinsic_L[T][j]; // 矩陣相乘(H1要轉至)	
				P_R[i][j] += Instrisic_R[i][T] * Extrinsic_R[T][j]; // 矩陣相乘(H1要轉至)	
			}
		}
	}
	Instrisic_L.clear(); 
	Instrisic_R.clear(); 
	Extrinsic_L.clear();
	Extrinsic_R.clear();


	std::cout.precision(16);					//cout顯示完整的double
	std::cout << std::endl;
	std::cout << "F=" << std::endl;
	for (int i = 0; i < F.size(); i++) {
		for (int j = 0; j < F[0].size(); j++) {
			std::cout << setw(21) << F[i][j] << ' ';
		}
		std::cout << std::endl;
	}

	std::cout << "P_R=" << std::endl;
	for (int i = 0; i < P_R.size(); i++) {
		for (int j = 0; j < P_R[0].size(); j++) {
			std::cout << setw(21) << P_R[i][j] << ' ';
		}
		std::cout << std::endl;
	}


					
	////***		7.主程式 		***//////////////////////////////////////////////


	//打開要輸出openfile 的3D圖
	file_xyz.open(file_name, ios::out);
	L_scan_color = imread("L/L_Color.JPG", CV_LOAD_IMAGE_COLOR);		//彩色圖案
	R_scan_color = imread("R/R_Color.JPG", CV_LOAD_IMAGE_COLOR);

	for (int w = 0; w <= 292; w++) {					//圖片張數
		std::cout << w <<endl;
		//file_xyz << "-------------------------" << w<< std::endl;

		//////  開啟檔案
		sprintf(L_image_path, "L/L%03d.jpg", w);		//int to string   左圖
		L_scan =imread(L_image_path, CV_LOAD_IMAGE_COLOR);
		cv::cvtColor(L_scan, L_scan, CV_BGR2GRAY);		//轉灰階
		sprintf(R_image_path, "R/R%03d.jpg", w);		//int to string   右圖
		R_scan = imread(R_image_path, CV_LOAD_IMAGE_COLOR);
		cv::cvtColor(R_scan, R_scan, CV_BGR2GRAY);

		////// 初始化 最大值最小值
		L_x_max.clear();
		L_y_max.clear();
		L_val.clear();
		R_x_max.clear();
		R_y_max.clear();
		R_val.clear();
		X_3D.clear();

		L_x_max.resize(L_scan.rows, 0);
		L_y_max.resize(L_scan.rows, 0);
		L_val.resize(L_scan.rows, 0);
		R_x_max.resize(R_scan.rows, 0);					
		R_y_max.resize(R_scan.rows, 0);
		R_val.resize(R_scan.rows, 0);
		X_3D.resize(X_lenght, 0);
		////// 主程式
		for (int j = 0; j < L_scan.rows; j++) {				//左圖片row
			#pragma omp parallel for  schedule(static)		//平行處理
			for (int T = 0; T <= L_scan.cols; T++) {		//左圖片col
					////// 取最大值
					if (L_val[j] < L_scan.at<uchar>(j, T)) {
						L_x_max[j] = T;						//左圖片最亮點的X
						L_y_max[j] = j;						//左圖片最亮點的Y	
						L_val[j] = L_scan.at<uchar>(j, T);	//左圖片最亮點的數值
					}
			}

			if (L_val[j] > light_gate) {					//左圖片亮度閥值 >75
				////// 定義I_R矩陣內容
				#pragma omp parallel for  schedule(static)		
				for (int T = 0; T < F.size(); T++) {
					I_R[T] = F[T][0] * L_x_max[j] + F[T][1] * L_y_max[j] + F[T][2];
				}

				////// 掃描R圖片
				#pragma omp parallel for  schedule(static)		
				for (int T = 0; T < L_scan.rows; T++) {
					Ry = T;												//右邊的  Y
					Rx = - (Ry* I_R[1] + I_R[2]) / I_R[0];				//解聯立  X * I_R[0] +  Y * I_R[1] + I_R[2]=0
					//std::cout << Rx << "   " << Ry << std::endl << std::endl;

					if (Rx >= 0 && Rx < L_scan.cols) {					//確認範圍
						////// 取最大值
						if (R_val[T] < R_scan.at<uchar>(Ry, Rx)) {	
							R_val[T] = R_scan.at<uchar>(Ry, Rx);		//右圖片最亮點的數值
							R_x_max[T] = Rx;						//右圖片最亮點的X
							R_y_max[T] = Ry;						//右圖片最亮點的Y
						}
					}
				}

				if (R_val[j] > light_gate) {							//符合閥值的部分

					//***		定義3D矩陣 B			***//

					for (int c = 0; c < X_lenght; c++) {
						B[0][c] = L_x_max[j] * P_L[2][c] - P_L[0][c];
						B[1][c] = L_y_max[j] * P_L[2][c] - P_L[1][c];
						B[2][c] = R_x_max[j] * P_R[2][c] - P_R[0][c];
						B[3][c] = R_y_max[j] * P_R[2][c] - P_R[1][c];
					}



					//***		SVD			***//
					int ret = Used_SVD(B, X_lenght, X_lenght, U2, D2, V2);

					//***	  輸出3D矩陣	  ***//
					for (int CC = 0; CC < X_lenght; CC++) {
						X_3D[CC] = V2[V2.size() - 1][CC];
					}

					//***	  正規化	  ***//
					for (int CC = 0; CC < X_lenght; CC++) {
						X_3D[CC] = X_3D[CC]/ X_3D[X_3D.size() - 1];
					}

					r_color = 0, g_color = 0 ,b_color = 0;
					b_color = L_scan_color.at<Vec3b>(L_y_max[j], L_x_max[j])[0];						//取顏色
					g_color = L_scan_color.at<Vec3b>(L_y_max[j], L_x_max[j])[1];
					r_color = L_scan_color.at<Vec3b>(L_y_max[j], L_x_max[j])[2];

					//去除掉重複的點
					if (x_3d != (int)X_3D[0] || y_3d != (int)X_3D[1] || z_3d != (int)X_3D[2]){
						x_3d = (int)X_3D[0];
						y_3d = (int)X_3D[1];
						z_3d = (int)X_3D[2];
						file_xyz << (int)X_3D[0] << "\t" << (int)X_3D[1] << "\t" << (int)X_3D[2] << "\t" << r_color <<  "\t" << g_color <<  "\t" << b_color << std::endl;

					}

				}
			}
		}
	}
	file_xyz.close();


	cvWaitKey(0);
	return 0;
}
