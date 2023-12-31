clc;clear;

%%
%8個點找F

%L
x1=[427,713,1]';
x2=[210,708,1]';
x3=[248,685,1]';
x4=[466,666,1]';
x5=[394,411,1]';
x6=[261,578,1]';
x7=[499,808,1]';
x8=[528,690,1]';
%R
xp1=[378,725,1]';
xp2=[204,716,1]';
xp3=[217,704,1]';
xp4=[435,677,1]';
xp5=[379,429,1]';
xp6=[218,594,1]';
xp7=[495,825,1]';
xp8=[536,708,1]';

A=[
xp1(1)*x1(1) xp1(1)*x1(2) xp1(1) x1(1)*xp1(2) x1(2)*xp1(2) xp1(2) x1(1) x1(2);
xp2(1)*x2(1) xp2(1)*x2(2) xp2(1) x2(1)*xp2(2) x2(2)*xp2(2) xp2(2) x2(1) x2(2);
xp3(1)*x3(1) xp3(1)*x3(2) xp3(1) x3(1)*xp3(2) x3(2)*xp3(2) xp3(2) x3(1) x3(2);
xp4(1)*x4(1) xp4(1)*x4(2) xp4(1) x4(1)*xp4(2) x4(2)*xp4(2) xp4(2) x4(1) x4(2);
xp5(1)*x5(1) xp5(1)*x5(2) xp5(1) x5(1)*xp5(2) x5(2)*xp5(2) xp5(2) x5(1) x5(2);
xp6(1)*x6(1) xp6(1)*x6(2) xp6(1) x6(1)*xp6(2) x6(2)*xp6(2) xp6(2) x6(1) x6(2);
xp7(1)*x7(1) xp7(1)*x7(2) xp7(1) x7(1)*xp7(2) x7(2)*xp7(2) xp7(2) x7(1) x7(2);
xp8(1)*x8(1) xp8(1)*x8(2) xp8(1) x8(1)*xp8(2) x8(2)*xp8(2) xp8(2) x8(1) x8(2)];
ans_=[
    -1;
    -1;
    -1;
    -1;
    -1;
    -1;
    -1;
    -1;
    ];

f=inv(A)*ans_;
% 得到F
F=[f(1) f(2) f(3);
   f(4) f(5) f(6);
   f(7) f(8)  1.0;
    ];

%%獲得左側與右側的P
k_left=[1035.278669095568 0.000000000000 295.500377771516;
   0.000000000000 1034.880664685675 598.224722223280;
   0.000000000000 0.000000000000 1.000000000000];
k_left_rt=[1.000000000000 0.000000000000 0.000000000000 0.000000000000;
    0.000000000000 1.000000000000 0.000000000000 0.000000000000;
    0.000000000000 0.000000000000 1.000000000000 0.000000000000;
];
p_left=k_left*k_left_rt;

k_right=[1036.770200759934 0.000000000000 403.040387412710;
   0.000000000000 1037.186415753241 612.775486819306;
   0.000000000000 0.000000000000 1.000000000000];
k_right_rt=[0.958173249509 0.009400631103 0.286034354684 -69.855978076557;
    -0.009648701145 0.999953303246 -0.000542119475 0.110435878514;
   -0.286026094074 -0.002240415626 0.958219209809 14.517584144224;
];
p_right=k_right*k_right_rt;

p1=p_left(1,1:4)';
p2=p_left(2,1:4)';
p3=p_left(3,1:4)';

pp1=p_right(1,1:4)';
pp2=p_right(2,1:4)';
pp3=p_right(3,1:4)';



%%
%匯入預先抓取到的亮點TXT檔
SamplePath1 =  'L\';
SamplePath2 =  'R\';  
fileExt = '*.jpg'; 
%獲取所有檔案及其數量
files = dir(fullfile(SamplePath1,fileExt)); 
len1 = size(files,1);
%讀取左右側各自對應frame的亮點數
[data_col_number_L]=textread('L_col_number.txt','%n',293);
[data_col_number_R]=textread('R_col_number.txt','%n',293);
%開啟檔案以存取3D點
fid_get3d = fopen('3d_point.xyz','wt');

%抓取3D點
for i=1:len1
    txt_fileName_L =strcat(SamplePath1,mat2str(i),'.txt');
    [data1_L,data2_L]=textread(txt_fileName_L,'%n%n',data_col_number_L(i));
    data3_L=ones(data_col_number_L(i),1);
    data_L=[data1_L data2_L data3_L];
    for i2=1:data_col_number_L(i)
        x_left=[data_L(i2,1) data_L(i2,2) data_L(i2,3)];
        LP1=F*x_left';
        F
        

             txt_fileName_R =strcat(SamplePath2,mat2str(i),'.txt');
             [data1_R,data2_R]=textread(txt_fileName_R,'%n%n',data_col_number_R(i));
             data3_R=ones(data_col_number_R(i),1);
             data_R=[data1_R data2_R data3_R];
             store_min=ones(data_col_number_R(i),1);
             
              for i4=1:data_col_number_R(i)
                  
                  x_right=[data_R(i4,1) data_R(i4,2) data_R(i4,3)];
                  store_min(i4,1)=abs(x_right*LP1);
              end
              [m,p]=min(store_min);
              x_final_right=[data_R(p,1) data_R(p,2) data_R(p,3)];

                   u=x_left(1,1);
                   v=x_left(1,2);
                   up=x_final_right(1,1);
                   vp=x_final_right(1,2);
                   A=[u*p3'-p1';
                        v*p3'-p2';
                        up*pp3'-pp1';
                        vp*pp3'-pp2'];
                   [U,S,V]=svd(A);
                   X=V(1:4,4);
                   X=X/X(4,1);
                   X_final=X(1:3,1)';
                   x_result=[round(X_final(1,1)) round(X_final(1,2)) round(X_final(1,3))];
                   fprintf(fid_get3d,'%g\t',x_result);
                   fprintf(fid_get3d,'\n');

    end
end
fclose(fid_get3d);

mcc -m find_3dpoint.m