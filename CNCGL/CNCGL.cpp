#define _CRT_SECURE_NO_WARNINGS
//#include <opencv2/opencv.hpp> 
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <iostream> 
#include <time.h>
#include <omp.h>
#include <math.h>
#include <conio.h>
#include <Windows.h>
#include <string>
#include <vector>
#include <fstream>
#include <conio.h>
#include <ppl.h>
# include <cstdlib>


#define GL_GLEXT_PROTOTYPES
#include <GL/GL.h>
//#include <GL/gl.h>
//#include <GL/glu.h>
#include <GL/GLU.h>
//#include <glaux.h>

#pragma comment (lib, "opengl32.lib")

using namespace concurrency;
using namespace std;
using namespace cv;


// нули получаются в ходе развертывания пулинга, то есть мы записываем нули не во все клетки листимаге 
// и нужно перемешывать все картинкиперед обучение одной эпохи
// поискать еще где могут получаться нули все отлажывать

const int kolMergePoolLayers = 1;//кол-во повторяющихся слоев пулинга
const int amtBGRConvMaps = 16;// Кол-во каналов в первом сверточном слое 
const int imageSizeX = 120, imageSizeY = 120;//Размеры исходного изображения
const int kolConvLayers = 3;// Кол-во двумерных сверточных слоев сети
const int kolBGRConvMaps = 16;// Кол-во каналов в первом сверточном слое 
const int maskSize = 3;//Размер маски
const int maskSizeOnFirstLayer = 3;//Размер маски на первом слое
const int kolLayers = 4;//Кол-во слоев пулинга и свертки
const int kolPercNeurons[4] = { 256,128,64,16 };
const int percSizeY = 60, percSizeX = 60;//параметры масок для перехода от свёртки к персептрону
const int kolPoolLayers = 1;//Кол-во слое пулинга
const int kolTestImages = 0;//Кол=во тестовых изображений
const int kolLearningImages = 16;//Кол-во обучающих изображений 
const int kolAllImages = kolTestImages + kolLearningImages;//Кол-во изображений всего
const int flows = 15;//количество потоков, для разделения на разные ядра
const int sec = 1000, minute = 60 * sec, hour = 60 * minute, workingTime = 6 * hour;//Временные константы
const int ansSign = 1;// 1: expected - actual, -1: actual - expecte
const int ImageToEra = 1000;
const int L = 0, T = 1;
bool f12 = true;

float dy[3][3] = { {-1,-1,-1}, {0,0,0}, {1,1,1} };
float dx[3][3] = { {-1,0,1},{-1,0,1},{-1,0,1} };
float xD[120][120];
float yD[120][120];
float D[120][120];
float D1[120][120];
float DAngel[120][120];

long double Graph[15001];

int rec[1000000];

vector<int>kolLayerMaps = { 16,16,32 };//Кол-во карт признаков на слое сверточном
vector<int> layerMapsSize = { 120,120,60,60};//Размеры слоя

vector<int> forSizeKol = { 16,16,32,32 };
vector<int> kolConvMapsAtLayer = { 16,16,32};// Кол-во каналов на каждом двумерном сверточном слое(один ложный для выхода в полносвязный персептрон)
//vector<int> listNbMPool = { 1,3,6 };//Список номеров сливающих(дублирующих) слоев пулинга в нейронной сети

const long double e = 0.0000000001;//погрешность для сравнения дробных чисел
const long double maxWeight = 2000;//модуль максимального веса
const double minMask = 0.0000001;//модуль минимального веса маски 
const double maksError = 0.5;
const long double limit = 0.81;
const long double baseLearningRate = 0.0191;
const long double baseLearnPercRate = 0.6;//0.83;
//const long double baseLearnPercRate = 0.1;
long double PercLearningRate = baseLearnPercRate;
long double learningRate = baseLearningRate;//Шаг обучения
long double ansThreshold = 0.85;

const bool flagDebug = false;//флаг вывода дополнительной информации
const bool flagPrintMasks = false;//вывод масок 3Х3 
const bool flagLog = false;//флаг сохранения лога ответов нейронки
const bool flagLearning = true;//флаг обучения неронки
vector<bool> blListLayers = { 1,0,1};//Список слоев: 1-сверточный слой, 0-пулинг (кол-во на 1 меньше, так как первый слой обрабатывается отдельно)
vector<int> PovtorLayers = {2};
//ofstream logEras, logAllInf;//файлы логов
vector<int>ans(4);

vector<vector<int>> answers(2, vector<int>(kolLearningImages, 0));
vector<int> orderImage(kolLearningImages, 0);
vector<vector<vector<vector<long double>>>> listImage(kolLayers);//Текущее изображение

vector<vector<vector<long double>>> listStartMasks(kolBGRConvMaps, vector<vector<long double>>(maskSizeOnFirstLayer, vector<long double>(maskSizeOnFirstLayer)));//Маски для исходного изображения
vector<vector<vector<vector<long double>>>> listMasks(kolConvLayers);//Маски для двумерных сверточных слоев
vector<vector<vector<long double>>> toPerceptronMasks(kolPercNeurons[0], vector<vector<long double>>(percSizeY, vector<long double>(percSizeX)));//масски перехода в персептрон
vector<vector<vector<long double>>> percWeights(3);//Веса переходов между слоями в персептроне 

vector<long double> neuralOut(kolPercNeurons[2]);//Результат работы нейронной сети 
vector<vector<long double>> percLvls(4);//Персептрон
vector<vector<long double>> percErrors(4, vector<long double>(kolPercNeurons[0]));//ошибки для данного слоя перцептрона
vector<vector<long double>> percExitErrors(percSizeY, vector<long double>(percSizeX));//Матрица ошибок для выхода из персептрона

vector<vector<long double>> listStartImage(120, vector<long double>(120));//Исходные изображения в трехмерной матрице


vector<vector<vector<vector<long double>>>> listPoolErrors(kolMergePoolLayers);

vector<vector<long double>> trueAns(kolPercNeurons[2], vector<long double>(1));

vector<vector<vector<long double>>> deltas(3, vector<vector<long double>>(kolPercNeurons[0], vector<long double>(kolPercNeurons[1])));

vector<vector<long double>> deltasBias(3, vector<long double>(kolPercNeurons[1]));

vector<vector<long double>> weightBias(3,vector<long double>(128));
vector<vector<long double>> weightBias1(3, vector<long double>(128));
vector<vector<long double>> weightBias2(3, vector<long double>(128));
vector<vector<long double>> weightBias3(3, vector<long double>(128));

vector<vector<int>> answers1(2, vector<int>(kolLearningImages, 0));
vector<vector<vector<vector<long double>>>> listImage1(kolLayers);//Текущее изображение

vector<vector<vector<long double>>> listStartMasks1(kolBGRConvMaps, vector<vector<long double>>(maskSizeOnFirstLayer, vector<long double>(maskSizeOnFirstLayer)));//Маски для исходного изображения
vector<vector<vector<vector<long double>>>> listMasks1(kolConvLayers);//Маски для двумерных сверточных слоев
vector<vector<vector<long double>>> toPerceptronMasks1(kolPercNeurons[0], vector<vector<long double>>(percSizeY, vector<long double>(percSizeX)));//масски перехода в персептрон
vector<vector<vector<long double>>> percWeights1(3);//Веса переходов между слоями в персептроне 

vector<long double> neuralOut1(kolPercNeurons[2]);//Результат работы нейронной сети 
vector<vector<long double>> percLvls1(4);//Персептрон
vector<vector<long double>> percErrors1(4, vector<long double>(kolPercNeurons[0]));//ошибки для данного слоя перцептрона
vector<vector<long double>> percExitErrors1(percSizeY, vector<long double>(percSizeX));//Матрица ошибок для выхода из персептрона

vector<vector<long double>> listStartImage1(120, vector<long double>(120));//Исходные изображения в трехмерной матрице

vector<vector<vector<vector<long double>>>> listImage2(kolLayers);//Текущее изображение
vector<vector<vector<vector<long double>>>> listImage3(kolLayers);//Текущее изображение

vector<vector<long double>> percLvls2(4);//Персептрон
vector<vector<long double>> percLvls3(4);//Персептрон

vector<long double> neuralOut2(kolPercNeurons[2]);//Результат работы нейронной сети
vector<long double> neuralOut3(kolPercNeurons[2]);//Результат работы нейронной сети

vector<vector<long double>> listStartImage2(120, vector<long double>(120));//Исходные изображения в трехмерной матрице
vector<vector<long double>> listStartImage3(120, vector<long double>(120));//Исходные изображения в трехмерной матрице

vector<vector<vector<long double>>> listStartMasks2(kolBGRConvMaps, vector<vector<long double>>(maskSizeOnFirstLayer, vector<long double>(maskSizeOnFirstLayer)));//Маски для исходного изображения
vector<vector<vector<vector<long double>>>> listMasks2(kolConvLayers);//Маски для двумерных сверточных слоев
vector<vector<vector<long double>>> toPerceptronMasks2(kolPercNeurons[0], vector<vector<long double>>(percSizeY, vector<long double>(percSizeX)));//масски перехода в персептрон
vector<vector<vector<long double>>> percWeights2(3);//Веса переходов между слоями в персептроне

vector<vector<vector<long double>>> listStartMasks3(kolBGRConvMaps, vector<vector<long double>>(maskSizeOnFirstLayer, vector<long double>(maskSizeOnFirstLayer)));//Маски для исходного изображения
vector<vector<vector<vector<long double>>>> listMasks3(kolConvLayers);//Маски для двумерных сверточных слоев
vector<vector<vector<long double>>> toPerceptronMasks3(kolPercNeurons[0], vector<vector<long double>>(percSizeY, vector<long double>(percSizeX)));//масски перехода в персептрон
vector<vector<vector<long double>>> percWeights3(3);//Веса переходов между слоями в персептроне

vector<string> folderWeight = { "C:/Users/panih/source/repos/CNCGL/CNCGL/weight/","C:/Users/panih/source/repos/CNCGL/CNCGL/weight1/","C:/Users/panih/source/repos/CNCGL/CNCGL/weight2/","C:/Users/panih/source/repos/CNCGL/CNCGL/weight3/" };
vector<string> folderBias = { "C:/Users/panih/source/repos/CNCGL/CNCGL/bias/","C:/Users/panih/source/repos/CNCGL/CNCGL/bias1/","C:/Users/panih/source/repos/CNCGL/CNCGL/bias2/","C:/Users/panih/source/repos/CNCGL/CNCGL/bias3/" };
vector<string> folderMask = { "C:/Users/panih/source/repos/CNCGL/CNCGL/mask/","C:/Users/panih/source/repos/CNCGL/CNCGL/mask1/","C:/Users/panih/source/repos/CNCGL/CNCGL/mask2/","C:/Users/panih/source/repos/CNCGL/CNCGL/mask3/" };

vector<long long> myCount(16, 0);
bool showOnScreen = false;
string sRead; // kol-poz from file

vector<vector<vector<int>>> pozFill(32, vector<vector<int>>(22, vector<int>(2 , 0)));

vector<string> pozFilter = {"Стоя, руки,ноги вместе","Правая рука вправо на 45","Правая рука вправо на 90","Правая рука вправо 135","Правая рука вверх","Левая влево на 45","Левая влево на 90",
"Левая влево на 135","Левая вверх","Корпус на 90 влево","корпус на 90 вправо","Корпус влево на 90 руки вверх","Корпус вправо на 90 руки вверх","Руки на 90","Корпус на 90 вперед","Корпус на 90 назад",
"Руки вперед на 90","Руки вверх","руки вперед на 45"};

string printTime(int Timer);// Вывод времени

clock_t timer = clock();
int current;
unsigned int srandTemp = 10;
long double bestError = 100, sumPercDelta;
const string endBlock = "-------------------------------------------------------------/n";


HGLRC           hRC = NULL;
HDC             hDC = NULL;
HWND            hWnd = NULL;
HINSTANCE       hInstance;

bool    keys[256];
bool    active = TRUE;
bool    fullscreen = TRUE;

LPSTR b;
GLfloat     rtri;
GLfloat     rquad;
const GLfloat m[16] = { 1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1 };
GLfloat xx = 0.0f;
GLfloat yy = 0.5f;
GLfloat zz = -1.5f;

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

struct klxl //структура передачи
{
	int x, y;
};
struct klxl1
{
	GLfloat x, y, z;
};
klxl mk[13], mk0[13]; //2 массива передавания 
klxl1 mk1[13]; 
int iii;


void ImgFromCam() {
	VideoCapture cap(0);
	Mat frame;
	cap >> frame;
	for (int i = 0; i < frame.rows; i++) {
		for (int j = 0; j < frame.cols; j++) {
			//listStartImageBGR[0][i][j] = frame.at<Vec3b>(i, j)[0];
			//listStartImageBGR[1][i][j] = frame.at<Vec3b>(i, j)[1];
			//listStartImageBGR[2][i][j] = frame.at<Vec3b>(i, j)[2];
		}
	}
}

void LoadImg(int Num, int kol) {
	stringstream ss;
	stringstream ss1;
	ss << orderImage[Num - 1];
	ss1 << myCount[orderImage[Num - 1]];
	string name = ss1.str();
	string name1 = ss.str();
	//string name2 = "1";//ss.str();
	//name2 += ".jpg";
	name += ".jpg";
	//name1 += ".txt";
	//cout << name << " " << name1;	
	myCount[orderImage[Num - 1]]++;
	if (myCount[orderImage[Num - 1]] >= 1000) {
		myCount[orderImage[Num - 1]] = 0;
	}

	if (kol == 0) name = "tr5/" + name1 + "/" + name;
	else if (kol == 1) name = "tr5/" + name1 + "/" + name;
	else if (kol == 2) name = "tr6/" + name1 + "/" + name;
	else if (kol == 3) name = "tr6/" + name1 + "/" + name;
	//cout << name<<endl;
	//name1 = "txt2/" + name1;
	//name2 = "tr6/" + name2;
	Mat img = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
	//cout << name << endl;
	for (int iRead = 0; iRead < img.rows; iRead++) {
		for (int jRead = 0; jRead < img.cols; jRead++) {
			 listStartImage[iRead][jRead] = img.at<uchar>(iRead, jRead);
		}
	};
	img.release();
	for (int i = 0; i < 119; i++) {
		for (int j = 0; j < 119; j++) {
			for (int ii = -1; ii < 2; ii++) {
				for (int jj = -1; jj < 2; jj++) {
					if ((i + ii >= 0) && (j + jj >= 0)) {
						xD[i][j] += listStartImage[i + ii][j + jj] * dx[ii + 1][jj + 1];
						yD[i][j] += listStartImage[i + ii][j + jj] * dy[ii + 1][jj + 1];
					}
				}
			}
			D[i][j] = sqrt(pow(xD[i][j], 2) + pow(yD[i][j], 2));
			float pd = yD[i][j] / xD[i][j];
			DAngel[i][j] = atan(pd) + 3.14 / 2;
			xD[i][j] = 0; yD[i][j] = 0;
		}
	}
	for (int i = 0; i < 120; i++) {
		for (int j = 0; j < 120; j++) {
			if (DAngel[i][j] > 3.14) DAngel[i][j] -= 3.14 * 2;
			if (DAngel[i][j] <= 0.3839724 && DAngel[i][j] > -0.3839724) {
				if (D[i - 1][j] > D[i][j]) {
					D[i][j] = 0;
				}
				else if (D[i + 1][j] > D[i][j]) {
					D[i][j] = 0;
				}
			}//0
			if (DAngel[i][j] > 0.3839724 && DAngel[i][j] <= 1.169371) {
				if (i - 1 >= 0 && j + 1 < 120) {
					if (D[i - 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (j - 1 >= 0 && i + 1 < 120) {
					if (D[i + 1][j - 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//45
			if (DAngel[i][j] > 1.169371 && DAngel[i][j] <= 1.9547688) {
				if (j + 1 < 120) {
					if (D[i][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (j - 1 >= 0) {
					if (D[i][j - 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//90
			if (DAngel[i][j] > 1.9547688 && DAngel[i][j] <= 2.7401669) {
				if (i + 1 < 120 && j + 1 < 120) {
					if (D[i + 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i - 1 >= 0 && j - 1 >= 0) {
					if (D[i - 1][j - 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//135
			if (DAngel[i][j] > 2.7401669 && DAngel[i][j] <= 3.15) {
				if (i + 1 < 120) {
					if (D[i + 1][j] > D[i][j] || D[i - 1][j] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i - 1 >= 0) {
					if (D[i - 1][j] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//180

			if (DAngel[i][j] <= -0.3839724 && DAngel[i][j] > -1.169371) {
				if (i - 1 >= 0 && j - 1 >= 0) {
					if (D[i - 1][j - 1] > D[i][j] || D[i + 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i + 1 < 120 && j + 1 < 120) {
					if (D[i + 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//-45
			if (DAngel[i][j] <= -1.169371 && DAngel[i][j] > -1.9547688) {
				if (j - 1 >= 0) {
					if (D[i][j - 1] > D[i][j] || D[i][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (j + 1 < 120) {
					if (D[i][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//-90
			if (DAngel[i][j] <= -1.9547688 && DAngel[i][j] > -2.7401669) {
				if (i + 1 < 120 && j - 1 >= 0) {
					if (D[i + 1][j - 1] > D[i][j] || D[i - 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i - 1 >= 0 && j + 1 < 120) {
					if (D[i - 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//-135
			if (DAngel[i][j] <= -2.7401669 && DAngel[i][j] > -3.15) {
				if (i + 1 < 120) {
					if (D[i + 1][j] > D[i][j] || D[i - 1][j] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i - 1 >= 0) {
					if (D[i - 1][j] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//-180
			if (D[i][j] < 250) D[i][j] = 0;
			else if (D[i][j] > 300) D[i][j] = 255;
			else D[i][j] = 50;
		}
	}
	for (int i = 0; i < 120; i += 1) {
		for (int j = 0; j < 120; j += 1) {
			if (D[i][j] == 50) {
				if ((D[i - 1][j - 1] == 255) || (D[i - 1][j] == 255) || (D[i - 1][j + 1] == 255) || (D[i][j + 1] == 255) || (D[i + 1][j + 1] == 255) || (D[i + 1][j] == 255) || (D[i + 1][j - 1] == 255) || (D[i][j - 1] == 255))
				{
					D[i][j] = 255;
				}
				else D[i][j] = 0;

			}
		}
	}
	//Mat img1 = imread("tr1/1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < 120; i++) {
		for (int j = 0; j < 120; j++) {
			listStartImage[i][j] = D[i][j];
		//	img1.at<uchar>(i, j) = D[i][j];
			  //  cout << listStartImage[i][j];
		}
		//	cout << endl;
	}
	//cout << name2 << endl;
	//imwrite(name2, img1);
};

void handelVideo(int Num) {
	stringstream ss;
	ss << Num;
	string name = ss.str();//"C:/Project/Pic1.txt";
	string name1 = ss.str();//"C:/Project/Pic2.txt";//ss.str();
	name = "C:/Users/panih/source/repos/CNCGL/CNCGL/handel1/" + name+ ".jpg";
	name1 = "C:/Users/panih/source/repos/CNCGL/CNCGL/handel2/" + name1 + ".jpg";
	//cout << name << " " << name1;
	Mat img;
	Mat img1;
	img = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
	resize(img, img, Size(120, 120), 0, 0, INTER_CUBIC);
	img1 = imread(name1, CV_LOAD_IMAGE_GRAYSCALE);
	resize(img1, img1, Size(120, 120), 0, 0, INTER_CUBIC);
	for (int iRead = 0; iRead < img.rows; iRead++) {
		for (int jRead = 0; jRead < img.cols; jRead++) {
			listStartImage[iRead][jRead] = img.at<uchar>(iRead, jRead);
		}
	};
	img.release();
	//cout << name << " " << name1;
	for (int i = 0; i < 119; i++) {
		for (int j = 0; j < 119; j++) {
			for (int ii = -1; ii < 2; ii++) {
				for (int jj = -1; jj < 2; jj++) {
					if ((i + ii >= 0) && (j + jj >= 0)) {
						xD[i][j] += listStartImage[i + ii][j + jj] * dx[ii + 1][jj + 1];
						yD[i][j] += listStartImage[i + ii][j + jj] * dy[ii + 1][jj + 1];
					}
				}
			}
			D[i][j] = sqrt(pow(xD[i][j], 2) + pow(yD[i][j], 2));
			float pd = yD[i][j] / xD[i][j];
			DAngel[i][j] = atan(pd) + 3.14 / 2;
			xD[i][j] = 0; yD[i][j] = 0;
		}
	}
	for (int i = 0; i < 120; i++) {
		for (int j = 0; j < 120; j++) {
			if (DAngel[i][j] > 3.14) DAngel[i][j] -= 3.14 * 2;
			if (DAngel[i][j] <= 0.3839724 && DAngel[i][j] > -0.3839724) {
				if (D[i - 1][j] > D[i][j]) {
					D[i][j] = 0;
				}
				else if (D[i + 1][j] > D[i][j]) {
					D[i][j] = 0;
				}
			}//0
			if (DAngel[i][j] > 0.3839724 && DAngel[i][j] <= 1.169371) {
				if (i - 1 >= 0 && j + 1 < 120) {
					if (D[i - 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (j - 1 >= 0 && i + 1 < 120) {
					if (D[i + 1][j - 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//45
			if (DAngel[i][j] > 1.169371 && DAngel[i][j] <= 1.9547688) {
				if (j + 1 < 120) {
					if (D[i][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (j - 1 >= 0) {
					if (D[i][j - 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//90
			if (DAngel[i][j] > 1.9547688 && DAngel[i][j] <= 2.7401669) {
				if (i + 1 < 120 && j + 1 < 120) {
					if (D[i + 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i - 1 >= 0 && j - 1 >= 0) {
					if (D[i - 1][j - 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//135
			if (DAngel[i][j] > 2.7401669 && DAngel[i][j] <= 3.15) {
				if (i + 1 < 120) {
					if (D[i + 1][j] > D[i][j] || D[i - 1][j] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i - 1 >= 0) {
					if (D[i - 1][j] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//180

			if (DAngel[i][j] <= -0.3839724 && DAngel[i][j] > -1.169371) {
				if (i - 1 >= 0 && j - 1 >= 0) {
					if (D[i - 1][j - 1] > D[i][j] || D[i + 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i + 1 < 120 && j + 1 < 120) {
					if (D[i + 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//-45
			if (DAngel[i][j] <= -1.169371 && DAngel[i][j] > -1.9547688) {
				if (j - 1 >= 0) {
					if (D[i][j - 1] > D[i][j] || D[i][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (j + 1 < 120) {
					if (D[i][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//-90
			if (DAngel[i][j] <= -1.9547688 && DAngel[i][j] > -2.7401669) {
				if (i + 1 < 120 && j - 1 >= 0) {
					if (D[i + 1][j - 1] > D[i][j] || D[i - 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i - 1 >= 0 && j + 1 < 120) {
					if (D[i - 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//-135
			if (DAngel[i][j] <= -2.7401669 && DAngel[i][j] > -3.15) {
				if (i + 1 < 120) {
					if (D[i + 1][j] > D[i][j] || D[i - 1][j] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i - 1 >= 0) {
					if (D[i - 1][j] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//-180
			if (D[i][j] < 250) D[i][j] = 0;
			else if (D[i][j] > 300) D[i][j] = 255;
			else D[i][j] = 50;
		}
	}
	for (int i = 0; i < 120; i += 1) {
		for (int j = 0; j < 120; j += 1) {
			if (D[i][j] == 50) {
				if ((D[i - 1][j - 1] == 255) || (D[i - 1][j] == 255) || (D[i - 1][j + 1] == 255) || (D[i][j + 1] == 255) || (D[i + 1][j + 1] == 255) || (D[i + 1][j] == 255) || (D[i + 1][j - 1] == 255) || (D[i][j - 1] == 255))
				{
					D[i][j] = 255;
				}
				else D[i][j] = 0;

			}
		}
	}
	//	cout << name;
	Mat img4 = imread("C:/Users/panih/source/repos/CNCGL/CNCGL/tr1/1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < 120; i++) {
		for (int j = 0; j < 120; j++) {
			listStartImage[i][j] = D[i][j];
			listStartImage1[i][j] = D[i][j];
				img4.at<uchar>(i, j) = D[i][j];
		}
		//	cout << endl;
	}
	//cout << name2 << endl;
	imwrite("Pic6.jpg", img4);
	img4.release();

	for (int iRead = 0; iRead < img1.rows; iRead++) {
		for (int jRead = 0; jRead < img1.cols; jRead++) {
			listStartImage2[iRead][jRead] = img1.at<uchar>(iRead, jRead);
			xD[iRead][jRead] = 0; yD[iRead][jRead] = 0; D[iRead][jRead] = 0; D1[iRead][jRead] = 0; DAngel[iRead][jRead] = 0;
		}
	};
	img1.release();

	for (int i = 0; i < 119; i++) {
		for (int j = 0; j < 119; j++) {
			for (int ii = -1; ii < 2; ii++) {
				for (int jj = -1; jj < 2; jj++) {
					if ((i + ii >= 0) && (j + jj >= 0)) {
						xD[i][j] += listStartImage2[i + ii][j + jj] * dx[ii + 1][jj + 1];
						yD[i][j] += listStartImage2[i + ii][j + jj] * dy[ii + 1][jj + 1];
					}
				}
			}
			D[i][j] = sqrt(pow(xD[i][j], 2) + pow(yD[i][j], 2));
			float pd = yD[i][j] / xD[i][j];
			DAngel[i][j] = atan(pd) + 3.14 / 2;
			xD[i][j] = 0; yD[i][j] = 0;
		}
	}
	for (int i = 0; i < 120; i++) {
		for (int j = 0; j < 120; j++) {
			if (DAngel[i][j] > 3.14) DAngel[i][j] -= 3.14 * 2;
			if (DAngel[i][j] <= 0.3839724 && DAngel[i][j] > -0.3839724) {
				if (D[i - 1][j] > D[i][j]) {
					D[i][j] = 0;
				}
				else if (D[i + 1][j] > D[i][j]) {
					D[i][j] = 0;
				}
			}//0
			if (DAngel[i][j] > 0.3839724 && DAngel[i][j] <= 1.169371) {
				if (i - 1 >= 0 && j + 1 < 120) {
					if (D[i - 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (j - 1 >= 0 && i + 1 < 120) {
					if (D[i + 1][j - 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//45
			if (DAngel[i][j] > 1.169371 && DAngel[i][j] <= 1.9547688) {
				if (j + 1 < 120) {
					if (D[i][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (j - 1 >= 0) {
					if (D[i][j - 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//90
			if (DAngel[i][j] > 1.9547688 && DAngel[i][j] <= 2.7401669) {
				if (i + 1 < 120 && j + 1 < 120) {
					if (D[i + 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i - 1 >= 0 && j - 1 >= 0) {
					if (D[i - 1][j - 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//135
			if (DAngel[i][j] > 2.7401669 && DAngel[i][j] <= 3.15) {
				if (i + 1 < 120) {
					if (D[i + 1][j] > D[i][j] || D[i - 1][j] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i - 1 >= 0) {
					if (D[i - 1][j] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//180

			if (DAngel[i][j] <= -0.3839724 && DAngel[i][j] > -1.169371) {
				if (i - 1 >= 0 && j - 1 >= 0) {
					if (D[i - 1][j - 1] > D[i][j] || D[i + 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i + 1 < 120 && j + 1 < 120) {
					if (D[i + 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//-45
			if (DAngel[i][j] <= -1.169371 && DAngel[i][j] > -1.9547688) {
				if (j - 1 >= 0) {
					if (D[i][j - 1] > D[i][j] || D[i][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (j + 1 < 120) {
					if (D[i][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//-90
			if (DAngel[i][j] <= -1.9547688 && DAngel[i][j] > -2.7401669) {
				if (i + 1 < 120 && j - 1 >= 0) {
					if (D[i + 1][j - 1] > D[i][j] || D[i - 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i - 1 >= 0 && j + 1 < 120) {
					if (D[i - 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//-135
			if (DAngel[i][j] <= -2.7401669 && DAngel[i][j] > -3.15) {
				if (i + 1 < 120) {
					if (D[i + 1][j] > D[i][j] || D[i - 1][j] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i - 1 >= 0) {
					if (D[i - 1][j] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//-180
			if (D[i][j] < 250) D[i][j] = 0;
			else if (D[i][j] > 300) D[i][j] = 255;
			else D[i][j] = 50;
		}
	}
	for (int i = 0; i < 120; i += 1) {
		for (int j = 0; j < 120; j += 1) {
			if (D[i][j] == 50) {
				if ((D[i - 1][j - 1] == 255) || (D[i - 1][j] == 255) || (D[i - 1][j + 1] == 255) || (D[i][j + 1] == 255) || (D[i + 1][j + 1] == 255) || (D[i + 1][j] == 255) || (D[i + 1][j - 1] == 255) || (D[i][j - 1] == 255))
				{
					D[i][j] = 255;
				}
				else D[i][j] = 0;

			}
		}
	}
	Mat img2 = imread("C:/Users/panih/source/repos/CNCGL/CNCGL/tr1/1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < 120; i++) {
		for (int j = 0; j < 120; j++) {
			listStartImage2[i][j] = D[i][j];
			listStartImage3[i][j] = D[i][j];
				img2.at<uchar>(i, j) = D[i][j];
				//  cout << listStartImage[i][j];
		}
	}
	imwrite("pic3.jpg", img2);
	img2.release();
}

void loadImgDisp(int Num) {
	//stringstream ss;
	//ss << Num;
	string name = "C:/Project/Pic1.txt";
	string name1 = "C:/Project/Pic2.txt";//ss.str();
	Mat img;
	Mat img1;
	img = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
	resize(img, img, Size(120, 120), 0, 0, INTER_CUBIC);
	img1 = imread(name1, CV_LOAD_IMAGE_GRAYSCALE);
	resize(img1, img1, Size(120, 120), 0, 0, INTER_CUBIC);
	for (int iRead = 0; iRead < img.rows; iRead++) {
		for (int jRead = 0; jRead < img.cols; jRead++) {
			listStartImage[iRead][jRead] = img.at<uchar>(iRead, jRead);
		}
	};
	img.release();
	//cout << name << " " << name1;
	for (int i = 0; i < 119; i++) {
		for (int j = 0; j < 119; j++) {
			for (int ii = -1; ii < 2; ii++) {
				for (int jj = -1; jj < 2; jj++) {
					if ((i + ii >= 0) && (j + jj >= 0)) {
						xD[i][j] += listStartImage[i + ii][j + jj] * dx[ii + 1][jj + 1];
						yD[i][j] += listStartImage[i + ii][j + jj] * dy[ii + 1][jj + 1];
					}
				}
			}
			D[i][j] = sqrt(pow(xD[i][j], 2) + pow(yD[i][j], 2));
			float pd = yD[i][j] / xD[i][j];
			DAngel[i][j] = atan(pd) + 3.14 / 2;
			xD[i][j] = 0; yD[i][j] = 0;
		}
	}
	for (int i = 0; i < 120; i++) {
		for (int j = 0; j < 120; j++) {
			if (DAngel[i][j] > 3.14) DAngel[i][j] -= 3.14 * 2;
			if (DAngel[i][j] <= 0.3839724 && DAngel[i][j] > -0.3839724) {
				if (D[i - 1][j] > D[i][j]) {
					D[i][j] = 0;
				}
				else if (D[i + 1][j] > D[i][j]) {
					D[i][j] = 0;
				}
			}//0
			if (DAngel[i][j] > 0.3839724 && DAngel[i][j] <= 1.169371) {
				if (i - 1 >= 0 && j + 1 < 120) {
					if (D[i - 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (j - 1 >= 0 && i + 1 < 120) {
					if (D[i + 1][j - 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//45
			if (DAngel[i][j] > 1.169371 && DAngel[i][j] <= 1.9547688) {
				if (j + 1 < 120) {
					if (D[i][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (j - 1 >= 0) {
					if (D[i][j - 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//90
			if (DAngel[i][j] > 1.9547688 && DAngel[i][j] <= 2.7401669) {
				if (i + 1 < 120 && j + 1 < 120) {
					if (D[i + 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i - 1 >= 0 && j - 1 >= 0) {
					if (D[i - 1][j - 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//135
			if (DAngel[i][j] > 2.7401669 && DAngel[i][j] <= 3.15) {
				if (i + 1 < 120) {
					if (D[i + 1][j] > D[i][j] || D[i - 1][j] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i - 1 >= 0) {
					if (D[i - 1][j] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//180

			if (DAngel[i][j] <= -0.3839724 && DAngel[i][j] > -1.169371) {
				if (i - 1 >= 0 && j - 1 >= 0) {
					if (D[i - 1][j - 1] > D[i][j] || D[i + 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i + 1 < 120 && j + 1 < 120) {
					if (D[i + 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//-45
			if (DAngel[i][j] <= -1.169371 && DAngel[i][j] > -1.9547688) {
				if (j - 1 >= 0) {
					if (D[i][j - 1] > D[i][j] || D[i][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (j + 1 < 120) {
					if (D[i][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//-90
			if (DAngel[i][j] <= -1.9547688 && DAngel[i][j] > -2.7401669) {
				if (i + 1 < 120 && j - 1 >= 0) {
					if (D[i + 1][j - 1] > D[i][j] || D[i - 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i - 1 >= 0 && j + 1 < 120) {
					if (D[i - 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//-135
			if (DAngel[i][j] <= -2.7401669 && DAngel[i][j] > -3.15) {
				if (i + 1 < 120) {
					if (D[i + 1][j] > D[i][j] || D[i - 1][j] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i - 1 >= 0) {
					if (D[i - 1][j] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//-180
			if (D[i][j] < 250) D[i][j] = 0;
			else if (D[i][j] > 300) D[i][j] = 255;
			else D[i][j] = 50;
		}
	}
	for (int i = 0; i < 120; i += 1) {
		for (int j = 0; j < 120; j += 1) {
			if (D[i][j] == 50) {
				if ((D[i - 1][j - 1] == 255) || (D[i - 1][j] == 255) || (D[i - 1][j + 1] == 255) || (D[i][j + 1] == 255) || (D[i + 1][j + 1] == 255) || (D[i + 1][j] == 255) || (D[i + 1][j - 1] == 255) || (D[i][j - 1] == 255))
				{
					D[i][j] = 255;
				}
				else D[i][j] = 0;

			}
		}
	}
//	cout << name;
//	Mat img4 = imread("C:/Users/panih/source/repos/CNCGL/CNCGL/tr1/1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < 120; i++) {
		for (int j = 0; j < 120; j++) {
			listStartImage[i][j] = D[i][j];
			listStartImage1[i][j] = D[i][j];
		//	img4.at<uchar>(i, j) = D[i][j];
		}
		//	cout << endl;
	}
	//cout << name2 << endl;
//	imwrite("Pic6.jpg", img4);
//	img4.release();

	for (int iRead = 0; iRead < img1.rows; iRead++) {
		for (int jRead = 0; jRead < img1.cols; jRead++) {
			listStartImage2[iRead][jRead] = img1.at<uchar>(iRead, jRead);
			xD[iRead][jRead] = 0; yD[iRead][jRead] = 0; D[iRead][jRead] = 0; D1[iRead][jRead] = 0; DAngel[iRead][jRead] = 0;
		}
	};
	img1.release();

	for (int i = 0; i < 119; i++) {
		for (int j = 0; j < 119; j++) {
			for (int ii = -1; ii < 2; ii++) {
				for (int jj = -1; jj < 2; jj++) {
					if ((i + ii >= 0) && (j + jj >= 0)) {
						xD[i][j] += listStartImage2[i + ii][j + jj] * dx[ii + 1][jj + 1];
						yD[i][j] += listStartImage2[i + ii][j + jj] * dy[ii + 1][jj + 1];
					}
				}
			}
			D[i][j] = sqrt(pow(xD[i][j], 2) + pow(yD[i][j], 2));
			float pd = yD[i][j] / xD[i][j];
			DAngel[i][j] = atan(pd) + 3.14 / 2;
			xD[i][j] = 0; yD[i][j] = 0;
		}
	}
	for (int i = 0; i < 120; i++) {
		for (int j = 0; j < 120; j++) {
			if (DAngel[i][j] > 3.14) DAngel[i][j] -= 3.14 * 2;
			if (DAngel[i][j] <= 0.3839724 && DAngel[i][j] > -0.3839724) {
				if (D[i - 1][j] > D[i][j]) {
					D[i][j] = 0;
				}
				else if (D[i + 1][j] > D[i][j]) {
					D[i][j] = 0;
				}
			}//0
			if (DAngel[i][j] > 0.3839724 && DAngel[i][j] <= 1.169371) {
				if (i - 1 >= 0 && j + 1 < 120) {
					if (D[i - 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (j - 1 >= 0 && i + 1 < 120) {
					if (D[i + 1][j - 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//45
			if (DAngel[i][j] > 1.169371 && DAngel[i][j] <= 1.9547688) {
				if (j + 1 < 120) {
					if (D[i][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (j - 1 >= 0) {
					if (D[i][j - 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//90
			if (DAngel[i][j] > 1.9547688 && DAngel[i][j] <= 2.7401669) {
				if (i + 1 < 120 && j + 1 < 120) {
					if (D[i + 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i - 1 >= 0 && j - 1 >= 0) {
					if (D[i - 1][j - 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//135
			if (DAngel[i][j] > 2.7401669 && DAngel[i][j] <= 3.15) {
				if (i + 1 < 120) {
					if (D[i + 1][j] > D[i][j] || D[i - 1][j] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i - 1 >= 0) {
					if (D[i - 1][j] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//180

			if (DAngel[i][j] <= -0.3839724 && DAngel[i][j] > -1.169371) {
				if (i - 1 >= 0 && j - 1 >= 0) {
					if (D[i - 1][j - 1] > D[i][j] || D[i + 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i + 1 < 120 && j + 1 < 120) {
					if (D[i + 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//-45
			if (DAngel[i][j] <= -1.169371 && DAngel[i][j] > -1.9547688) {
				if (j - 1 >= 0) {
					if (D[i][j - 1] > D[i][j] || D[i][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (j + 1 < 120) {
					if (D[i][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//-90
			if (DAngel[i][j] <= -1.9547688 && DAngel[i][j] > -2.7401669) {
				if (i + 1 < 120 && j - 1 >= 0) {
					if (D[i + 1][j - 1] > D[i][j] || D[i - 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i - 1 >= 0 && j + 1 < 120) {
					if (D[i - 1][j + 1] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//-135
			if (DAngel[i][j] <= -2.7401669 && DAngel[i][j] > -3.15) {
				if (i + 1 < 120) {
					if (D[i + 1][j] > D[i][j] || D[i - 1][j] > D[i][j]) {
						D[i][j] = 0;
					}
				}
				else if (i - 1 >= 0) {
					if (D[i - 1][j] > D[i][j]) {
						D[i][j] = 0;
					}
				}
			}//-180
			if (D[i][j] < 250) D[i][j] = 0;
			else if (D[i][j] > 300) D[i][j] = 255;
			else D[i][j] = 50;
		}
	}
	for (int i = 0; i < 120; i += 1) {
		for (int j = 0; j < 120; j += 1) {
			if (D[i][j] == 50) {
				if ((D[i - 1][j - 1] == 255) || (D[i - 1][j] == 255) || (D[i - 1][j + 1] == 255) || (D[i][j + 1] == 255) || (D[i + 1][j + 1] == 255) || (D[i + 1][j] == 255) || (D[i + 1][j - 1] == 255) || (D[i][j - 1] == 255))
				{
					D[i][j] = 255;
				}
				else D[i][j] = 0;

			}
		}
	}
	//Mat img2 = imread("C:/Users/panih/source/repos/CNCGL/CNCGL/tr1/1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < 120; i++) {
		for (int j = 0; j < 120; j++) {
			listStartImage2[i][j] = D[i][j];
			listStartImage3[i][j] = D[i][j];
		//	img2.at<uchar>(i, j) = D[i][j];
			//  cout << listStartImage[i][j];
		}
	}
	//imwrite("pic3.jpg", img2);
	//img2.release();
};

void upLoadWeight(int Num, int koli, int kolj, string folderName) {
	stringstream ss;
	ss << Num;
	string name = ss.str();
	name += ".txt";
	name = folderName + name;
	ofstream file1(name);
	for (int iL = 0; iL < koli; iL++) {
		for (int jL = 0; jL < kolj; jL++) {
			file1 << percWeights[Num][iL][jL] << " ";
		}
		file1 << "\n";
	}
	file1.close();
}

void loadWeight(int Num, int koli, int kolj, vector<vector<vector<long double>>>& percWeights, string na) {
	stringstream ss;
	ss << Num;
	string name = ss.str();
	name += ".txt";
	name = na + name;
	ifstream file1(name);
	for (int iL = 0; iL < koli; iL++) {
		for (int jL = 0; jL < kolj; jL++) {
			file1 >> percWeights[Num][iL][jL];
		}
	}
	file1.close();
}

void loadBias(int Num, int koli, vector<vector<long double>>& weightBias, string na) {
	stringstream ss;
	ss << Num;
	string name = ss.str();
	name += ".txt";
	name = na + name;
	//cout << name;
	ifstream file1(name);
	for (int iL = 0; iL < koli; iL++) {
		//long float khf;
		//for (int jL = 0; jL < kolj; jL++) {
		file1 >> weightBias[Num][iL];
		//cout << weightBias[Num][iL] << " ";
		//}
	}
	file1.close();
}

void upLoadBias(int Num, int koli, string folderName) {
	stringstream ss;
	ss << Num;
	string name = ss.str();
	name += ".txt";
	name = folderName + name;
	ofstream file1(name);
	for (int iL = 0; iL < koli; iL++) {
		//for (int jL = 0; jL < kolj; jL++) {
		file1 << weightBias[Num][iL] << " ";
		//}
		//file1 << "\n";
	}
	file1.close();
}

void upLoadMask(string folderName) {
	string name2 = "BGRMask";
	name2 += ".txt";
	name2 = folderName + name2;
	ofstream file(name2);
	for (int n = 0; n < kolBGRConvMaps; n++) {
			for (int iL = 0; iL < 3; iL++) {
				for (int jL = 0; jL < 3; jL++) {
					file << listStartMasks[n][iL][jL] << " ";
				}
				file << "\n";
			}
			file << "\n";
	}
	file.close();
	for (int n = 0; n < kolConvLayers; n++) {
		stringstream ss;
		ss << n;
		string name = ss.str();
		name += ".txt";
		name = "mask/" + name;
		ofstream file1(name);
		for (int ij = 0; ij < listMasks[n].size(); ij++) {
			for (int iL = 0; iL < 3; iL++) {
				for (int jL = 0; jL < 3; jL++) {
					file1 << listMasks[n][ij][iL][jL] << " ";
					//if (n == 1 && ij == 5) cout << listMasks[n][ij][iL][jL] << " ";
				}
				file1 << "\n";
			}
			file1 << "\n";
		}
		file1.close();
	}
	string name1 = "konvMask";
	name1 += ".txt";
	name1= folderName + name1;
	ofstream file2(name1);
	for (int n = 0; n < kolPercNeurons[0]; n++) {
		for (int iL = 0; iL < percSizeX; iL++) {
			for (int jL = 0; jL < percSizeY; jL++) {
				file2 << toPerceptronMasks[n][iL][jL] << " ";
			}
			file2 << "\n";
	    }
		file2 << "\n";
	}
	file2.close();
}

void loadMask(vector<vector<vector<long double>>>& listStartMasks, vector<vector<vector<vector<long double>>>>& listMasks, vector<vector<vector<long double>>>& toPerceptronMasks, string na) {
	string name2 = "BGRMask";
	name2 += ".txt";
	name2 = na + name2;
	ifstream file1(name2);
	for (int n = 0; n < kolBGRConvMaps; n++) {
			for (int iL = 0; iL < 3; iL++) {
				for (int jL = 0; jL < 3; jL++) {
					file1 >> listStartMasks[n][iL][jL];
				}
			}
	}
	file1.close();
	for (int n = 0; n < kolConvLayers; n++) {
		stringstream ss;
		ss << n;
		string name = ss.str();
		name += ".txt";
		name = na + name;
		ifstream file(name);
		for (int ij = 0; ij < listMasks[n].size(); ij++) {
			for (int iL = 0; iL < 3; iL++) {
				for (int jL = 0; jL < 3; jL++) {
					file >> listMasks[n][ij][iL][jL];
				}
			}
		}
		file.close();
	}
	string name1 = "konvMask";
	name1 += ".txt";
	name1 = na + name1;
	ifstream file2(name1);
	for (int n = 0; n < kolPercNeurons[0]; n++) {
		for (int iL = 0; iL < percSizeX; iL++) {
			for (int jL = 0; jL < percSizeY; jL++) {
				file2 >> toPerceptronMasks[n][iL][jL];
			}
		}
	}
	file2.close();
}

long double activateFunction(long double x) {
	return 1.0 - 2.0 / (exp(2 * x) + 1);
}

long double activateFunctionDX(long double x) {
	return 1 - x * x;
}

long double sigm(long double x) {
	return 1 / (1 + exp(-x));
}

long double sigmDX(long double x) {
	return x * (1 - x);
}

void matrActivateFunction(vector<vector<vector<long double>>>& listImage,int suc)
{
	int mq = listImage.size();
	int mi = listImage[0].size();
	int mj = listImage[0][0].size();
	parallel_for(int(0), mq, [&](int q) {
		for (int i = 0; i < mi; i++) {
			for (int j = 0; j < mj; j++) {
				listImage[q][i][j] = activateFunction(listImage[q][i][j]);
			}
		}
	});
}

long double pressFunction(long double a, long double b, long double c, long double d) {
	return max(max(a, b), max(c, d));
}

long double pressFunction1(long double a, long double b, long double c, long double d, long double e, long double f, long double j12, long double h, long double k) {
	return max(max(a, b), max(max(max(c, d), max(e, f)),max(max(j12, h), k)));
}

void makeFirstLayer(vector<vector<vector<long double>>> &listImage, vector<vector<vector<long double>>>& Masks, vector<vector<long double>>& startImage)
{
	int mx = listImage.size();
	parallel_for(int(0), mx, [&](int sl) {
	//	for (int sl = 0; sl < mx; sl++){
			//Обрабодка углов изображения
		listImage[sl][0][0] += startImage[0][0] * Masks[sl][0][0] + startImage[0][0] * Masks[sl][0][1] + startImage[0][1] * Masks[sl][0][2] +
			startImage[0][0] * Masks[sl][1][0] + startImage[0][0] * Masks[sl][1][1] + startImage[0][1] * Masks[sl][1][2] +
			startImage[1][0] * Masks[sl][2][0] + startImage[1][0] * Masks[sl][2][1] + startImage[1][1] * Masks[sl][2][2];
		listImage[sl][0][imageSizeX - 1] += startImage[0][imageSizeX - 2] * Masks[sl][0][0] + startImage[0][imageSizeX - 1] * Masks[sl][0][1] + startImage[0][imageSizeX - 1] * Masks[sl][0][2] +
			startImage[0][imageSizeX - 2] * Masks[sl][1][0] + startImage[0][imageSizeX - 1] * Masks[sl][1][1] + startImage[0][imageSizeX - 1] * Masks[sl][1][2] +
			startImage[1][imageSizeX - 2] * Masks[sl][2][0] + startImage[1][imageSizeX - 1] * Masks[sl][2][1] + startImage[1][imageSizeX - 1] * Masks[sl][2][2];
		listImage[sl][imageSizeY - 1][0] += startImage[imageSizeY - 2][0] * Masks[sl][0][0] + startImage[imageSizeY - 2][0] * Masks[sl][0][1] + startImage[imageSizeY - 2][1] * Masks[sl][0][2] +
			startImage[imageSizeY - 1][0] * Masks[sl][1][0] + startImage[imageSizeY - 1][0] * Masks[sl][1][1] + startImage[imageSizeY - 1][1] * Masks[sl][1][2] +
			startImage[imageSizeY - 1][0] * Masks[sl][2][0] + startImage[imageSizeY - 1][0] * Masks[sl][2][1] + startImage[imageSizeY - 1][1] * Masks[sl][2][2];
		listImage[sl][imageSizeY - 1][imageSizeX - 1] += startImage[imageSizeY - 2][imageSizeX - 2] * Masks[sl][0][0] + startImage[imageSizeY - 2][imageSizeX - 1] * Masks[sl][0][1]
			+ startImage[imageSizeY - 2][imageSizeX - 1] * Masks[sl][0][2] +
			startImage[imageSizeY - 1][imageSizeX - 2] * Masks[sl][1][0] + startImage[imageSizeY - 1][imageSizeX - 1] * Masks[sl][1][1]
			+ startImage[imageSizeY - 1][imageSizeX - 1] * Masks[sl][1][2] +
			startImage[imageSizeY - 1][imageSizeX - 2] * Masks[sl][2][0] + startImage[imageSizeY - 1][imageSizeX - 1] * Masks[sl][2][1]
			+ startImage[imageSizeY - 1][imageSizeX - 1] * Masks[sl][2][2];
		//Обрабока краев изображения
		for (int i = 1; i < imageSizeY - 1; i++)
			for (int j = -1; j < 2; j++)
				listImage[sl][i][0] += startImage[i + j][0] * (Masks[sl][j + 1][0] + Masks[sl][1 + j][1]) + startImage[i + j][1] * Masks[sl][1 + j][2];
		for (int i = 1; i < imageSizeX - 1; i++)
			for (int j = -1; j < 2; j++)
				listImage[sl][0][i] += startImage[0][i + j] * (Masks[sl][0][1 + j] + Masks[sl][1][j + 1]) + startImage[1][i + j] * Masks[sl][2][1 + j];
		for (int i = 1; i < imageSizeX - 1; i++)
			for (int j = -1; j < 2; j++)
				listImage[sl][imageSizeY - 1][i] += startImage[imageSizeY - 1][i + j] * (Masks[sl][2][1 + j] + Masks[sl][1][1 + j]) + startImage[imageSizeY - 2][i + j] * Masks[sl][0][1 + j];
		for (int i = 1; i < imageSizeY - 1; i++)
			for (int j = -1; j < 2; j++)
				listImage[sl][i][imageSizeX - 1] += startImage[i + j][imageSizeX - 1] * (Masks[sl][1 + j][2] + Masks[sl][1 + j][1]) + startImage[i + j][imageSizeX - 2] * Masks[sl][1 + j][0];
		//Обрабодка внутренней области изображения 
		for (int i = 1; i < imageSizeY - 1; i++)
			for (int j = 1; j < imageSizeX - 1; j++)
				for (int ii = -1; ii < 2; ii++)
					for (int jj = -1; jj < 2; jj++)
						listImage[sl][i][j] += startImage[i + ii][j + jj] * Masks[sl][ii + 1][jj + 1];
	});//130mc
	matrActivateFunction(listImage,0);//Пропускаем все через активационную ф-ю //33mc
}

void convolution(vector<vector<vector<vector<long double>>>>& listImage, vector<vector<vector<long double>>>& Masks, int currentLayer, int succLayer, int imageSizeY, int imageSizeX)
{
	int mx = Masks.size();
	parallel_for(int(0), mx, [&](int c) {
			//Обрабодка углов изображения
		listImage[succLayer][c][0][0] += listImage[currentLayer][c][0][0] * Masks[c][0][0] + listImage[currentLayer][c][0][0] * Masks[c][0][1] + listImage[currentLayer][c][0][1] * Masks[c][0][2] +
			listImage[currentLayer][c][0][0] * Masks[c][1][0] + listImage[currentLayer][c][0][0] * Masks[c][1][1] + listImage[currentLayer][c][0][1] * Masks[c][1][2] +
			listImage[currentLayer][c][1][0] * Masks[c][2][0] + listImage[currentLayer][c][1][0] * Masks[c][2][1] + listImage[currentLayer][c][1][1] * Masks[c][2][2];
		listImage[succLayer][c][0][imageSizeX - 1] += listImage[currentLayer][c][0][imageSizeX - 2] * Masks[c][0][0] + listImage[currentLayer][c][0][imageSizeX - 1] * Masks[c][0][1] + listImage[currentLayer][c][0][imageSizeX - 1] * Masks[c][0][2] +
			listImage[currentLayer][c][0][imageSizeX - 2] * Masks[c][1][0] + listImage[currentLayer][c][0][imageSizeX - 1] * Masks[c][1][1] + listImage[currentLayer][c][0][imageSizeX - 1] * Masks[c][1][2] +
			listImage[currentLayer][c][1][imageSizeX - 2] * Masks[c][2][0] + listImage[currentLayer][c][1][imageSizeX - 1] * Masks[c][2][1] + listImage[currentLayer][c][1][imageSizeX - 1] * Masks[c][2][2];
		listImage[succLayer][c][imageSizeY - 1][0] += listImage[currentLayer][c][imageSizeY - 2][0] * Masks[c][0][0] + listImage[currentLayer][c][imageSizeY - 2][0] * Masks[c][0][1] + listImage[currentLayer][c][imageSizeY - 2][1] * Masks[c][0][2] +
			listImage[currentLayer][c][imageSizeY - 1][0] * Masks[c][1][0] + listImage[currentLayer][c][imageSizeY - 1][0] * Masks[c][1][1] + listImage[currentLayer][c][imageSizeY - 1][1] * Masks[c][1][2] +
			listImage[currentLayer][c][imageSizeY - 1][0] * Masks[c][2][0] + listImage[currentLayer][c][imageSizeY - 1][0] * Masks[c][2][1] + listImage[currentLayer][c][imageSizeY - 1][1] * Masks[c][2][2];
		listImage[succLayer][c][imageSizeY - 1][imageSizeX - 1] += listImage[currentLayer][c][imageSizeY - 2][imageSizeX - 2] * Masks[c][0][0] + listImage[currentLayer][c][imageSizeY - 2][imageSizeX - 1] * Masks[c][0][1]
			+ listImage[currentLayer][c][imageSizeY - 2][imageSizeX - 1] * Masks[c][0][2] +
			listImage[currentLayer][c][imageSizeY - 1][imageSizeX - 2] * Masks[c][1][0] + listImage[currentLayer][c][imageSizeY - 1][imageSizeX - 1] * Masks[c][1][1]
			+ listImage[currentLayer][c][imageSizeY - 1][imageSizeX - 1] * Masks[c][1][2] +
			listImage[currentLayer][c][imageSizeY - 1][imageSizeX - 2] * Masks[c][2][0] + listImage[currentLayer][c][imageSizeY - 1][imageSizeX - 1] * Masks[c][2][1]
			+ listImage[currentLayer][c][imageSizeY - 1][imageSizeX - 1] * Masks[c][2][2];
		//Обрабока краев изображения
		for (int i = 1; i < imageSizeY - 1; i++)
			for (int j = -1; j < 2; j++)
				listImage[succLayer][c][i][0] += listImage[currentLayer][c][i + j][0] * (Masks[c][j + 1][0] + Masks[c][1 + j][1]) + listImage[currentLayer][c][i + j][1] * Masks[c][1 + j][2];
		for (int i = 1; i < imageSizeX - 1; i++)
			for (int j = -1; j < 2; j++)
				listImage[succLayer][c][0][i] += listImage[currentLayer][c][0][i + j] * (Masks[c][0][1 + j] + Masks[c][1][j + 1]) + listImage[currentLayer][c][1][i + j] * Masks[c][2][1 + j];
		for (int i = 1; i < imageSizeX - 1; i++)
			for (int j = -1; j < 2; j++)
				listImage[succLayer][c][imageSizeY - 1][i] += listImage[currentLayer][c][imageSizeY - 1][i + j] * (Masks[c][2][1 + j] + Masks[c][1][1 + j]) + listImage[currentLayer][c][imageSizeY - 2][i + j] * Masks[c][0][1 + j];
		for (int i = 1; i < imageSizeY - 1; i++)
			for (int j = -1; j < 2; j++)
				listImage[succLayer][c][i][imageSizeX - 1] += listImage[currentLayer][c][i + j][imageSizeX - 1] * (Masks[c][1 + j][2] + Masks[c][1 + j][1]) + listImage[currentLayer][c][i + j][imageSizeX - 2] * Masks[c][1 + j][0];
		//Обрабодка внутренней области изображения
		for (int i = 1; i < imageSizeY - 1; i++)
			for (int j = 1; j < imageSizeX - 1; j++)
				for (int ii = -1; ii < 2; ii++)
					for (int jj = -1; jj < 2; jj++)
						listImage[succLayer][c][i][j] += listImage[currentLayer][c][i + ii][j + jj] * Masks[c][ii + 1][jj + 1];
	});
	matrActivateFunction(listImage[succLayer],succLayer);//Пропускаем все через активационную ф-ю
}

void Pooling(vector<vector<vector<vector<long double>>>>& listImage, int curConvLayer, int imgsizex, int imgsizey, int succLayer) {
	bool split = true;
	parallel_for(int(0), forSizeKol[curConvLayer], [&](int channel) {
			for (int i = 0; i < imgsizey; i += 2)
				for (int j = 0; j < imgsizex; j += 2) {
					long double locMax = pressFunction(listImage[curConvLayer][channel][i][j],
						listImage[curConvLayer][channel][i + 1][j],
						listImage[curConvLayer][channel][i][j + 1],
						listImage[curConvLayer][channel][i + 1][j + 1]); // сжать 4 пикселя в 1

					listImage[succLayer][channel][i / 2][j / 2] = locMax;// отправит результат сжатия на следующий слой

					if (split)//отправить дубликат, если есть раздвоение
						listImage[succLayer][channel + forSizeKol[curConvLayer]][i / 2][j / 2] = locMax;
				}
		});
}

void exitToPerc(int curConvLayer, vector<vector<vector<long double>>>& listImage, vector<long double>& percFirstLvl,
	vector<vector<vector<long double>>>& toPerceptronMasks)
{
	int maskToChannel = kolPercNeurons[0] / kolConvMapsAtLayer[curConvLayer];
	parallel_for(int(0), kolConvMapsAtLayer[curConvLayer], [&](int channel) {
		for (int maskNum = 0; maskNum < maskToChannel; maskNum++) {
			long double res = 0;
			for (int i = 0; i < percSizeY; i++) {
				for (int j = 0; j < percSizeX; j++) {
					res += toPerceptronMasks[channel * maskToChannel + maskNum][i][j] * listImage[channel][i][j];
				}
			}
			percFirstLvl[channel * maskToChannel + maskNum] = sigm(res);
		}
		});
}

void percToNextLvl(vector<long double>& curLvl, int curNeroAmt, vector<long double>& succLvl, int succNeroAmt, vector<vector<long double>>& curWeights, int numOfLayer,
	vector<vector<long double>>& weightBias)
{
	for (int i = 0; i < curNeroAmt; i++) {
		for (int j = 0; j < succNeroAmt; j++) {
			succLvl[j] += curLvl[i] * curWeights[i][j];
		}
		}
	for (int i = 0; i < succNeroAmt; i++) {
		succLvl[i] += 1. * weightBias[numOfLayer][i];
	}
	for (int j = 0; j < succNeroAmt; j++){
			succLvl[j] = sigm(succLvl[j]);
			}
}

void fillStartMasks(vector<vector<vector<long double>>>& Masks)//Рандомное заполнение масок для исходного изображения
{
	for (int q = 0; q < amtBGRConvMaps; q++)//Канал
			for (int i = 0; i < maskSize; i++){//Координаты
				for (int j = 0; j < maskSize; j++) {
					Masks[q][i][j] = (long double)(rand() % 10001 - 4000) / 10000;
				}
            }
}

void fillMask(vector<vector<vector<vector<long double>>>>& listMasks, vector<vector<vector<long double>>>& toPerceptronMasks) {
	int curAmtConvLayer = 1;
	for (int i = 1; i < kolLayers; i++)
	{
		if (blListLayers[i - 1])
		{
			listMasks[curAmtConvLayer-1].assign(forSizeKol[i], vector<vector<long double>>(maskSize, vector<long double>(maskSize))); //!!!!!!!!!!!
			for (int j = 0; j < forSizeKol[i]; j++) {
				for (int i1 = 0; i1 < maskSize; i1++) {
					for (int j1 = 0; j1 < maskSize; j1++) {
						listMasks[curAmtConvLayer-1][j][i1][j1] = (long double)(rand() % 10001 - 5000) / 10000;
					}
				}
			}
			curAmtConvLayer++;
		}
	}
	for (int channel = 0; channel < kolPercNeurons[0]; channel++)
		for (int i = 0; i < percSizeY; i++)
			for (int j = 0; j < percSizeX; j++)
				toPerceptronMasks[channel][i][j] = (float)(rand() % 10001 - 4500) / 1000;//маски перехода в персептрон

}

void fillWeight(vector<vector<long double>>& percLvls, vector<vector<vector<long double>>>& percWeights, vector<vector<long double>>& weightBias) {
	percLvls.assign(4, vector<long double>(kolPercNeurons[0]));
	percWeights[0].assign(kolPercNeurons[0], vector<long double>(kolPercNeurons[1]));
	percWeights[1].assign(kolPercNeurons[1], vector<long double>(kolPercNeurons[2]));
	percWeights[2].assign(kolPercNeurons[2], vector<long double>(kolPercNeurons[3]));
	for (int i = 0; i < percWeights[0].size(); i++) {
		for (int j = 0; j < percWeights[0][0].size(); j++) {
			percWeights[0][i][j] = ((long double)(rand() % 100)) * 0.01 / percWeights[0][0].size();
		}
	}
	for (int i = 0; i < percWeights[1].size(); i++) {
		for (int j = 0; j < percWeights[1][0].size(); j++) {
			percWeights[1][i][j] = ((long double)(rand() % 100)) * 0.01 / percWeights[1][0].size();
		}
	}
	for (int i = 0; i < percWeights[2].size(); i++) {
		for (int j = 0; j < percWeights[2][0].size(); j++) {
			percWeights[2][i][j] = ((long double)(rand() % 100)) * 0.01 / percWeights[2][0].size();
		}
	}
	for (int i = 0; i < kolPercNeurons[1]; i++) {
		weightBias[0][i] = ((long double)(rand() % 10001 - 5000)) / 30000;
	}
	for (int i = 0; i < kolPercNeurons[2]; i++) {
		weightBias[1][i] = ((long double)(rand() % 10001 - 5000)) / 30000;
	}
	for (int i = 0; i < kolPercNeurons[3]; i++) {
		weightBias[2][i] = ((long double)(rand() % 10001 - 5000)) / 30000;
	}
}

void mainLayers(vector<vector<vector<vector<long double>>>>& listImage, vector<vector<long double>>& percLvls, vector<long double> &neuralOut, vector<vector<long double>>& startImage, 
	vector<vector<vector<long double>>>& percWeights, vector<vector<vector<long double>>>& listStartMasks, vector<vector<vector<vector<long double>>>>& listMasks, vector<vector<long double>>& weightBias,
	vector<vector<vector<long double>>>& toPerceptronMasks)
{
	listImage[0].assign(16, vector<vector<long double>>(120, vector<long double>(120)));
	makeFirstLayer(listImage[0], listStartMasks, startImage); //150mc
	int curAmtConvLayer = 1;
	for (int i = 1; i < kolLayers; i++){
			if (blListLayers[i - 1])
			{
				listImage[i].assign(forSizeKol[i], vector<vector<long double>>(layerMapsSize[i], vector<long double>(layerMapsSize[i])));// переделать
				convolution(listImage, listMasks[curAmtConvLayer - 1], i - 1, i, listImage[i - 1][0].size(), listImage[i - 1][0][0].size());
				curAmtConvLayer++;
			}
			else {
				listImage[i].assign(forSizeKol[i], vector<vector<long double>>(layerMapsSize[i], vector<long double>(layerMapsSize[i])));
				Pooling(listImage, i - 1, listImage[i - 1][0].size(), listImage[i - 1][0][0].size(), i);

			}
	}//);
	exitToPerc(curAmtConvLayer - 1, listImage[kolLayers-1], percLvls[0], toPerceptronMasks);//41mc
	percToNextLvl(percLvls[0], kolPercNeurons[0], percLvls[1], kolPercNeurons[1], percWeights[0],0, weightBias);
	percToNextLvl(percLvls[1], kolPercNeurons[1], percLvls[2], kolPercNeurons[2], percWeights[1],1, weightBias);
	percToNextLvl(percLvls[2], kolPercNeurons[2], percLvls[3], kolPercNeurons[3], percWeights[2], 2, weightBias);
}

void correctMaskAtMap3x3(vector<vector<long double>>& Mask, long double Error, long double inpSum, vector<vector<long double>>& mapPiece,int curLayer,int channel,int curConvLayer) {
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++) {
			Mask[i][j] += activateFunctionDX(inpSum) * Error * mapPiece[i][j] * learningRate;
		}
}

void correctMaskWeights(vector<vector<long double>>& Mask, vector<vector<long double>>& Errors, vector<vector<long double>>& succMap, vector<vector<long double>>& curMap, int imageSizeX, int imageSizeY, int curLayer, int channel, int curConvLayer) {
	vector<vector<long double>> tmp(3, vector<long double>(3));

	tmp = { { succMap[0][0], succMap[0][0], succMap[0][1] },
	{ succMap[0][0], succMap[0][0], succMap[0][1] },
	{ succMap[1][0], succMap[1][0], succMap[1][1] } };
	correctMaskAtMap3x3(Mask, Errors[0][0], curMap[0][0], tmp,curLayer,channel, curConvLayer);

	tmp = { { succMap[0][imageSizeX - 2], succMap[0][imageSizeX - 1], succMap[0][imageSizeX - 2] },
	{ succMap[0][imageSizeX - 2], succMap[0][imageSizeX - 1], succMap[0][imageSizeX - 2] },
	{ succMap[1][imageSizeX - 2], succMap[1][imageSizeX - 1], succMap[1][imageSizeX - 1] } };
	correctMaskAtMap3x3(Mask, Errors[0][imageSizeX - 1], curMap[0][imageSizeX - 1], tmp, curLayer, channel, curConvLayer);

	tmp = { { succMap[imageSizeY - 2][0], succMap[imageSizeY - 2][0], succMap[imageSizeY - 2][1] },
	{ succMap[imageSizeY - 1][0], succMap[imageSizeY - 1][0], succMap[imageSizeY - 1][1] },
	{ succMap[imageSizeY - 1][0], succMap[imageSizeY - 1][0], succMap[imageSizeY - 1][1] } };
	correctMaskAtMap3x3(Mask, Errors[imageSizeY - 1][0], curMap[imageSizeY - 1][0], tmp, curLayer, channel, curConvLayer);

	tmp = { { succMap[imageSizeY - 2][imageSizeX - 2], succMap[imageSizeY - 2][imageSizeX - 1], succMap[imageSizeY - 2][imageSizeX - 1] },
	{ succMap[imageSizeY - 1][imageSizeX - 2], succMap[imageSizeY - 1][imageSizeX - 1], succMap[imageSizeY - 1][imageSizeX - 1] },
	{ succMap[imageSizeY - 1][imageSizeX - 2], succMap[imageSizeY - 1][imageSizeX - 1], succMap[imageSizeY - 1][imageSizeX - 1] } };
	correctMaskAtMap3x3(Mask, Errors[imageSizeY - 1][imageSizeX - 1], curMap[imageSizeY - 1][imageSizeX - 1], tmp, curLayer, channel, curConvLayer);

	for (int i = 1; i < imageSizeX - 1; i++) {
		tmp = { { succMap[0][i - 1], succMap[0][i], succMap[0][i + 1] },
		{ succMap[0][i - 1], succMap[0][i], succMap[0][i + 1] },
		{ succMap[1][i - 1], succMap[1][i], succMap[1][i + 1] } };
		correctMaskAtMap3x3(Mask, Errors[0][i], curMap[0][i], tmp, curLayer, channel, curConvLayer);

		tmp = { { succMap[i - 1][0], succMap[i - 1][0], succMap[i - 1][1] },
		{ succMap[i][0], succMap[i][0], succMap[i][1] },
		{ succMap[i + 1][0], succMap[i + 1][0], succMap[i + 1][1] } };
		correctMaskAtMap3x3(Mask, Errors[i][0], curMap[i][0], tmp, curLayer, channel, curConvLayer);

		tmp = { { succMap[imageSizeY - 2][i - 1], succMap[imageSizeY - 2][i], succMap[imageSizeY - 2][i + 1] },
		{ succMap[imageSizeY - 1][i - 1], succMap[imageSizeY - 1][i], succMap[imageSizeY - 1][i + 1] },
		{ succMap[imageSizeY - 1][i - 1], succMap[imageSizeY - 1][i], succMap[imageSizeY - 1][i + 1] } };
		correctMaskAtMap3x3(Mask, Errors[imageSizeY - 1][i], curMap[imageSizeY - 1][i], tmp, curLayer, channel, curConvLayer);

		tmp = { { succMap[i - 1][imageSizeX - 2], succMap[i - 1][imageSizeX - 1], succMap[i - 1][imageSizeX - 1] },
		{ succMap[i][imageSizeX - 2], succMap[i][imageSizeX - 1], succMap[i][imageSizeX - 1] },
		{ succMap[i + 1][imageSizeX - 2], succMap[i + 1][imageSizeX - 1], succMap[i + 1][imageSizeX - 1] } };
		correctMaskAtMap3x3(Mask, Errors[i][imageSizeX - 1], curMap[i][imageSizeX - 1], tmp, curLayer, channel, curConvLayer);
	}

	parallel_for(int(1), imageSizeY-1, [&](int ii) {
		for (int jj = 1; jj < imageSizeX - 1; jj++)
			for (int i = -1; i <= 1; i++)
				for (int j = -1; j <= 1; j++) {
					Mask[i + 1][j + 1] += Errors[ii][jj] * activateFunctionDX(curMap[ii][jj]) * succMap[ii + i][jj + j] * learningRate;
				}
		});
}

void learningConvolution3D(vector<vector<long double>>& Errors, int channelNum, int curImage) {
	correctMaskWeights(listStartMasks[channelNum], Errors, listStartImage, listImage[0][channelNum], imageSizeX, imageSizeY, 0, channelNum, 0);
}

void learningConvolution(vector<vector<long double>>& Errors, int curConvLayer, int curPoolLayer, int curLayer, int channelNum, int curImage);

void learningPooling(int curLayer, int curMap, vector<vector<long double>>& curErrors, int curConvLayer, int curPoolLayer, int curImage);

void learningPercToConv(long double Error, int neyroNum, int curImage) {
	int maskToChannel = kolPercNeurons[0] / kolConvMapsAtLayer[kolConvLayers-1];
	for (int i = 0; i < percSizeY; i++)
		for (int j = 0; j < percSizeX; j++) {
			long double delta = Error * activateFunctionDX(percLvls[0][neyroNum]) * listImage[kolLayers - 1][neyroNum / maskToChannel][i][j] * PercLearningRate;
			percExitErrors[i][j] += toPerceptronMasks[neyroNum][i][j] * Error;
			toPerceptronMasks[neyroNum][i][j] += delta;
		}
	if (neyroNum % maskToChannel == maskToChannel - 1)
	{
		if (blListLayers[kolLayers - 2]) {
			learningConvolution(percExitErrors, kolConvLayers - 2, kolPoolLayers - 1, kolLayers - 1, neyroNum / maskToChannel, curImage);
		}
		else
			learningPooling(kolLayers - 1, neyroNum / maskToChannel, percExitErrors, kolConvLayers - 2, kolPoolLayers - 1, curImage);
	}
}

void learningPerc(vector<vector<long double>>& Errors, int percLvl, int amtNerous, int curImage) {
	if (percLvl > 0) {
		if (percLvl == 3) {
			parallel_for(int(0), amtNerous, [&](int i) {
				Errors[percLvl][i] = sigmDX(percLvls[percLvl][i]) * Errors[percLvl][i];
				});
		}
		else {
			for (int i = 0; i < amtNerous; i++) {
				for (int j = 0; j < kolPercNeurons[percLvl + 1]; j++) {
					Errors[percLvl][i] += percWeights[percLvl][i][j] * Errors[percLvl + 1][j];
				}
				Errors[percLvl][i] *= sigmDX(percLvls[percLvl][i]);
				}//);
		}
		parallel_for(int(0), kolPercNeurons[percLvl - 1], [&](int i) {
			for (int j = 0; j < kolPercNeurons[percLvl]; j++) {
				deltas[percLvl - 1][i][j] = PercLearningRate * Errors[percLvl][j] * percLvls[percLvl - 1][i];
			}
			});

		for (int i = 0; i < kolPercNeurons[percLvl]; i++) {
			deltasBias[percLvl - 1][i] = PercLearningRate * Errors[percLvl][i];
		}

	    learningPerc(Errors, percLvl - 1, kolPercNeurons[percLvl - 1], curImage);
	}
	else if ((percLvl == 0)) {
		for (int i = 0; i < kolPercNeurons[percLvl]; i++) {
			for (int j = 0; j < kolPercNeurons[percLvl + 1]; j++) {
				Errors[percLvl][i] += percWeights[percLvl][i][j] * Errors[percLvl + 1][j];
			}
			Errors[percLvl][i] *= sigmDX(percLvls[percLvl][i]);
			learningPercToConv(Errors[0][i], i, curImage);
			}//);
	}
}

void mixImage() {
	for (int i = 0; i < kolLearningImages; i++) {
		orderImage[i] = i+1;
	}
	for (int i = 0; i < kolLearningImages; i++) {
		swap(orderImage[i], orderImage[rand() % kolLearningImages]);
	}
}

void learn() {
	srand(srandTemp);
	for (int kolPer = 0; kolPer < 4; kolPer++) {
		mixImage();
		fillStartMasks(listStartMasks);
		fillMask(listMasks, toPerceptronMasks);
		fillWeight(percLvls, percWeights, weightBias);
		for (int c = 0; c < 16; c++) myCount[c] = 0;
		long double counter = 0;
		for (int epoch = 0; epoch < 4000; epoch++) {
			mixImage();
			for (int num = 1; num < kolLearningImages + 1; num++) {
				//cout << clock() << endl;
				LoadImg(num, kolPer);
				//cout << clock() << endl;
				for (int i = 0; i < kolPercNeurons[1]; i++) {
					percLvls[1][i] = 0;
				}
				for (int i = 0; i < kolPercNeurons[2]; i++) {
					percLvls[2][i] = 0;
				}
				for (int i = 0; i < kolPercNeurons[3]; i++) {
					percLvls[3][i] = 0;
				}
				//cout << clock() << endl;
				mainLayers(listImage, percLvls, neuralOut, listStartImage, percWeights, listStartMasks, listMasks, weightBias, toPerceptronMasks);
				//cout << clock() << endl;
				for (int i = 0; i < kolMergePoolLayers; i++) {
					listPoolErrors[i].assign(listImage[PovtorLayers[i]].size() / 2, vector<vector<long double>>(listImage[PovtorLayers[i]][0].size(), vector<long double>(listImage[PovtorLayers[i]][0][0].size(), 0)));
				}
				//cout << clock() << endl;
				percExitErrors.assign(percSizeY, vector<long double>(percSizeX, 0));//Обнуляем матрицу ошибок для выхода с персептрона 
				percErrors[0].assign(kolPercNeurons[0], 0);
				percErrors[1].assign(kolPercNeurons[1], 0);
				percErrors[2].assign(kolPercNeurons[2], 0);
				percErrors[3].assign(kolPercNeurons[2], 0);

				for (int i = 0; i < kolPercNeurons[3]; i++) {
					//percErrors[3][i] = trueAns[i][0] - percLvls[3][i];
					if (i == orderImage[num - 1] - 1) {
						percErrors[3][i] = 1. - percLvls[3][i];
					}
					else percErrors[3][i] = 0. - percLvls[3][i];
					//cout << percErrors[3][i] << " ";
					//cout << trueAns[i][0];
					//if (num == 1) {
					Graph[epoch] += abs(percErrors[3][i]);
					counter += abs(percErrors[3][i]);
					//}
				}
				//cout << clock() << endl;
				learningPerc(percErrors, 3, kolPercNeurons[3], 0);//877
				//cout << clock() << endl;
				for (int i = 0; i < kolPercNeurons[0]; i++) {
#pragma omp parallel for schedule(guided, kolPercNeurons[1] / flows)
					for (int j = 0; j < kolPercNeurons[1]; j++) {
						percWeights[0][i][j] += deltas[0][i][j];
					}
				}
				for (int i = 0; i < kolPercNeurons[1]; i++) {
#pragma omp parallel for schedule(guided, kolPercNeurons[2] / flows)
					for (int j = 0; j < kolPercNeurons[2]; j++) {
						percWeights[1][i][j] += deltas[1][i][j];
					}
				}
				for (int i = 0; i < kolPercNeurons[2]; i++) {
#pragma omp parallel for schedule(guided, kolPercNeurons[3] / flows)
					for (int j = 0; j < kolPercNeurons[3]; j++) {
						percWeights[2][i][j] += deltas[2][i][j];
					}
				}

				for (int i = 0; i < kolPercNeurons[3]; i++) weightBias[2][i] += deltasBias[2][i];
				for (int i = 0; i < kolPercNeurons[2]; i++) weightBias[1][i] += deltasBias[1][i]; for (int i = 0; i < kolPercNeurons[1]; i++) weightBias[0][i] += deltasBias[0][i];


				if (epoch % 10 == 0 && epoch != 0 && num == 1) {
					upLoadWeight(0, percWeights[0].size(), percWeights[0][0].size(), folderWeight[kolPer]);
					upLoadWeight(1, percWeights[1].size(), percWeights[1][0].size(), folderWeight[kolPer]);
					upLoadWeight(2, percWeights[2].size(), percWeights[2][0].size(), folderWeight[kolPer]);

					upLoadBias(0, kolPercNeurons[1], folderBias[kolPer]);
					upLoadBias(1, kolPercNeurons[2], folderBias[kolPer]);
					upLoadBias(2, kolPercNeurons[3], folderBias[kolPer]);

					upLoadMask(folderMask[kolPer]);
					cout << "Weight saved. " << "PercErrors*100*Epoch:" << counter << " Time: " << clock()<<"! -------------------------------------------------------------" << endl;
					counter = 0;
				}
			}
		}
	}
}

void loadAll() {
//	cout << "SUCK";
	fillStartMasks(listStartMasks);
	fillMask(listMasks, toPerceptronMasks);
	fillWeight(percLvls, percWeights, weightBias);
	loadWeight(0, percWeights[0].size(), percWeights[0][0].size(), percWeights, "C:/Users/panih/source/repos/CNCGL/CNCGL/weight/");
	loadWeight(1, percWeights[1].size(), percWeights[1][0].size(), percWeights, "C:/Users/panih/source/repos/CNCGL/CNCGL/weight/");
	loadWeight(2, percWeights[2].size(), percWeights[2][0].size(), percWeights, "C:/Users/panih/source/repos/CNCGL/CNCGL/weight/");
	loadMask(listStartMasks, listMasks, toPerceptronMasks, "C:/Users/panih/source/repos/CNCGL/CNCGL/mask/");
	loadBias(0, kolPercNeurons[1], weightBias, "C:/Users/panih/source/repos/CNCGL/CNCGL/bias/");
	loadBias(1, kolPercNeurons[2], weightBias, "C:/Users/panih/source/repos/CNCGL/CNCGL/bias/");
	loadBias(2, kolPercNeurons[3], weightBias, "C:/Users/panih/source/repos/CNCGL/CNCGL/bias/");

//	cout << "SUCK";
	fillStartMasks(listStartMasks1);
	fillMask(listMasks1, toPerceptronMasks1);
	fillWeight(percLvls1, percWeights1, weightBias1);
	loadWeight(0, percWeights[0].size(), percWeights[0][0].size(), percWeights1, "C:/Users/panih/source/repos/CNCGL/CNCGL/weight1/");
	loadWeight(1, percWeights[1].size(), percWeights[1][0].size(), percWeights1, "C:/Users/panih/source/repos/CNCGL/CNCGL/weight1/");
	loadWeight(2, percWeights[2].size(), percWeights[2][0].size(), percWeights1, "C:/Users/panih/source/repos/CNCGL/CNCGL/weight1/");
	loadMask(listStartMasks1, listMasks1, toPerceptronMasks1, "C:/Users/panih/source/repos/CNCGL/CNCGL/mask1/");
	loadBias(0, kolPercNeurons[1], weightBias1, "C:/Users/panih/source/repos/CNCGL/CNCGL/bias1/");
	loadBias(1, kolPercNeurons[2], weightBias1, "C:/Users/panih/source/repos/CNCGL/CNCGL/bias1/");
	loadBias(2, kolPercNeurons[3], weightBias1, "C:/Users/panih/source/repos/CNCGL/CNCGL/bias1/");
//	cout << "SUCK";
	fillStartMasks(listStartMasks2);
	fillMask(listMasks2, toPerceptronMasks2);
	fillWeight(percLvls2, percWeights2, weightBias2);
	loadWeight(0, percWeights[0].size(), percWeights[0][0].size(), percWeights2, "C:/Users/panih/source/repos/CNCGL/CNCGL/weight2/");
	loadWeight(1, percWeights[1].size(), percWeights[1][0].size(), percWeights2, "C:/Users/panih/source/repos/CNCGL/CNCGL/weight2/");
	loadWeight(2, percWeights[2].size(), percWeights[2][0].size(), percWeights2, "C:/Users/panih/source/repos/CNCGL/CNCGL/weight2/");
	loadMask(listStartMasks2, listMasks2, toPerceptronMasks2, "C:/Users/panih/source/repos/CNCGL/CNCGL/mask2/");
	loadBias(0, kolPercNeurons[1], weightBias2, "C:/Users/panih/source/repos/CNCGL/CNCGL/bias2/");
	loadBias(1, kolPercNeurons[2], weightBias2, "C:/Users/panih/source/repos/CNCGL/CNCGL/bias2/");
	loadBias(2, kolPercNeurons[3], weightBias2, "C:/Users/panih/source/repos/CNCGL/CNCGL/bias2/");
//	cout << "SUCK";
	fillStartMasks(listStartMasks3);
	fillMask(listMasks3, toPerceptronMasks3);
	fillWeight(percLvls3, percWeights3, weightBias3);
	loadWeight(0, percWeights[0].size(), percWeights[0][0].size(), percWeights3, "C:/Users/panih/source/repos/CNCGL/CNCGL/weight3/");
	loadWeight(1, percWeights[1].size(), percWeights[1][0].size(), percWeights3, "C:/Users/panih/source/repos/CNCGL/CNCGL/weight3/");
	loadWeight(2, percWeights[2].size(), percWeights[2][0].size(), percWeights3, "C:/Users/panih/source/repos/CNCGL/CNCGL/weight3/");
	loadMask(listStartMasks3, listMasks3, toPerceptronMasks3, "C:/Users/panih/source/repos/CNCGL/CNCGL/mask3/");
	loadBias(0, kolPercNeurons[1], weightBias3, "C:/Users/panih/source/repos/CNCGL/CNCGL/bias3/");	
	loadBias(1, kolPercNeurons[2], weightBias3, "C:/Users/panih/source/repos/CNCGL/CNCGL/bias3/");
	loadBias(2, kolPercNeurons[3], weightBias3, "C:/Users/panih/source/repos/CNCGL/CNCGL/bias3/");


	cout << "All was successfully load in memory!" << endl;
}

void showOn();

void recognitionFromImg() {
	setlocale(LC_ALL, "Russian");
	loadAll();
	handelVideo(578);
	//cout << clock() << endl; 
	mainLayers(listImage, percLvls, neuralOut, listStartImage, percWeights, listStartMasks, listMasks, weightBias, toPerceptronMasks);//550
	//cout << clock() << endl;
	cout << "True:";
	long double ma = -1;
	long double num = 0;
	long double m = -1;
    for (int i = 0; i < 16; i++) {
		cout << percLvls[3][i] << " ";
		if (percLvls[3][i] > m) {
			m = percLvls[3][i];
			ma = m;
			ans[0] = i+1;
			num = ans[0];
			//cout << num;
		}
	}
	cout << endl << endl;
	if (m <= 0.8) {
		mainLayers(listImage1, percLvls1, neuralOut1, listStartImage1, percWeights1, listStartMasks1, listMasks1, weightBias1, toPerceptronMasks1);
		m = -1;
		for (int i = 0; i < 16; i++) {
			cout << percLvls1[3][i] << " ";
			if (percLvls1[3][i] > m) {
				m = percLvls1[3][i];
				ans[1] = i + 1;
				if (m > ma) {
					ma = m; num = ans[1]; //cout << num;
				}
			}
		}
		cout << endl << endl;
	}
	if (m <= 0.8) {
		mainLayers(listImage2, percLvls2, neuralOut2, listStartImage2, percWeights2, listStartMasks2, listMasks2, weightBias2, toPerceptronMasks2);
		m = -1;
		for (int i = 0; i < 16; i++) {
			cout << percLvls2[3][i] << " ";
			if (percLvls2[3][i] > m) {
				m = percLvls2[3][i];
				ans[2] = i + 1;
				if (m > ma) {
					ma = m; num = ans[2]; //cout << num;
				}
			}
		}
		cout << endl << endl;
	}
	if (m <= 0.8) {
		mainLayers(listImage3, percLvls3, neuralOut3, listStartImage3, percWeights3, listStartMasks3, listMasks3, weightBias3, toPerceptronMasks3);
		m = -1;
		for (int i = 0; i < 16; i++) {
			cout << percLvls3[3][i] << " ";
			if (percLvls3[3][i] > m) {
				m = percLvls3[3][i];
				ans[3] = i + 1;
				if (m > ma) {
					ma = m; num = ans[3]; //cout << num;
				}
			}
		}
	}
	cout << endl << endl;

	cout << ans[0] << " " << ans[1] << " " << ans[2] << " " << ans[3];
	cout << endl << pozFilter[num-1] << " " << num  << endl;
}

void hendelOfVideo() {
	setlocale(LC_ALL, "Russian");
	for (int kol = 1; kol <= 3628; kol++) {
		char red;
		ifstream file1("C:/Users/panih/source/repos/CNCGL/CNCGL/but.txt");
		file1 >> red;
		file1.close();
		ofstream file("C:/Users/panih/source/repos/CNCGL/CNCGL/but.txt");
		if (red == '0') {
			showOnScreen = false;
		}
		else if (red == '1') {
			showOnScreen = true;
			file << "0";
		}
		file.close();
		//loadImgDisp(ko);
		handelVideo(kol);
		//cout << clock() << endl;
		mainLayers(listImage, percLvls, neuralOut, listStartImage, percWeights, listStartMasks, listMasks, weightBias, toPerceptronMasks);//550
		//cout << clock() << endl;
		//cout << "True:";
		long double m = -1;
		for (int i = 0; i < 16; i++) {
			//cout << percLvls[3][i] << " ";
			if (percLvls[3][i] > m) {
				m = percLvls[3][i];
				ans[0] = i + 1;
			}
		}
		//cout << m<<endl;
		rec[kol] = ans[0];
		//cout << endl << endl;
		if (m <= 0.9) {
			mainLayers(listImage1, percLvls1, neuralOut1, listStartImage1, percWeights1, listStartMasks1, listMasks1, weightBias1, toPerceptronMasks1);
			m = -1;
			for (int i = 0; i < 16; i++) {
				//	cout << percLvls1[3][i] << " ";
				if (percLvls1[3][i] > m) {
					m = percLvls1[3][i];
					ans[1] = i + 1;
				}
			}
			//cout << endl << endl;
			//cout << m << endl;
			rec[kol] = ans[1];
		}
		if (m <= 0.9) {
			mainLayers(listImage2, percLvls2, neuralOut2, listStartImage2, percWeights2, listStartMasks2, listMasks2, weightBias2, toPerceptronMasks2);
			m = -1;
			for (int i = 0; i < 16; i++) {
				//cout << percLvls2[3][i] << " ";
				if (percLvls2[3][i] > m) {
					m = percLvls2[3][i];
					ans[2] = i + 1;
				}
			}
			//cout << endl << endl;
			//cout << m << endl;
			rec[kol] = ans[2];
		}
		if (m <= 0.9) {
			mainLayers(listImage3, percLvls3, neuralOut3, listStartImage3, percWeights3, listStartMasks3, listMasks3, weightBias3, toPerceptronMasks3);
			m = -1;
			for (int i = 0; i < 16; i++) {
				//	cout << percLvls3[3][i] << " ";
				if (percLvls3[3][i] > m) {
					m = percLvls3[3][i];
					ans[3] = i + 1;
				}
			}
			//cout << m << endl;
			rec[kol] = ans[3];
		}
		cout << endl <<pozFilter[rec[kol]-1]<<" " <<rec[kol]<<" "<<kol<<endl;
		if (showOnScreen) {
			//cout << showOnScreen << endl;
			showOn();
		}
	}
	ofstream file("video.txt");
	for (int i = 1; i <= 3628; i++) {
		file << rec[i] << " ";
	}
	file.close();
}

void recognitionFromVideo() {
	long long kol=0;
	while (!_kbhit())
	{
		char red;
		ifstream file1("C:/Users/panih/source/repos/CNCGL/CNCGL/but.txt");
		file1 >> red;
		file1.close();
		ofstream file("C:/Users/panih/source/repos/CNCGL/CNCGL/but.txt");
		if (red == '0') { 
			showOnScreen = false;
		}
		else if (red == '1'){
			showOnScreen = true;
			file << "0";
	    }
		file.close();
		int an=0;
			loadImgDisp(0);
			//cout << clock() << endl;
			mainLayers(listImage, percLvls, neuralOut, listStartImage, percWeights, listStartMasks, listMasks, weightBias, toPerceptronMasks);//550
			//cout << clock() << endl;
			//cout << "True:";
			long double m = -1;
			for (int i = 0; i < 16; i++) {
				//cout << percLvls[3][i] << " ";x
				if (percLvls[3][i] > m) {
					m = percLvls[3][i];
					ans[0] = i + 1;
				}
			}
			//cout << m<<endl;
			an = ans[0];
			//cout << endl << endl;
			if (m <= 0.9) {
				mainLayers(listImage1, percLvls1, neuralOut1, listStartImage1, percWeights1, listStartMasks1, listMasks1, weightBias1, toPerceptronMasks1);
				m = -1;
				for (int i = 0; i < 16; i++) {
				//	cout << percLvls1[3][i] << " ";
					if (percLvls1[3][i] > m) {
						m = percLvls1[3][i];
						ans[1] = i + 1;
					}
				}
				//cout << endl << endl;
				//cout << m << endl;
				an = ans[1];
			}
			if (m <= 0.9) {
				mainLayers(listImage2, percLvls2, neuralOut2, listStartImage2, percWeights2, listStartMasks2, listMasks2, weightBias2, toPerceptronMasks2);
				m = -1;
				for (int i = 0; i < 16; i++) {
					//cout << percLvls2[3][i] << " ";
					if (percLvls2[3][i] > m) {
						m = percLvls2[3][i];
						ans[2] = i + 1;
					}
				}
				//cout << endl << endl;
				//cout << m << endl;
				an = ans[2];
			}
			if (m <= 0.9) {
				mainLayers(listImage3, percLvls3, neuralOut3, listStartImage3, percWeights3, listStartMasks3, listMasks3, weightBias3, toPerceptronMasks3);
				m = -1;
				for (int i = 0; i < 16; i++) {
				//	cout << percLvls3[3][i] << " ";
					if (percLvls3[3][i] > m) {
						m = percLvls3[3][i];
						ans[3] = i + 1;
					}
				}
				//cout << m << endl;
				an = ans[3];
			}
			//cout <<endl<< ans[0] << " " << ans[1] << " " << ans[2] << " " << ans[3]<<endl;
			ofstream fil("C:/Users/panih/source/repos/CNCGL/CNCGL/video.txt");
			fil << an;
			fil.close();

			if (showOnScreen) {
				//cout << showOnScreen << endl;
				showOn(); 
			}
	}
}

void setup() {
	//loadAll();
	loadImgDisp(17);
	//cout << clock() << endl;
	mainLayers(listImage, percLvls, neuralOut, listStartImage, percWeights, listStartMasks, listMasks, weightBias, toPerceptronMasks);//550
	//cout << clock() << endl;
	cout << "True:";
	long double m = -1;
	for (int i = 0; i < 16; i++) {
		cout << percLvls[3][i] << " ";
		if (percLvls[3][i] > m) {
			m = percLvls[3][i];
			ans[0] = i + 1;
		}
	}
	cout<<endl << "Right Cam Out: ";
	cout<< m << endl;
	loadImgDisp(17);
	mainLayers(listImage2, percLvls2, neuralOut2, listStartImage2, percWeights2, listStartMasks2, listMasks2, weightBias2, toPerceptronMasks2);
	m = -1;
	for (int i = 0; i < 16; i++) {
		cout << percLvls2[3][i] << " ";
		if (percLvls2[3][i] > m) {
			m = percLvls2[3][i];
			ans[2] = i + 1;
		}
	}
	cout << endl << "Left Cam Out: ";
	cout << m;
}

void writeToFolder() {
	cout << "When you are ready write </ready>" << endl;
	for (int i = 1; i <= 32; i++) {
		string s;
		cin >> s;
		while (s != "/ready") {
			cin >> s;
		}
		long long t = clock();
		long long k = 0;
		int kol = 5;
		int numm = 0;
		while (k - t <= 5000) {
			k = clock();
			if (k % 1000 == 0 && numm!=k) {
				cout << kol<<" sec!"<<clock() << endl;
				numm = k;
				kol--;
			}
		}
		int j = 0;
		kol = 10;
		int nummm = 0;
		k = 0;
		while (j < 1000) {
			if(clock()%10==0){
				
				k = clock();
				if (k % 1000 == 0 && numm != k) {
					cout << kol << " sec!" << clock() << endl;
					numm = k;
					kol--;
				}

				stringstream ss;
				ss << j;
				string name = ss.str();
				stringstream s;
				s << i-1;
				string na = s.str();
				name += ".jpg";
				name = "C:/Users/panih/source/repos/CNCGL/CNCGL/tr5/"+ na+ "/" + name;
				Mat img;
				img = imread("C:/Users/panih/source/repos/CNCGL/CNCGL/tr1/1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
				resize(img, img, Size(120, 120), 0, 0, INTER_CUBIC);
				imwrite(name, img);

				stringstream ss1;
				ss1 << j;
				string name1 = ss1.str();
				stringstream s1;
				s1 << i-1;
				string na1 = s1.str();
				name1 += ".jpg";
				name1 = "C:/Users/panih/source/repos/CNCGL/CNCGL/tr6/" + na1 + "/" + name1;
				Mat img1;
				img1 = imread("C:/Users/panih/source/repos/CNCGL/CNCGL/tr1/2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
				resize(img1, img1, Size(120, 120), 0, 0, INTER_CUBIC);
				imwrite(name1, img1);
				j++;
			}
		}
		cout << "I'm wait you!" << endl;
	}
}

GLvoid ReSizeGLScene(GLsizei width, GLsizei height)
{
	if (height == 0)
	{
		height = 1;
	}

	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	// Calculate The Aspect Ratio Of The Window
	gluPerspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

int InitGL(GLvoid)
{
	glShadeModel(GL_SMOOTH);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	return TRUE;
}

GLfloat Tangx(int x, int y) {
	if (x != 0)
		return GLfloat((float)x / 640);
	if (y != 0)
		return GLfloat((float)y / 480);
}

int DrawGLScene(GLvoid)
{
	if (keys[87]) //W
		yy -= 0.001f;
	if (keys[83]) //S
		yy += 0.001f;
	if (keys[65]) //A
		xx += 0.001f;
	if (keys[68]) //D
		xx -= 0.001f;
	if (keys[49]) //Vverh
		zz += 0.001f;
	if (keys[50]) //Vniz
		zz -= 0.001f;
	if (keys[51])
		rtri += 0.05f;
	if (keys[52])
		rtri -= 0.05f;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	//
	glMultMatrixf(m);
	glTranslatef(xx, yy, zz);
	glRotatef(rtri, 0.0f, 0.1f, 0.0f);
	glBegin(GL_LINES);
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(mk1[1].x, mk1[1].y, 0.0f);
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(mk1[2].x, mk1[2].y, 0.0f);

	glColor3f(1.1f, 0.0f, 0.0f);
	glVertex3f(mk1[2].x, mk1[2].y, 0.0f);
	glColor3f(1.1f, 0.0f, 0.0f);
	glVertex3f(mk1[4].x, mk1[4].y, 0.0f);

	glColor3f(1.0f, 0.1f, 0.0f);
	glVertex3f(mk1[1].x, mk1[1].y, 0.0f);
	glColor3f(1.0f, 0.1f, 0.0f);
	glVertex3f(mk1[3].x, mk1[3].y, 0.0f);

	glColor3f(1.0f, 0.0f, 0.1f);
	glVertex3f(mk1[3].x, mk1[3].y, 0.0f);
	glColor3f(1.0f, 0.0f, 0.1f);
	glVertex3f(mk1[5].x, mk1[5].y, 0.0f);

	glColor3f(1.1f, 0.1f, 0.0f);
	glVertex3f(mk1[1].x, mk1[1].y, 0.0f);
	glColor3f(1.1f, 0.1f, 0.0f);
	glVertex3f(mk1[6].x, mk1[6].y, 0.0f);

	glColor3f(1.0f, 0.1f, 0.1f);
	glVertex3f(mk1[6].x, mk1[6].y, 0.0f);
	glColor3f(1.0f, 0.1f, 0.1f);
	glVertex3f(mk1[7].x, mk1[7].y, 0.0f);

	glColor3f(1.1f, 0.2f, 0.1f);
	glVertex3f(mk1[7].x, mk1[7].y, 0.0f);
	glColor3f(1.1f, 0.2f, 0.1f);
	glVertex3f(mk1[8].x, mk1[8].y, 0.0f);

	glColor3f(1.3f, 0.2f, 0.2f);
	glVertex3f(mk1[8].x, mk1[8].y, 0.0f);
	glColor3f(1.3f, 0.2f, 0.2f);
	glVertex3f(mk1[10].x, mk1[10].y, 0.0f);

	glColor3f(1.6f, 0.2f, 0.2f);
	glVertex3f(mk1[7].x, mk1[7].y, 0.0f);
	glColor3f(1.6f, 0.2f, 0.2f);
	glVertex3f(mk1[9].x, mk1[9].y, 0.0f);

	glColor3f(1.6f, 0.2f, 0.6f);
	glVertex3f(mk1[9].x, mk1[9].y, 0.0f);
	glColor3f(1.6f, 0.2f, 0.6f);
	glVertex3f(mk1[11].x, mk1[11].y, 0.0f);
	glEnd();

	return true;
}

GLvoid KillGLWindow(GLvoid)
{
	if (fullscreen) {
		ChangeDisplaySettings(NULL, 0);
		ShowCursor(TRUE);
	}
	if (hRC) {
		if (!wglMakeCurrent(NULL, NULL)) {
			MessageBox(NULL, L"Release Of DC And RC Failed.", L"SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
		}
		if (!wglDeleteContext(hRC)) {
			MessageBox(NULL, L"Release Rendering Context Failed.", L"SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
		}
		hRC = NULL;
	}
	if (hDC && !ReleaseDC(hWnd, hDC)) {
		MessageBox(NULL, L"Release Device Context Failed.", L"SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
		hDC = NULL;
	}
	if (hWnd && !DestroyWindow(hWnd)) {
		MessageBox(NULL, L"Could Not Release hWnd.", L"SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
		hWnd = NULL;
	}
	if (!UnregisterClass(L"OpenGL", hInstance)) {
		MessageBox(NULL, L"Could Not Unregister Class.", L"SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
		hInstance = NULL;
	}
}

BOOL CreateGLWindow(LPCTSTR title, int width, int height, int bits, bool fullscreenflag)
{
	GLuint      PixelFormat;
	WNDCLASS    wc;
	DWORD       dwExStyle;
	DWORD       dwStyle;
	RECT WindowRect;

	WindowRect.left = (long)0;
	WindowRect.right = (long)width;
	WindowRect.top = (long)0;
	WindowRect.bottom = (long)height;

	fullscreen = fullscreenflag;

	hInstance = GetModuleHandle(NULL);
	wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wc.lpfnWndProc = (WNDPROC)WndProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = hInstance;
	wc.hIcon = LoadIcon(NULL, IDI_WINLOGO);
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = NULL;
	wc.lpszMenuName = NULL;
	wc.lpszClassName = L"OpenGL";

	if (!RegisterClass(&wc)) {
		MessageBox(NULL, L"Failed To Register The Window Class.", L"ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE;
	}
	if (fullscreen) {
		DEVMODE dmScreenSettings;
		memset(&dmScreenSettings, 0, sizeof(dmScreenSettings));
		dmScreenSettings.dmSize = sizeof(dmScreenSettings);
		dmScreenSettings.dmPelsWidth = width;
		dmScreenSettings.dmPelsHeight = height;
		dmScreenSettings.dmBitsPerPel = bits;
		dmScreenSettings.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;

		if (ChangeDisplaySettings(&dmScreenSettings, CDS_FULLSCREEN) != DISP_CHANGE_SUCCESSFUL) {
			if (MessageBox(NULL, L"The Requested Fullscreen Mode Is Not Supported By\nYour Video Card. Use Windowed Mode Instead?", L"NeHe GL", MB_YESNO | MB_ICONEXCLAMATION) == IDYES) {
				fullscreen = FALSE;
			}
			else {
				MessageBox(NULL, L"Program Will Now Close.", L"ERROR", MB_OK | MB_ICONSTOP);
				return FALSE;
			}
		}
	}
	if (fullscreen) {
		dwExStyle = WS_EX_APPWINDOW;
		dwStyle = WS_POPUP;
		ShowCursor(FALSE);
	}
	else {
		dwExStyle = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;
		dwStyle = WS_OVERLAPPEDWINDOW;
	}
	AdjustWindowRectEx(&WindowRect, dwStyle, FALSE, dwExStyle);
	if (!(hWnd = CreateWindowEx(dwExStyle,
		L"OpenGL",
		title,
		WS_CLIPSIBLINGS |
		WS_CLIPCHILDREN |
		dwStyle,
		0, 0,
		WindowRect.right - WindowRect.left,
		WindowRect.bottom - WindowRect.top,
		NULL,
		NULL,
		hInstance,
		NULL)))
	{
		KillGLWindow();
		MessageBox(NULL, L"Window Creation Error.", L"ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE;
	}
	static  PIXELFORMATDESCRIPTOR pfd =
	{
		sizeof(PIXELFORMATDESCRIPTOR),
		1,
		PFD_DRAW_TO_WINDOW |
		PFD_SUPPORT_OPENGL |
		PFD_DOUBLEBUFFER,
		PFD_TYPE_RGBA,
		bits,
		0, 0, 0, 0, 0, 0,
		0,
		0,
		0,
		0, 0, 0, 0,
		16,
		0,
		0,
		PFD_MAIN_PLANE,
		0,
		0, 0, 0
	};
	if (!(hDC = GetDC(hWnd))) {
		KillGLWindow();
		MessageBox(NULL, L"Can't Create A GL Device Context.", L"ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE;
	}
	if (!(PixelFormat = ChoosePixelFormat(hDC, &pfd))) {
		KillGLWindow();
		MessageBox(NULL, L"Can't Find A Suitable PixelFormat.", L"ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE;
	}
	if (!SetPixelFormat(hDC, PixelFormat, &pfd)) {
		KillGLWindow();
		MessageBox(NULL, L"Can't Set The PixelFormat.", L"ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE;
	}
	if (!(hRC = wglCreateContext(hDC))) {
		KillGLWindow();
		MessageBox(NULL, L"Can't Create A GL Rendering Context.", L"ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE;
	}
	if (!wglMakeCurrent(hDC, hRC)) {
		KillGLWindow();
		MessageBox(NULL, L"Can't Activate The GL Rendering Context.", L"ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE;
	}
	ShowWindow(hWnd, SW_SHOW);
	SetForegroundWindow(hWnd);
	SetFocus(hWnd);
	ReSizeGLScene(width, height);
	if (!InitGL()) {
		KillGLWindow();
		MessageBox(NULL, L"Initialization Failed.", L"ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE;
	}
	return true;
}

LRESULT CALLBACK WndProc(HWND    hWnd,
	UINT    uMsg,
	WPARAM  wParam,
	LPARAM  lParam)
{
	switch (uMsg) {
	case WM_ACTIVATE:
	{
		if (!HIWORD(wParam)) {
			active = TRUE;
		}
		else {
			active = FALSE;
		}
		return 0;
	}
	case WM_SYSCOMMAND:
	{
		switch (wParam)
		{
		case SC_SCREENSAVE:
		case SC_MONITORPOWER:
			return 0;
		}
		break;
	}
	case WM_CLOSE:
	{
		PostQuitMessage(0);
		return 0;
	}
	case WM_KEYDOWN:
	{
		keys[wParam] = TRUE;
		return 0;
	}
	case WM_KEYUP:
	{
		keys[wParam] = FALSE;
		return 0;
	}
	case WM_SIZE:
	{
		ReSizeGLScene(LOWORD(lParam), HIWORD(lParam));
		return 0;
	}
	}
	return DefWindowProc(hWnd, uMsg, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE   hInstance,
	HINSTANCE   hPrevInstance,
	LPSTR       lpCmdLine,
	int     nCmdShow)
{
	MSG msg;
	BOOL    done = FALSE;
	fullscreen = FALSE;

	if (!CreateGLWindow(L"wHaT!?", 640, 480, 16, fullscreen))
		return 0;

	//cout << done << endl;
	while (!done) {
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
			if (msg.message == WM_QUIT)
			{
				done = TRUE;
				showOnScreen = false;
			}
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else {
			if (active) {
				if (keys[VK_ESCAPE])
				{
					done = TRUE;
					showOnScreen = false;
				}
				else
				{
					DrawGLScene();
					SwapBuffers(hDC);
				}
			}
		}
	}
	showOnScreen = false;
	KillGLWindow();
	return (msg.wParam);
}
	
void showOn() {
	ifstream file2("C:/Users/panih/source/repos/CNCGL/CNCGL/poz.txt");
	for (int i = 1; i <= 11; i++) {
		file2 >> mk[i].x >> mk[i].y;
	}
	for (int i = 1; i < 11; i++) {
		file2 >> mk0[i].x >> mk0[i].y;
	}
	file2.close();
	for (int i = 1; i < 11; i++) {
		mk1[i].x = Tangx(((mk[i].x + mk0[i].x) / 2), 0);
		mk1[i].y = -Tangx(0, ((mk[i].y + mk0[i].y) / 2));
		mk1[i].z = mk1[i].x;
	}
	WinMain(hInstance, hInstance, b, 290);
}

int main()
{
	//hendelOfVideo();
	char red;
	ifstream file1("C:/Users/panih/source/repos/CNCGL/CNCGL/but.txt");
	file1 >> red;
	file1.close();
	if (red == '0') showOnScreen = false;
	else if (red == '1')showOnScreen = true;
	
	string s;
	ifstream file("C:/Users/panih/source/repos/CNCGL/CNCGL/input.txt");
	file >> s;
	file.close();

	while (s.size() == 0) {
		if (clock() % 100 == 0) {
			ifstream file("C:/Users/panih/source/repos/CNCGL/CNCGL/input1.txt");
			file >> s;
			file.close();
			cout << s;
		}
	}
	FILE* ptrFile1 = fopen("C:/Users/panih/source/repos/CNCGL/CNCGL/input.txt", "w");
	fclose(ptrFile1);
	if (s == "/Start") {
		//cout << 1;
		loadAll();
		cout << "Recognition from video start!" << endl;
		recognitionFromVideo();
		cout << "Recognitionn from video end!" << endl;
	}
	else if (s == "/Learn") {
		//loadAll();
		cout << "Training has begun!" << endl;
		learn();
		cout << "Training over!" << endl;
	}
	else if (s == "/Load") {
		loadAll();
		s = "";
		//bool fdg = false;
		while (s != "/Start") {
			if (s == "/Setup") {
				setup();
				s = "";
			}
			ifstream file("C:/Users/panih/source/repos/CNCGL/CNCGL/input.txt");
			file >> s;
			file.close();
			while (s.size() == 0) {
				if (clock() % 100 == 0) {
					ifstream file("C:/Users/panih/source/repos/CNCGL/CNCGL/input.txt");
					file >> s;
					file.close();
					cout << s;
					if (s.size() != 0) {
						FILE* ptrFile = fopen("C:/Users/panih/source/repos/CNCGL/CNCGL/input.txt", "w");
						fclose(ptrFile);
					}
				}
			}
		}
	}
	else if (s == "/Fill") {
		writeToFolder();
	}
	return (0);
}


void learningConvolution(vector<vector<long double>>& Errors, int curConvLayer, int curPoolLayer, int curLayer, int channelNum, int curImage) {
	long double imageSizeY1, imageSizeX1;

	imageSizeY1 = Errors.size(); imageSizeX1 = Errors[0].size();
	vector<vector<long double>> NewErrors(imageSizeY1, vector<long double>(imageSizeX1, 0));
	vector<vector<long double>> prevMask = listMasks[curConvLayer][channelNum];

	correctMaskWeights(listMasks[curConvLayer][channelNum], Errors, listImage[curLayer - 1][channelNum], listImage[curLayer][channelNum], imageSizeX1, imageSizeY1,curLayer, channelNum, curConvLayer);//!!!!

	for (int ii = 1; ii < imageSizeY1 - 1; ii++)
		for (int jj = 1; jj < imageSizeX1 - 1; jj++)
			for (int i = -1; i <= 1; i++)
				for (int j = -1; j <= 1; j++) {
					
						NewErrors[ii + i][jj + j] += ((listMasks[curConvLayer][channelNum][-i + 1][-j + 1]) - prevMask[-i + 1][-j + 1]) * listMasks[curConvLayer][channelNum][-i + 1][-j + 1];
				}
	if (curLayer == 1)
		learningConvolution3D(NewErrors, channelNum, curImage);
	else if (blListLayers[curLayer - 2])
		learningConvolution(NewErrors, curConvLayer - 1, curPoolLayer, curLayer - 1, channelNum, curImage);
	else
		learningPooling(curLayer - 1, channelNum, NewErrors, curConvLayer - 1, curPoolLayer, curImage);

}

void learningPooling(int curLayer, int curMap, vector<vector<long double>>& curErrors, int curConvLayer, int curPoolLayer, int curImage)
{

	vector<vector<long double>> succErrors(listImage[curLayer - 1][0].size(), vector<long double>(listImage[curLayer - 1][0][0].size(), 0));
	bool fMerge = (listImage[curLayer].size() != listImage[curLayer - 1].size());
	if (fMerge)
	{
		if (curLayer != 9) {
			if (curMap < listImage[curLayer - 1].size())
			{
				listPoolErrors[curPoolLayer][curMap] = curErrors; //!!!! error
				return;
			}
			int m1 = listImage[curLayer - 1][0].size();
			int m2 = listImage[curLayer - 1][0][0].size();
			for (int i = 0; i < m1; i += 2)
				for (int j = 0; j < m2; j += 2)
				{
					int mi = 0, mj = 0;
					for (int ii = 0; ii < 2; ii++)
						for (int jj = 0; jj < 2; jj++)
							if (listImage[curLayer - 1][curMap - listImage[curLayer - 1].size()][i + ii][j + jj] > listImage[curLayer - 1][curMap - listImage[curLayer - 1].size()][i + mi][j + mj])
							{
								mi = ii;
								mj = jj;
							}
					for (int isc = i; isc < i + 2; isc++) {
						for (int jsc = j; jsc < j + 2; jsc++) {
							if (curErrors[i / 2][j / 2] + listPoolErrors[curPoolLayer][curMap - listImage[curLayer - 1].size()][i / 2][j / 2] < maksError) {
								succErrors[isc][jsc] = curErrors[i / 2][j / 2] + listPoolErrors[curPoolLayer][curMap - listImage[curLayer - 1].size()][i / 2][j / 2];
							}
							else succErrors[isc][jsc] = maksError;
						}
					}
				}
		}
		else{
			if (curMap < listImage[curLayer - 1].size())
			{
				listPoolErrors[curPoolLayer][curMap] = curErrors;
				return;
			}
			int m1 = listImage[curLayer - 1][0].size();
			int m2 = listImage[curLayer - 1][0][0].size();
			for (int i = 0; i < m1; i += 3)
				for (int j = 0; j < m2; j += 3)
				{
					int mi = 0, mj = 0;
					for (int ii = 0; ii < 3; ii++)
						for (int jj = 0; jj < 3; jj++)
							if (listImage[curLayer - 1][curMap - listImage[curLayer - 1].size()][i + ii][j + jj] > listImage[curLayer - 1][curMap - listImage[curLayer - 1].size()][i + mi][j + mj])
							{
								mi = ii;
								mj = jj;
							}
					for (int isc = i; isc < i + 3; isc++) {
						for (int jsc = j; jsc < j + 3; jsc++) {
							succErrors[isc][jsc] = curErrors[i / 3][j / 3] + listPoolErrors[curPoolLayer][curMap - listImage[curLayer - 1].size()][i / 3][j / 3];
						}
					}
				}
		}
		if (curLayer == 1) {//возможно не так

			learningConvolution3D(succErrors, curMap - listImage[curLayer - 1].size(), curImage);
		}
		else
			learningConvolution(succErrors, curConvLayer, curPoolLayer - 1, curLayer - 1, curMap - listImage[curLayer - 1].size(), curImage);
	}
	else {
		if (curLayer + 1 != 9) {
			int m1 = listImage[curLayer - 1][0].size();
			int m2 = listImage[curLayer - 1][0][0].size();
			for (int i = 0; i < m1; i += 2) {
				for (int j = 0; j < m2; j += 2)
				{
					int mi = 0, mj = 0;
					for (int ii = 0; ii < 2; ii++)
						for (int jj = 0; jj < 2; jj++)
							if (listImage[curLayer - 1][curMap][i + ii][j + jj] > listImage[curLayer - 1][curMap][i + mi][j + mj])
							{
								mi = ii;
								mj = jj;
							}
					succErrors[i + mi][j + mj] = curErrors[i / 2][j / 2];
				}
			}
		}
		else {
			int m1 = listImage[curLayer - 1][0].size();
			int m2 = listImage[curLayer - 1][0][0].size();
			for (int i = 0; i < m1; i += 3) {
				for (int j = 0; j < m2; j += 3)
				{
					int mi = 0, mj = 0;
					for (int ii = 0; ii < 3; ii++)
						for (int jj = 0; jj < 3; jj++)
							if (listImage[curLayer - 1][curMap][i + ii][j + jj] > listImage[curLayer - 1][curMap][i + mi][j + mj])
							{
								mi = ii;
								mj = jj;
							}
					succErrors[i + mi][j + mj] = curErrors[i / 3][j / 3];
				}
			}
		}

		if (curLayer == 1)//возможно не так
			learningConvolution3D(succErrors, curMap, curImage);
		else
			learningConvolution(succErrors, curConvLayer, curPoolLayer - 1, curLayer - 1, curMap, curImage);
	}
}



