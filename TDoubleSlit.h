#ifndef TDOUBLE_SLIT_H
#define TDOUBLE_SLIT_H

#include <iostream>
#include <random>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "DoubleSlitConst.h"

//二重スリットシミュレーションを行うクラス
class TDoubleSlit
{
private:
	//録画用のVideoWriterオブジェクト
	cv::VideoWriter writer;
	//録画用の動画ファイル名
	std::string video_filename;
	//スクリーン描画用のMatを初期化
	cv::Mat canvas;
	//スクリーン描画用Mat
	cv::Mat display_mat;
	//列ごとの数を格納するベクター
	std::vector<int> col_counts;
	//ウィンドウ名
	std::string window_name;
	//CUDA用の乱数状態を保持するデバイスメモリポインタ
	void *d_rand_states = nullptr;
	//CUDA用の列ごとの数を保持するデバイスメモリポインタ
	int *d_counts = nullptr;
	//前回のフレームの幅を保持する変数
	int last_width = 0;
private:
	//録画準備
	cv::VideoWriter prepare_video_writer(bool isCuda);
	//現在の状態をMatに描画する関数（CPU版とCUDA版で共通）
	bool draw_current_state(cv::Mat& canvas, const std::vector<int>& col_counts, int frame_count, double elapsed_time_ms,long long total_electrons, double simulation_seconds, bool is_cuda);
	//(CPU版処理)毎フレーム500個の電子をループでサンプリング
	bool cpu_sampling(std::vector<int>& col_counts);
	// CPU版の描画スケール調整
	bool cpu_draw_adjustment(cv::Mat& canvas, const std::vector<int>& col_counts, std::mt19937& gen, std::uniform_real_distribution<float>& dis_y);
	// CUDA版の描画スケール調整
	bool cuda_draw_adjustment(cv::Mat& canvas, const std::vector<int>& col_counts, std::mt19937& gen, std::uniform_real_distribution<float>& dis_y);
	// CUDA関数の前方宣言(カーネル関数呼び出しなどのCUDA独自の処理)
	void run_cuda_simulation(int* h_counts, int width, int num_electrons, float d, float w, unsigned long seed);
public:
	//コンストラクタ
	TDoubleSlit() = default;
	//デストラクタ
	virtual ~TDoubleSlit() = default;
public:
	//CUDAを用いて処理するC++側処理
	bool use_cuda_cpp_func();
	//CUDA終了時の後処理（GPUメモリ解放）
	void end_cuda_simulation();
	//CUDAなしの処理（C++23標準 CPUシングルスレッド処理）
	bool none_cuda_cpp_func();
};

#endif
