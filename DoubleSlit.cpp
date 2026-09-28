#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <chrono>
#include <numbers>
#include <print>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "TDoubleSlit.h"

using namespace std::chrono;

//---------------------------------------------
//CUDAを用いて処理するC++側処理
//---------------------------------------------
bool TDoubleSlit::use_cuda_cpp_func()
{
	//ウィンドウ名を設定
    window_name = "Double Slit Simulation (C++23 + OpenCV + CUDA)";
	//OpenCVのウィンドウを作成
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
	//スクリーン描画用のMatを初期化
    canvas = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);

	//乱数生成器の初期化
    std::random_device rd;
    std::mt19937 gen(rd());
	//Y座標の乱数分布を設定
    std::uniform_real_distribution<float> dis_y(0.0f, static_cast<float>(HEIGHT));
	//列ごとの数を格納するベクターを初期化
    std::vector<int> col_counts(WIDTH, 0);

    //録画準備
    writer = prepare_video_writer(true);
	//録画可能かチェック
    if (writer.isOpened() == false)
    {
        std::print("エラー: 動画ファイルを開けませんでした。\n");
        return false;
    }
	//フレームカウントの初期化
    int frame_count = 0;

    //メインループ
    while (true)
    {
        //動画が5秒(150フレーム)に達したら終了する
        if (frame_count >= FPS * RECORD_SECONDS)
        {
            break;
        }
        //列ごとの数の初期化
        std::fill(col_counts.begin(), col_counts.end(), 0);
		// 計算時間計測開始
        std::chrono::time_point<std::chrono::high_resolution_clock> calc_start = std::chrono::high_resolution_clock::now();
        
        // CUDAを呼び出してX座標の着弾カウントを更新
        run_cuda_simulation(col_counts.data(), WIDTH, GPU_ELECTRONS_PER_FRAME, static_cast<float>(SLIT_DISTANCE), static_cast<float>(SLIT_WIDTH), rd());

        auto calc_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = calc_end - calc_start;

        // CUDA版の描画スケール調整
        cuda_draw_adjustment(canvas, col_counts, gen, dis_y);
  
		//描画ウィンドウのキャプションを更新（C++23のstd::formatを使用）
		std::string title = std::format("{} | GPU Time: {:.1f} ms", window_name, elapsed.count());
		//ウィンドウタイトルを更新
        cv::setWindowTitle(window_name, title);
        //スクリーン描画用Mat
        display_mat = canvas.clone();

        //処理した電子の数
        long long total_electrons = (long long)frame_count * GPU_ELECTRONS_PER_FRAME;
        //現在のフレーム数をインクリメント
        frame_count++;
        //シミュレートした時間（秒）を計算
        double simulation_seconds = frame_count / static_cast<double>(FPS);

        //現在の状態をMatに描画する
        draw_current_state(display_mat, col_counts, frame_count, elapsed.count(), total_electrons, simulation_seconds, true);

        //録画が出来るならフレーム追加
        if (writer.isOpened() == true) 
        {
            writer.write(display_mat);
        }
		//スクリーンの描画
        cv::imshow(window_name, display_mat);

		//描画を確実に行うためのウェイトを入れる
        //このウェイトはシミュレーション時間の対象外とする
        cv::waitKey(3);
    }
    //録画終了
    writer.release();
	//全てのOpenCVウィンドウを閉じる
	cv::destroyAllWindows();

	return true;
}
//---------------------------------------------
//CUDAなしの処理（C++23標準 CPUシングルスレッド処理）
//---------------------------------------------
bool TDoubleSlit::none_cuda_cpp_func()
{
	//ウィンドウ名を設定
    window_name = "Double Slit Simulation (C++23 + OpenCV | CPU-Only)";
	//OpenCVのウィンドウを作成
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    //スクリーン描画用のMatを初期化
    canvas = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
	//列ごとの数を格納するベクターを初期化
    col_counts.resize(WIDTH, 0);
	//録画準備
    writer= prepare_video_writer(false);
    //録画可能かチェック
    if (!writer.isOpened())
    {
        std::print("エラー: 動画ファイルを開けませんでした。\n");
        return false;   
    }
	//乱数生成器の初期化
    std::random_device rd;
    std::mt19937 gen(rd());
	//X座標とY座標の乱数分布を設定
    std::uniform_real_distribution<float> dis_x(-WIDTH / 2.0f, WIDTH / 2.0f);
    std::uniform_real_distribution<float> dis_y(0.0f, static_cast<float>(HEIGHT));
    std::uniform_real_distribution<float> dis_prob(0.0f, 1.0f);

	//フレームカウントの初期化
    int frame_count = 0;

    //メインループ
    while (true)
    {
        //150フレーム(5秒分)に達したら終了する
        if (frame_count >= FPS * RECORD_SECONDS)
        {
            break;
        }
		//列ごとの数の初期化
        std::fill(col_counts.begin(), col_counts.end(), 0);
		// 計算時間計測開始
        steady_clock::time_point calc_start = high_resolution_clock::now();

        // (CPU版処理)毎フレーム500個の電子をループでサンプリング
        cpu_sampling(col_counts);

		// 計算時間計測終了
        steady_clock::time_point calc_end = high_resolution_clock::now();
		// 計算時間をミリ秒単位で取得
        std::chrono::duration<double, std::milli> elapsed = calc_end - calc_start;

        // CPU版の描画スケール調整
        cpu_draw_adjustment(canvas, col_counts, gen,dis_y);

		//描画ウィンドウのキャプションを更新（C++23のstd::formatを使用）
        std::string title = std::format("{} | CPU Time: {:.1f} ms", window_name, elapsed.count());
        cv::setWindowTitle(window_name, title);
		//スクリーン描画用Mat
        display_mat = canvas.clone();
        int fixed_max_scale = 50;

        //処理した電子の数
        long long total_electrons = (long long)frame_count * CPU_ELECTRONS_PER_FRAME;
		//現在のフレーム数をインクリメント
        frame_count++;
        //シミュレートした時間（秒）を計算
        double simulation_seconds = frame_count / static_cast<double>(FPS);

        //現在の状態をMatに描画する
        draw_current_state(display_mat, col_counts, frame_count, elapsed.count(), total_electrons, simulation_seconds, false);

        //録画可能かチェック
        if (writer.isOpened() == true) 
        {
			//1フレーム分のMatを動画に追加
            writer.write(display_mat);
        }
		//スクリーンの描画
        cv::imshow(window_name, display_mat);
        //CPU版は元の計算自体が重いため最短の1msで回す
        cv::waitKey(1); 
    }
	//録画終了
    writer.release();
	//全てのOpenCVウィンドウを閉じる
    cv::destroyAllWindows();

    return true;
}
//---------------------------------------------
//現在の状態をMatに描画する関数（CPU版とCUDA版で共通）
//---------------------------------------------
bool TDoubleSlit::draw_current_state(cv::Mat& canvas, const std::vector<int>& col_counts, int frame_count, double elapsed_time_ms,long long total_electrons, double simulation_seconds,bool is_cuda)
{

	//状態表示用の文字列をC++23のstd::formatで作成
    std::string ss_time  = std::format("Time: {:.1f} s", simulation_seconds);
    std::string ss_count = std::format("Electrons: {}", total_electrons);
    std::string ss_perf  = std::format("{} Loop: {:.1f} ms", is_cuda ? "CUDA" : "CPU", elapsed_time_ms);

	//Matに文字列を描画
    cv::putText(canvas, ss_time.c_str() , cv::Point(20,  40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(canvas, ss_count.c_str(), cv::Point(20,  75), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(canvas, ss_perf.c_str() , cv::Point(20, 110), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255)  , 2, cv::LINE_AA);
    
	return true;
}
//---------------------------------------------
//録画準備
//---------------------------------------------
cv::VideoWriter TDoubleSlit::prepare_video_writer(bool isCuda)
{
	//録画用の動画ファイル名を設定
    video_filename = isCuda ? "simulation_cuda.avi" : "simulation_non_cuda.avi";
	//録画用のVideoWriterオブジェクトを作成
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

	//録画用のVideoWriterオブジェクトを返す
    return cv::VideoWriter(video_filename, fourcc, FPS, cv::Size(WIDTH, HEIGHT));
}
//---------------------------------------------
// (CPU版処理)毎フレーム500個の電子をループでサンプリング
//---------------------------------------------
bool TDoubleSlit::cpu_sampling(std::vector<int>& col_counts)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis_x(-WIDTH / 2.0f, WIDTH / 2.0f);
    std::uniform_real_distribution<float> dis_prob(0.0f, 1.0f);
    for (int i = 0; i < CPU_ELECTRONS_PER_FRAME; ++i)
    {
        while (true)
        {
            float x = dis_x(gen);
            float pi = std::numbers::pi_v<float>;
            float d = static_cast<float>(SLIT_DISTANCE);
            float w = static_cast<float>(SLIT_WIDTH);

            float interference = std::cos(pi * d * x / 800.0f);
            float diffraction = (x == 0.0f) ? 1.0f : std::sin(pi * w * x / 800.0f) / (pi * w * x / 800.0f);
            float intensity = (interference * interference) * (diffraction * diffraction);

            if (dis_prob(gen) < intensity)
            {
                int screen_x = static_cast<int>(x + WIDTH / 2.0f);
                if (screen_x >= 0 && screen_x < WIDTH)
                {
                    col_counts[screen_x]++;
                }
                break;
            }
        }
    }
	return true;
}
//---------------------------------------------
// CPU版の描画スケール調整
//---------------------------------------------
bool TDoubleSlit::cpu_draw_adjustment(cv::Mat& canvas, const std::vector<int>& col_counts, std::mt19937& gen, std::uniform_real_distribution<float>& dis_y)
{
    for (int x = 0; x < WIDTH; ++x)
    {
		// CPU版では描画スケールを調整するために、ヒット数を5で割る
        int current_frame_hits = col_counts[x];
		// CPU版の描画スケール調整
        int dots_to_draw = current_frame_hits / CPU_ADJUSTMENT_DRAW_FACTOR;
		// 最大20個まで描画する
        if (dots_to_draw > MAX_DOTS_TO_DRAW)
        {
            dots_to_draw = MAX_DOTS_TO_DRAW;
        }
        for (int d = 0; d < dots_to_draw; ++d)
        {
            int screen_y = static_cast<int>(dis_y(gen));
            canvas.at<cv::Vec3b>(screen_y, x) = cv::Vec3b(0, 255, 100);
        }
    }
    return true;
}
//---------------------------------------------
// CUDA版の描画スケール調整
//---------------------------------------------
bool TDoubleSlit::cuda_draw_adjustment(cv::Mat& canvas, const std::vector<int>& col_counts, std::mt19937& gen, std::uniform_real_distribution<float>& dis_y)
{
    for (int x = 0; x < WIDTH; ++x)
    {
		// CUDA版では描画スケールを調整するために、ヒット数をCUDA_ADJUSTMENT_DRAW_FACTORで割る
        int current_frame_hits = col_counts[x];
		// CUDA版の描画スケール調整
        int dots_to_draw = current_frame_hits / CUDA_ADJUSTMENT_DRAW_FACTOR;
		// 最大20個まで描画する
        if (dots_to_draw > MAX_DOTS_TO_DRAW)
        {
            dots_to_draw = MAX_DOTS_TO_DRAW;
        }
        for (int d = 0; d < dots_to_draw; ++d)
        {
            int screen_y = static_cast<int>(dis_y(gen));
            canvas.at<cv::Vec3b>(screen_y, x) = cv::Vec3b(0, 255, 100);
        }
    }
    return true;
}
