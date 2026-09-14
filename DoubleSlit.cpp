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

using namespace std::chrono;


// CUDA関数の前方宣言(カーネル関数呼び出しなどのCUDA独自の処理)
void run_cuda_simulation(int* h_counts, int width, int num_electrons, float d, float w, unsigned long seed);
//CUDAを用いて処理するC++側処理
void use_cuda_cpp_func();
//CUDAなしの処理
void none_cuda_cpp_func();

//描画スクリーンのサイズ
constexpr const int WIDTH = 1000;
constexpr const int HEIGHT = 500;
//1秒あたりのフレーム数
constexpr const int FPS = 30;
//録画秒数
constexpr const int RECORD_SECONDS = 5;

//1フレームあたりの電子数
constexpr const int GPU_ELECTRONS_PER_FRAME = 50000; // GPUは5万個一瞬
constexpr const int CPU_ELECTRONS_PER_FRAME = 500;   // CPUは500個で滑らかに動かす


int g_slit_distance = 30;
int g_slit_width = 5;

void on_trackbar(int, void*) {}

//---------------------------------------------
//main関数
//---------------------------------------------
int main(int argc, char** argv)
{
	//CUDAを使うかのフラグ（true: CUDA版 / false: CPU単体版）
    bool is_use_cuda = false;

    //開始メッセージ表示 
    std::print("電子の二重スリット実験シミュレーション\n");
    std::print("動画として【ぴったり150フレーム(5.0秒分)】を確実に録画します\n");
    std::print("Enterキーで開始\n");
    getchar();

    if (is_use_cuda == true)
    {
        use_cuda_cpp_func(); // CUDA対応版
    }
    else
    {
        none_cuda_cpp_func(); // CPU単体版
    }

    cv::destroyAllWindows();
    return 0;
}

//---------------------------------------------
//CUDAを用いて処理するC++側処理
//---------------------------------------------
void use_cuda_cpp_func()
{
    std::string window_name = "Double Slit Simulation (C++23 + OpenCV + CUDA)";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    cv::Mat canvas = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis_y(0.0f, static_cast<float>(HEIGHT));
    std::vector<int> col_counts(WIDTH, 0);

    //録画準備
    std::string video_filename = "simulation_cuda.avi";
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    cv::VideoWriter writer(video_filename, fourcc, FPS, cv::Size(WIDTH, HEIGHT));

    if (writer.isOpened() == false)
    {
        std::print("エラー: 動画ファイルを開けませんでした。\n");
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
        
        // 1. CUDAを呼び出してX座標の着弾カウントを更新
        run_cuda_simulation(col_counts.data(), WIDTH, GPU_ELECTRONS_PER_FRAME, static_cast<float>(g_slit_distance), static_cast<float>(g_slit_width), rd());

        auto calc_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = calc_end - calc_start;

        // 確率に応じた比率でピンポイント描画（300で割るカスタム版）
        for (int x = 0; x < WIDTH; ++x)
        {
            int current_frame_hits = col_counts[x];
            int dots_to_draw = current_frame_hits / 300;

            if (dots_to_draw > 20) 
            {
                dots_to_draw = 20;
            }
            for (int d = 0; d < dots_to_draw; ++d)
            {
                int screen_y = static_cast<int>(dis_y(gen));
                canvas.at<cv::Vec3b>(screen_y, x) = cv::Vec3b(0, 255, 100);
            }
        }
        std::string title = window_name + " | GPU Time: " + std::to_string(elapsed.count()) + " ms";
        cv::setWindowTitle(window_name, title);
        //スクリーン描画用Mat
        cv::Mat display_mat = canvas.clone();

        //現在の録画フレーム数から、動画の再生時間（秒）を計算
        frame_count++;
        long long total_electrons = (long long)frame_count * GPU_ELECTRONS_PER_FRAME;
        double simulation_seconds = frame_count / 30.0;

        std::stringstream ss_time, ss_count, ss_perf;
        ss_time << "Time: " << std::fixed << std::setprecision(1) << simulation_seconds << " s";
        ss_count << "Electrons: " << total_electrons;
        ss_perf << "GPU Kernels: " << std::fixed << std::setprecision(1) << elapsed.count() << " ms";

        cv::putText(display_mat, ss_time.str(), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        cv::putText(display_mat, ss_count.str(), cv::Point(20, 75), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        cv::putText(display_mat, ss_perf.str(), cv::Point(20, 110), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

        //録画が出来るならフレーム追加
        if (writer.isOpened() == true) 
        {
            writer.write(display_mat);
        }
		//スクリーンの描画
        cv::imshow(window_name, display_mat);

        //30fps（約33ms）に近い速度で画面をプレビュー表示させる
        cv::waitKey(30);
    }
    writer.release();
}
//---------------------------------------------
//CUDAなしの処理（C++23標準 CPUシングルスレッド処理）
//---------------------------------------------
void none_cuda_cpp_func()
{
    std::string window_name = "Double Slit Simulation (C++23 + OpenCV | CPU-Only)";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    cv::Mat canvas = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
    std::vector<int> col_counts(WIDTH, 0);

	//録画準備
    std::string video_filename = "simulation_non_cuda.avi";
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    double fps = 30.0;
    cv::VideoWriter writer(video_filename, fourcc, fps, cv::Size(WIDTH, HEIGHT));
    //録画可能かチェック
    if (!writer.isOpened())
    {
        std::print("エラー: 動画ファイルを開けませんでした。\n");
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis_x(-WIDTH / 2.0f, WIDTH / 2.0f);
    std::uniform_real_distribution<float> dis_y(0.0f, static_cast<float>(HEIGHT));
    std::uniform_real_distribution<float> dis_prob(0.0f, 1.0f);

    int frame_count = 0;

    //メインループ
    while (true)
    {
        // 🌟【修正】CPU版も150フレームに達したらカチッと終了する
        if (frame_count >= 150)
        {
            break;
        }

        std::fill(col_counts.begin(), col_counts.end(), 0);

        auto calc_start = high_resolution_clock::now();

        // 【CPU版処理】毎フレーム5万個の電子をループでサンプリング
        for (int i = 0; i < CPU_ELECTRONS_PER_FRAME; ++i)
        {
            while (true)
            {
                float x = dis_x(gen);
                float pi = std::numbers::pi_v<float>;
                float d = static_cast<float>(g_slit_distance);
                float w = static_cast<float>(g_slit_width);

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

        auto calc_end = high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = calc_end - calc_start;

        // CPU版の描画スケール調整（/ 5）
        for (int x = 0; x < WIDTH; ++x)
        {
            int current_frame_hits = col_counts[x];
            int dots_to_draw = current_frame_hits / 5;

            if (dots_to_draw > 20) 
            {
                dots_to_draw = 20;
            }
            for (int d = 0; d < dots_to_draw; ++d)
            {
                int screen_y = static_cast<int>(dis_y(gen));
                canvas.at<cv::Vec3b>(screen_y, x) = cv::Vec3b(0, 255, 100);
            }
        }

        std::string title = window_name + " | CPU Time: " + std::to_string(elapsed.count()) + " ms";
        cv::setWindowTitle(window_name, title);
		//スクリーン描画用Mat
        cv::Mat display_mat = canvas.clone();
        int fixed_max_scale = 50;

        //現在の録画フレーム数から動画の再生時間(秒)を計算
        frame_count++;
        long long total_electrons = (long long)frame_count * CPU_ELECTRONS_PER_FRAME;
        double simulation_seconds = frame_count / 30.0;

        std::stringstream ss_time, ss_count, ss_perf;
        ss_time << "Time: " << std::fixed << std::setprecision(1) << simulation_seconds << " s";
        ss_count << "Electrons: " << total_electrons;
        ss_perf << "CPU Loop: " << std::fixed << std::setprecision(1) << elapsed.count() << " ms";

        cv::putText(display_mat, ss_time.str(), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        cv::putText(display_mat, ss_count.str(), cv::Point(20, 75), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        cv::putText(display_mat, ss_perf.str(), cv::Point(20, 110), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        //録画可能かチェック
        if (writer.isOpened() == true) 
        {
			//1フレーム分のMatを動画に追加
            writer.write(display_mat);
        }
        cv::imshow(window_name, display_mat);
        cv::waitKey(1); //CPU版は元の計算自体が重いため最短の1msで回す
    }
	//録画終了
    writer.release();
}
