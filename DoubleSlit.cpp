#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <chrono>
#include <numbers>
#include <opencv2/opencv.hpp>

// OpenCV 5.0.0 リンク用設定
#ifdef _DEBUG
#pragma comment(lib, "opencv_world500d.lib")
#else
#pragma comment(lib, "opencv_world500.lib")
#endif

// CUDA関数の前方宣言
extern "C" void run_cuda_simulation(int* h_counts, int width, int num_electrons, float d, float w, unsigned long seed);

const int WIDTH = 1000;
const int HEIGHT = 500;
int g_slit_distance = 30;
int g_slit_width = 5;

void on_trackbar(int, void*) {}

int main() 
{
    std::string window_name = "Double Slit Simulation (C++23 + OpenCV + CUDA)";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Slit Distance (d)", window_name, &g_slit_distance, 100, on_trackbar);
    cv::setTrackbarMin("Slit Distance (d)", window_name, 10);

    cv::Mat canvas = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
    std::vector<int> col_counts(WIDTH, 0);

    // 点の描画用にCPU側で使う乱数（Y座標用）
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis_y(0.0f, static_cast<float>(HEIGHT));

    //1フレームあたりの電子数を5万個に大幅増量！
    const int ELECTRONS_PER_FRAME = 50000;

    std::cout << "CUDAシミュレーションを開始します。ESCキーで終了。\n";

    while (true) 
    {
        auto start = std::chrono::high_resolution_clock::now();

        // 1. CUDAを呼び出してX座標の着弾カウントを更新 (毎フレーム5万個)
        run_cuda_simulation(col_counts.data(), WIDTH, ELECTRONS_PER_FRAME,
            static_cast<float>(g_slit_distance), static_cast<float>(g_slit_width), rd());

        // 電子の蓄積エフェクト（グラデーション描画）
        // 毎回画面をクリアせず、着弾カウントの多さに応じてリアルタイムに輝度を計算
        for (int y = 0; y < HEIGHT; ++y) 
        {
            for (int x = 0; x < WIDTH; ++x) 
            {
                // そのX座標に衝突した電子の総数
                int count = col_counts[x];

                // 衝突数に応じて輝度（明るさ）を決定。最大255で頭打ちにする
                // 割る数（ここでは15）を大きくすると、縞が浮かび上がるスピードがゆっくりになります
                int brightness = (count / 15);
                if (brightness > 255) brightness = 255;

                // ランダムにパチパチと点を灯らせる（実際のフィルム風エフェクト）
                // 衝突確率が高いX座標ほど、高い確率で緑の点が点灯する
                if (gen() % 300 < count / 20) 
                {
                    canvas.at<cv::Vec3b>(y, x) = cv::Vec3b(0, brightness, 0); // 衝突数に応じた明るさの緑
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::string title = window_name + " | GPU Time: " + std::to_string(elapsed.count()) + " ms";
        cv::setWindowTitle(window_name, title);

        // 下部に統計グラフを描画
        cv::Mat display_mat = canvas.clone();

        //グラフが天井にぶつからないように自動スケール調整
        //現在のカウントの最大値を探す
        int max_count = 1;
        for (int x = 0; x < WIDTH; ++x) 
        {
            if (col_counts[x] > max_count) 
            {
                max_count = col_counts[x];
            }
        }
        for (int x = 0; x < WIDTH - 1; ++x) 
        {
            // 最大値に基づいてグラフの高さを画面の下部1/4（HEIGHT/4）に綺麗に収める
            int y1 = HEIGHT - (col_counts[x]   * (HEIGHT / 4) / max_count);
            int y2 = HEIGHT - (col_counts[x+1] * (HEIGHT / 4) / max_count);
            cv::line(display_mat, cv::Point(x, y1), cv::Point(x + 1, y2), cv::Scalar(255, 255, 255), 1);
        }

        cv::imshow(window_name, display_mat);

        int key = cv::waitKey(16); // ~60fps
        if (key == 27) 
        {
            break;
        }
        static int last_d = g_slit_distance;
        if (g_slit_distance != last_d) 
        {
            canvas = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
            std::fill(col_counts.begin(), col_counts.end(), 0);
            last_d = g_slit_distance;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
