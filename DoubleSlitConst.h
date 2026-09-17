#ifndef TDOUBLE_SLIT_CONST_H
#define TDOUBLE_SLIT_CONST_H

//描画スクリーンのサイズ
constexpr const int WIDTH = 560;
constexpr const int HEIGHT = 640;
//1秒あたりのフレーム数
constexpr const int FPS = 30;
//録画秒数
constexpr const int RECORD_SECONDS = 5;
//1フレームあたりの電子数
constexpr const int GPU_ELECTRONS_PER_FRAME = 50000; // GPUは5万個
constexpr const int CPU_ELECTRONS_PER_FRAME = 500;   // CPUは500個で処理の遅さをカバー
//スリットの距離と幅
constexpr const int SLIT_DISTANCE = 30;
constexpr const int SLIT_WIDTH = 5;
//CUDA描画調整係数
constexpr const int CUDA_ADJUSTMENT_DRAW_FACTOR = 300;
//CPU描画調整係数
constexpr const int CPU_ADJUSTMENT_DRAW_FACTOR = 5;	
//最大描画ドット数
constexpr const int MAX_DOTS_TO_DRAW = 20;

#endif
