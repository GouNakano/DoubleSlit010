#include <print>
#include "TDoubleSlit.h"

//---------------------------------------------
//main関数
//---------------------------------------------
int main(int argc, char** argv)
{
	//CUDAを使うかのフラグ（true: CUDA版 / false: CPU単体版）
//    bool is_use_cuda = false;
	bool is_use_cuda = true;
	//二重スリットオブジェクトの作成
	TDoubleSlit double_slit;

	//開始メッセージ表示 
	std::print("電子の二重スリット実験シミュレーション\n");
	std::print("動画として5.0秒分を録画します\n");
	std::print("Enterキーで開始\n");
	getchar();

	if (is_use_cuda == true)
	{
		// CUDA対応版
		// CUDAを用いて処理するC++側処理
		double_slit.use_cuda_cpp_func();
		// CUDAシミュレーション終了時の後処理（GPUメモリ解放）
		double_slit.end_cuda_simulation();
	}
	else
	{
		// CPU単体版
		double_slit.none_cuda_cpp_func(); 
	}
	return 0;
}

