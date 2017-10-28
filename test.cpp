#include "file.hpp"
#include "fft.hpp"
#include <cmath>
#include <cassert>

int main() {
    bmp_file bmp(210, 100);
    unsigned char a[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    bmp.use_array(a, 16);
    bmp.save("C:\\Users\\Yifei Zheng\\Desktop\\b.bmp");
    return 0;
}

int main2(int argc, const char** argv) {
    wav_file wav(argv[1]);
    double buf[2206];
    double wsave[get_ret_size(2205)];
    wav.jump_to(5, 0);
    wav.get(50, buf + 1);
    npy_rffti(2205, wsave);
    npy_rfftf(2205, buf + 1, wsave);
    buf[0] = buf[1];
    buf[1] = 0;
    for (int i = 0; i * 20 < 4200; ++i)
        buf[i] = std::log2(std::pow(buf[i * 2], 2) + std::pow(buf[i * 2 + 1], 2)) / 2;
    for (int i = 0; i * 20 < 4200; ++i)
        printf("%.1f, ", buf[i]);
    return 0;
}