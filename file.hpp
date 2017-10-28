#ifndef FILE_HPP
#define FILE_HPP
#include <cstddef>
#include <cstdint>
#include <memory>

class wav_file {
public:
    wav_file(const char* name);
private:
#ifndef _WIN32
#else
    void* file;
    void* map;
    void* base;
#endif
    bool big_endian;
public:
    std::uint16_t fmt_flag;
    std::uint16_t channels;
    unsigned sample_rate;
    unsigned byte_rate;
    std::uint16_t bytes_per_sample;
    std::uint16_t bit_depth;
private:
    const void* data;
    std::size_t chunk_size;
    const void* cursor;
public:
    std::size_t size_of(int ms) {
        int records = ms * sample_rate;
        int corrected = records / 1000;
        int remainder = records % 1000;
        if (remainder >= 500) ++records;
        return corrected;
    }
    template <typename T>
    void get(int ms, T buf[]) {
        int corrected = size_of(ms);
        switch (bit_depth) {
        case 32:
        {
            const std::int32_t* cursor = reinterpret_cast<const std::int32_t*>(this->cursor);
            for (int i = 0; i != corrected; ++i, ++cursor)
                buf[i] = T(*cursor);
            break;
        }
        case 16:
            const std::int16_t* cursor = reinterpret_cast<const std::int16_t*>(this->cursor);
            for (int i = 0; i != corrected; ++i, ++cursor)
                buf[i] = T(*cursor);
            break;
        }
    }
    const void* get() {
        return data;
    }
    void jump_to(unsigned sec, unsigned ms) {
        int corrected = sec * sample_rate + size_of(ms);
        cursor = static_cast<const char*>(data) + corrected * bytes_per_sample;
    }
    void inc(unsigned ms) {
        cursor = static_cast<const char*>(cursor) + size_of(ms) * bytes_per_sample;
    }
    unsigned tell() const {
        unsigned ret = reinterpret_cast<const char*>(cursor) - 
                       reinterpret_cast<const char*>(data);
        ret = ret * 1000 / byte_rate;
        return ret;
    }
    ~wav_file();
};

class bmp_file {
    std::int32_t width, awidth;
    std::int32_t height, aheight;
	std::size_t size;
	int* base;
public:
    bmp_file(int, int);
private:
    void* data;
public:
    void use_array(unsigned char ar[], int sz) {
        if (sz > width)
            sz = width;
        for (int i = 0; i != aheight; ++i)
            for (int j = 0; j != sz; ++j)
                reinterpret_cast<std::uint8_t*>(data)[i * awidth + j] = ar[j];
    }
    void save(const char* filename);
    ~bmp_file() {
        delete base;
    }
};
#endif
