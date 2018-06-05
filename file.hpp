#ifndef FILE_HPP
#define FILE_HPP
#include <cstddef>
#include <cstdint>
#include <memory>

class wav_file {
#ifndef _WIN32
    typedef const char* fn_t;
#else
    typedef const wchar_t* fn_t;
#endif
public:
    wav_file(fn_t name);
private:
#ifndef _WIN32
    int fd;
    std::size_t sz;
    void* map;
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
    const void* data_end;
    const void* cursor;
public:
    std::size_t size_of(unsigned ms) const {
        int records = ms * sample_rate;
        int corrected = records / 1000;
        int remainder = records % 1000;
        if (remainder >= 500) ++records;
        return corrected;
    }
    template <typename T>
    bool get(unsigned ms, T buf[]) {
        int corrected = size_of(ms);
        switch (bit_depth) {
        case 32:
        {
            const std::int32_t* cursor = static_cast<const std::int32_t*>(this->cursor);
            if (cursor + corrected > data_end)
                return 1;
            for (int i = 0; i != corrected; ++i, ++cursor)
                buf[i] = T(*cursor);
            return 0;
        }
        case 16:
        {
            const std::int16_t* cursor = static_cast<const std::int16_t*>(this->cursor);
            if (cursor + corrected > data_end)
                return 1;
            for (int i = 0; i != corrected; ++i, ++cursor)
                buf[i] = T(*cursor);
            return 0;
        }
        }
        return 1;
    }
    bool jump_to(unsigned ms) {
        cursor = static_cast<const char*>(data) + size_of(ms) * bytes_per_sample;
        if (cursor > data_end)
            return 1;
        return 0;
    }
    bool inc(unsigned ms) {
        cursor = static_cast<const char*>(cursor) + size_of(ms) * bytes_per_sample;
        if (cursor > data_end)
            return 1;
        return 0;
    }
    unsigned tell() const {
        unsigned ret = reinterpret_cast<const char*>(cursor) - 
                       reinterpret_cast<const char*>(data);
        ret = ret * 1000 / byte_rate;
        return ret;
    }
    ~wav_file();
};
#endif
