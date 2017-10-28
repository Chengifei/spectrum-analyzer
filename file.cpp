#include <cstdio>
#include <cassert>
#include <cstring>
#ifndef _WIN32
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/fcntl.h>
#include <unistd.h>
#else
#include <windows.h>
#endif
#include "file.hpp"

template <typename read_t, typename ret_t = read_t>
ret_t read(const char*& ptr) {
    const char* _ = ptr;
    ptr += sizeof(read_t);
    return *reinterpret_cast<const read_t*>(_);
}

wav_file::wav_file(const char* filename) {
#ifndef _WIN32
    struct stat sb;
    int wav = open(filename, O_RDONLY);
    fstat(wav, &sb);
    const char* data = static_cast<const char*>(mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, wav, 0));
#else
    file = CreateFile(filename, GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    map = CreateFileMapping(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    base = MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0);
    const char* data = static_cast<const char*>(base);
#endif
    if (memcmp("RIFF", data, 4) == 0)
        big_endian = false;
    else if (memcmp("RIFX", data, 4) == 0)
        big_endian = true;
    else
        throw /* something */;
    data += 4;
    /* ret.file_size = */ read<std::uint32_t>(data) + 8; // Ignore the file size
    if (memcmp("WAVE", data, 4) || memcmp("fmt ", data + 4, 4))
        throw /* something */;
    data += 8;
    std::size_t chunk_size = read<std::uint32_t>(data);
    fmt_flag = read<std::uint16_t>(data);
    channels = read<std::uint16_t>(data);
    sample_rate = read<std::uint32_t>(data);
    byte_rate = read<std::uint32_t>(data);
    bytes_per_sample = read<std::uint16_t>(data);
    bit_depth = read<std::uint16_t>(data);
    data += chunk_size - 16;
    while (memcmp("data", data, 4))
        data += 8 + *reinterpret_cast<const std::uint32_t*>(data + 4);
    data += 4;
    this->chunk_size = read<std::uint32_t>(data);
    this->data = data;
    cursor = data;
}

wav_file::~wav_file() {
#ifndef _WIN32
#else
    UnmapViewOfFile(base);
    CloseHandle(map);
    CloseHandle(file);
#endif
}

template <typename T, typename P>
void write(P*& ptr, T t) {
    *reinterpret_cast<T*>(ptr) = t;
    ptr = reinterpret_cast<P*>(reinterpret_cast<char*>(ptr) + sizeof(T));
}

bmp_file::bmp_file(int w, int h) : width(w), height(h) {
    awidth = w + w % 4;
    aheight = h + h % 4;
    size = awidth * aheight + 1078;
    base = new int[size]{};
    char* data = reinterpret_cast<char*>(base);
    memcpy(data, "BM", 2);
    data += 2;
    write<std::uint32_t>(data, size); // file_size
    data += 4; // skip reserved stuff
    write<std::uint32_t>(data, 1078); // offset_to_pixels
    write<std::uint32_t>(data, 40); // size_of_dib_header
    write<std::int32_t>(data, w);
    write<std::int32_t>(data, h);
    write<std::uint16_t>(data, 1); // plane
    write<std::uint16_t>(data, 8); // bits per pixel
    write<std::uint32_t>(data, 0); // compression
    write<std::uint32_t>(data, awidth * aheight);
    write<std::int32_t>(data, 3779); // hres
    write<std::int32_t>(data, 3779); // vres
    write<std::int32_t>(data, 256); // colors in palatte
    write<std::int32_t>(data, 256); // important colors
    for (int i = 0; i != 256; ++i) {
        write<std::uint8_t>(data, i);
        write<std::uint8_t>(data, i);
        write<std::uint8_t>(data, i);
        write<std::uint8_t>(data, i);
    }
    this->data = data;
}

void bmp_file::save(const char* filename) {
#ifndef _WIN32
    int bmp = open(filename, O_WRONLY);
    write(bmp, base, size);
    close(bmp);
#else
    void* file = CreateFile(filename, GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                      CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    SetFilePointer(file, size, nullptr, FILE_BEGIN);
    SetEndOfFile(file);
    void* map = CreateFileMapping(file, nullptr, PAGE_READWRITE, 0, 0, nullptr);
    void* base = MapViewOfFile(map, FILE_MAP_WRITE, 0, 0, 0);
    memcpy(base, this->base, size);
    UnmapViewOfFile(base);
    CloseHandle(map);
    CloseHandle(file);
#endif
}
