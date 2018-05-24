#include <cassert>
#include <cstring>
#ifndef _WIN32
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/fcntl.h>
#include <unistd.h>
#include <err.h>
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
    fd = open(filename, O_RDONLY);
    fstat(fd, &sb);
    sz = sb.st_size;
    map = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED)
        err(-1, "mmap");
    const char* data = static_cast<const char*>(map);
#else
    file = CreateFile(filename, GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    map = CreateFileMapping(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    base = MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0);
    const char* data = static_cast<const char*>(base);
#endif
    if (memcmp("RIFF", data, 4) == 0)
        big_endian = false;
    else if (memcmp("RIFX", data, 4) == 0) {
        big_endian = true;
        throw /* something */;
    }
    else
        throw /* something */;
    data += 4;
    /* ret.file_size = */ read<std::uint32_t>(data) + 8; // Ignore the file size
    if (memcmp("WAVEfmt ", data, 8))
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
	std::size_t data_len = read<std::uint32_t>(data);
	this->data = data;
	this->data_end = data + data_len;
    cursor = data;
}

wav_file::~wav_file() {
#ifndef _WIN32
	if (munmap(map, sz))
		err(-1, "munmap");
	if (close(fd))
		err(-1, "close");
#else
	UnmapViewOfFile(base);
	CloseHandle(map);
	CloseHandle(file);
#endif
}

