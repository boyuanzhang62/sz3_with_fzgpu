//
// Created by Kai Zhao on 4/21/20.
//

#ifndef SZ_MEMORYOPS_HPP
#define SZ_MEMORYOPS_HPP

#include <cassert>
#include <cstring>
#include <string>
#include "SZ3/def.hpp"

namespace SZ3 {
    // read array
    template<class T1>
    void read(T1 *array, size_t num_elements, uchar const *&compressed_data_pos, size_t &remaining_length) {
        assert(num_elements * sizeof(T1) <= remaining_length);
        memcpy(array, compressed_data_pos, num_elements * sizeof(T1));
        remaining_length -= num_elements * sizeof(T1);
        compressed_data_pos += num_elements * sizeof(T1);
    }

    // read array
    template<class T1>
    void read(T1 *array, size_t num_elements, uchar const *&compressed_data_pos) {
        memcpy(array, compressed_data_pos, num_elements * sizeof(T1));
        compressed_data_pos += num_elements * sizeof(T1);
    }


    // read variable
    template<class T1>
    void read(T1 &var, uchar const *&compressed_data_pos) {
        memcpy(&var, compressed_data_pos, sizeof(T1));
        compressed_data_pos += sizeof(T1);
    }

    // read variable
    template<class T1>
    void read(T1 &var, uchar const *&compressed_data_pos, size_t &remaining_length) {
        assert(sizeof(T1) <= remaining_length);
        memcpy(&var, compressed_data_pos, sizeof(T1));
        remaining_length -= sizeof(T1);
        compressed_data_pos += sizeof(T1);
    }

    // write array
    template<class T1>
    void write(T1 const *array, size_t num_elements, uchar *&compressed_data_pos) {
        memcpy(compressed_data_pos, array, num_elements * sizeof(T1));
        compressed_data_pos += num_elements * sizeof(T1);
    }

    // write variable
    template<class T1>
    void write(T1 const var, uchar *&compressed_data_pos) {
        memcpy(compressed_data_pos, &var, sizeof(T1));
        compressed_data_pos += sizeof(T1);
    }

    void write_str(std::string const &var, uchar *&compressed_data_pos) {
        strcpy(reinterpret_cast<char *>(compressed_data_pos), var.c_str());
        compressed_data_pos += var.size() + 1;
    }

    void read_str(std::string &var, uchar const *&compressed_data_pos) {
        var = std::string(reinterpret_cast<const char *>(compressed_data_pos));
        compressed_data_pos += var.size() + 1;
    }
}
#endif //SZ_MEMORYOPS_HPP
