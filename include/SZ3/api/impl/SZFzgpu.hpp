#ifndef SZ3_SZ_FZGPU_HPP
#define SZ3_SZ_FZGPU_HPP

#include "SZ3/compressor/SZIterateCompressor.hpp"
#include "SZ3/compressor/SZGenericCompressor.hpp"
#include "SZ3/decomposition/LorenzoRegressionDecomposition.hpp"
#include "SZ3/quantizer/IntegerQuantizer.hpp"
#include "SZ3/predictor/ComposedPredictor.hpp"
#include "SZ3/predictor/LorenzoPredictor.hpp"
#include "SZ3/predictor/RegressionPredictor.hpp"
#include "SZ3/predictor/PolyRegressionPredictor.hpp"
#include "SZ3/lossless/Lossless_zstd.hpp"
#include "SZ3/utils/Iterator.hpp"
#include "SZ3/utils/Statistic.hpp"
#include "SZ3/utils/Extraction.hpp"
#include "SZ3/utils/QuantOptimizatioin.hpp"
#include "SZ3/utils/Config.hpp"
#include "SZ3/def.hpp"

#include <cmath>
#include <memory>

#include "fz_demo.hh"

namespace SZ3 {
    template<class T, uint N>
    size_t SZ_compress_fzgpu(Config &conf, T *data, uchar *cmpData, size_t cmpCap) {

        assert(N == conf.N);
        assert(conf.fzgpu);
        calAbsErrorBound(conf, data);

        int x = conf.dims[0];
        int y = conf.dims[1];
        int z = conf.dims[2];

        double eb = conf.absErrorBound;

        fzgpu::compressor_roundtrip_v0("/home/bozhan/dataset/00_CESM-ATM_yx_1800x3600\=6480000/ICEFRAC_1_1800_3600.f32", 3600, 1800, 1, 1e-3, false);
        fzgpu::compressor_roundtrip_v1("/home/bozhan/dataset/00_CESM-ATM_yx_1800x3600\=6480000/ICEFRAC_1_1800_3600.f32", 3600, 1800, 1, 1e-3, false);
    //     void compressor_roundtrip_v0(
    // std::string fname, int const x, int const y, int const z, double eb,
    // bool use_rel);
    
        return 0;
    }


    template<class T, uint N>
    void SZ_decompress_fzgpu(const Config &conf, const uchar *cmpData, size_t cmpSize, T *decData) {
        assert(conf.fzgpu);

        // auto cmpDataPos = cmpData;
        // LinearQuantizer<T> quantizer;
        // if (N == 3 && !conf.regression2 || (N == 1 && !conf.regression && !conf.regression2)) {
        //     // use fast version for 3D
        //     auto sz = make_compressor_sz_generic<T, N>(make_decomposition_lorenzo_regression<T, N>(conf, quantizer),
        //                                                HuffmanEncoder<int>(), Lossless_zstd());
        //     sz->decompress(conf, cmpDataPos, cmpSize, decData);
        //     return;

        // } else {
        //     auto sz = make_compressor_typetwo_lorenzo_regression<T, N>(conf, quantizer, HuffmanEncoder<int>(), Lossless_zstd());
        //     sz->decompress(conf, cmpDataPos, cmpSize, decData);
        //     return;
        // }

    }
}
#endif