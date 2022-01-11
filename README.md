# curand_fp16

A library for generating pseudo random FP16 numbers

## Throughput

<img src='./docs/throughput.svg'>

## Requirements
- C++ >= 17
- cmake >= 3.18
- CUDA
- cuRAND

## Build
```
mkdir build
cd build
cmake .. --DCMAKE_INSTALL_PREFIX=/path/to/install
make -j4
make install
```

## Sample codes
See [tests](./tests/)

## License
MIT
