export https_proxy=http://127.0.0.1:7890/
export http_proxy=http://127.0.0.01:7890/

if [ -d "build" ]; then
    rm -rf build
fi
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DRK3566_PLATFORM=ON
make -j$(nproc)
