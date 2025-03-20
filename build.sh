#!/bin/bash
rm -r CMakeCache.txt CMakeFiles
cmake .
echo 'Building rockchip_npu with stub LAPACK/BLAS libraries'
# Build the project
make -j$(nproc)

# Create a directory for deployment
mkdir -p deploy/bin deploy/lib

# Copy the executable and necessary libraries
cp rockchip_npu deploy/bin/
# The stub liblapack is now statically linked, so no need to copy it.
cp ./3rdparty/static/librknnrt.so deploy/lib/

# Find libraries needed by the executable, excluding problematic ones.
echo 'Copying needed libraries...'

# Try using ldd first
ldd_output=$(ldd rockchip_npu 2>&1)
if echo "$ldd_output" | grep -q "not a dynamic executable"; then
  echo "ldd failed; using aarch64-linux-gnu-readelf instead..."
  libs=$(aarch64-linux-gnu-readelf -d rockchip_npu | grep "Shared library:" | awk -F'[][]' '{print $2}' | grep -v 'liblapack|libblas|libGLX')
else
  libs=$(echo "$ldd_output" | grep '=> /' | awk '{print $3}' | grep -v 'liblapack|libblas|libGLX')
fi

for lib in $libs; do
  if [ -f $lib ]; then
    cp $lib deploy/lib/
    cp $lib deploy/bin/
  fi
done
cp camera-error.png deploy/bin/camera-error.png
rm deploy.zip
zip -9 -r deploy.zip deploy

echo 'Build complete. Deployment package in ./deploy/'
