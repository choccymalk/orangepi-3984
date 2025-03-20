# orangepi-3984
Code for the orange pi. It hosts a camera server and object detection with depth estimation. \
how tf do i compile ts? \
step 1. download the repository \
step 2. download this pypi package. https://files.pythonhosted.org/packages/dd/b0/26f06f9428b250d856f6d512413e9e800b78625f63801cbba13957432036/torch-2.6.0-cp313-cp313-manylinux_2_28_aarch64.whl \
step 3. extract the wheel. you can use the unzip command. \
step 4. when you extract it, go to torch/lib \
step 5. go to 3rdparty/libtorch and copy everything from the wheel you extracted \
step 6. install aarch64-linux-gnu-gcc and aarch64-linux-gnu-g++ and a few other things needed to cross compile, but i don't remember what you need \
step 7. run build.sh
