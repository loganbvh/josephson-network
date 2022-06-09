# josephson-network

Installing Ipopt with HSL linear solvers:

```bash
conda create -n jjnetwork python=3.9
conda activate jjnetwork
conda install -c conda-forge gfortran ipopt

cd /path/to/coinhsl-2021.05.05/

./configure \
    --prefix="/home/<username>/miniconda3/envs/jjnetwork/" \
    --with-blas="-L/home/<username>/miniconda3/envs/jjnetwork/lib/ -lblas" \
    LIBS="-llapack" \
    FC="/home/<username>/miniconda3/envs/jjnetwork/bin/gfortran" \
    CC="/home/<username>/miniconda3/envs/jjnetwork/bin/gcc"

make
make install
# Check installation
ldd /home/<username>/miniconda3/envs/jjnetwork/lib/libcoinhsl.so
# Create a copy of libcoinhsl.so called libhsl.so
cp /home/<username>/miniconda3/envs/jjnetwork/lib/libcoinhsl.so \
    /home/<username>/miniconda3/envs/jjnetwork/lib/libhsl.so
# Check installation
ldd /home/<username>/miniconda3/envs/jjnetwork/lib/libhsl.so

cd /path/to/josephson-network/
pip install -r requirements.txt
```