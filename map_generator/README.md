# Download LoFTR
rm -rf sample_data
git clone https://github.com/zju3dv/LoFTR --depth 1
mv LoFTR/* . && rm -rf LoFTR

# Download pretrained weights
mkdir weights
cd weights/
gdown --id 1w1Qhea3WLRMS81Vod_k5rxS_GNRgIi-O  # indoor-ds
gdown --id 1M-VD35-qdB5Iw-AtbDBCKC7hPolFW9UY  # outdoor-ds
cd ..