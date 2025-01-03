import tensorflow as tf

# 檢查 TensorFlow 版本
print("TensorFlow 版本:", tf.__version__)

# 檢查 GPU 是否可用
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# 列出所有 GPU 裝置
print("Physical Devices:", tf.config.experimental.list_physical_devices('GPU'))

print(tf.sysconfig.get_build_info()["cuda_version"])  # CUDA 版本
print(tf.sysconfig.get_build_info()["cudnn_version"])  # cuDNN 版本

