TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#g++ -std=c++11 -shared field_embed_op.cc -o field_embed.so -fPIC -I ${TF_INC} -O2 -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 -shared field_embed_op.cc -o field_embed.so -fPIC -I ${TF_INC} -O2
