�$	W����nY@y�0���?�b)���X@!�m��NZ@$	A�~.�g�?Q��k%�?p]�pkZ�?!O��먹@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�����Y@d���^�@1�e�/W@AE���V��?I)狽�@Y�cϞ��?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8��]izY@����_�@1ZGUDW@A�qs*�?IK����?Y�ݒ��@"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8s�}��Y@�;pϣ@1~����5W@A]p��?IP÷�n��?Y�$�jr�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8�o�[Y@�Xl��@1 �={.W@A�{�E{��?I�V��,s@Ye�u��?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8�m��NZ@|~!<*"@1:�6U�#W@A�����?I��0 @Y��Xl���?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8�b)���X@C���@1p��:�W@A%!��q�?I������?Y0��L�^�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8��iܛ�Y@Q���J/@1��L�W@At@����?I�nض@Y�z�Fw�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8wLݕ]�Y@+��t@1#�M)�W@A��v��?I����@Y�,��ot@"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails82�Mc{ Y@v���;
@1���h6W@Aު�PM��?Ie6�$c@Y�����?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8	�����X@��U�P�@1���1W@A6x_��?I�cҟ@Y>$|�o��?"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/
.7�zY@ ����@1�G�R W@I_D�1uW�?Y���|��?*	�/]0A2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator˹W�k�@!�L�|�X@)˹W�k�@1�L�|�X@:Preprocessing2P
Iterator::Model::Prefetch^��-�a@!��$@���?)^��-�a@1��$@���?:Preprocessing2F
Iterator::Model?�g͏�@!3�����?):<��Ӹ�?1 i)i�?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap�ZDl�@!�qw[5�X@)�\�E�?1}��>og?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*moderate2A5.2 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	�e�<,@�9cqK�?C���@!|~!<*"@	!       "$	w�W@>`��?ZGUDW@!���h6W@*	!       2	�a�5
3�?az����?!t@����?:$	��2� @�Vb�?_D�1uW�?!)狽�@B	!       J$	�m�����? �T����?�����?!�,��ot@R	!       Z$	�m�����? �T����?�����?!�,��ot@JGPU�"^
4gradient_tape/model_2/conv2d_17/Conv2DBackpropFilterConv2DBackpropFilter<��Y���?!<��Y���?"4
model_2/conv2d_17/Conv2DConv2D�V�r��?!�G�	��?"\
3gradient_tape/model_2/conv2d_17/Conv2DBackpropInputConv2DBackpropInput�.����?!tS!,}��?"4
model_2/conv2d_20/Conv2DConv2D3t����?!@x�~�?"^
4gradient_tape/model_2/conv2d_20/Conv2DBackpropFilterConv2DBackpropFilter6<㒾�?!��g�� �?"\
3gradient_tape/model_2/conv2d_20/Conv2DBackpropInputConv2DBackpropInput��l�o�?!�貉��?"b
8gradient_tape/model_2/batch_norm_17/FusedBatchNormGradV3FusedBatchNormGradV3­�qd[�?!��A���?"^
4gradient_tape/model_2/conv2d_18/Conv2DBackpropFilterConv2DBackpropFilter���,af�?!H��S\��?"^
4gradient_tape/model_2/conv2d_19/Conv2DBackpropFilterConv2DBackpropFilter�t�y�]�?!�ȃ�5F�?"4
model_2/conv2d_19/Conv2DConv2D�Sckkܗ?!��9����?2blackQ      Y@"�
device�Your program is NOT input-bound because only 1.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nomoderate"A5.2 % of the total step time sampled is spent on All Others time.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 