�$	p��CZ@�c�P�q@ޓ��Z-X@!�*��p�\@	����, @G�~�G�@!?��	 @"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6���?Y@/��$�@1]����6W@A�Ov3#�?I�Q�G�@Y
,�)�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8��7�s[@nh�N?0(@18gDi8W@A�(	����?IM֨�@Y8��w��?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8$a�N"
Y@&�\R��@1�s��W@A��I�2�?I���N@Y�{�_���?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8�K�[�Z@����q@1�:�/K$W@At��q5r�?I�j��/@Y�W��I�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8-%�I(�Z@_|�/d"@1s�]��(W@AN�@�C
�?I��a����?Y�Pi���@"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8��QI��Y@�����^@1<��X�W@A�熦���?I*�TPQ@YǛ��,�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8�*��p�\@�-s�d#@1�z1��V@A;�I/�?I����� @Y�1>�^N"@"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8lMK�IZ@d�bӚ@1�N[#�W@A��8+��?IPn�����?Yy�ZO@"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8(��я�Z@ۤ���@13��('W@A�vۅ�:�?I�m��) @Y�nK�s@"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8	:�`���Z@pz��e%@1����W@A~��g��?I�V{�@Y�����?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
ޓ��Z-X@Z�b+h��?1t(CUL W@I+���@*	�I
;1A2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator��&܋Q�@!E�c��X@)��&܋Q�@1E�c��X@:Preprocessing2P
Iterator::Model::Prefetch��P�4@!Y���L�?)��P�4@1Y���L�?:Preprocessing2F
Iterator::ModelE�a��4@!�����x�?)+���ڧ�?1ൾ�,m�?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapU���Q�@!�t4�X@)ys�V{أ?1\&`}�l?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 2.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*moderate2A6.7 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	@�U�K@��ce�	@Z�b+h��?!nh�N?0(@	!       "$	���W@ݦ�o��?�z1��V@!8gDi8W@*	!       2	��WCP�?���!�a�?!t��q5r�?:$	��I�E�@]И ���?��a����?!*�TPQ@B	!       J	�����@���ڂ@!�1>�^N"@R	!       Z	�����@���ڂ@!�1>�^N"@JGPU�"^
4gradient_tape/model_4/conv2d_33/Conv2DBackpropFilterConv2DBackpropFilter	���"��?!	���"��?"4
model_4/conv2d_33/Conv2DConv2D	DF�ҭ?!�iT۱��?"\
3gradient_tape/model_4/conv2d_33/Conv2DBackpropInputConv2DBackpropInput�oM��߫?!sŧЭ��?"4
model_4/conv2d_36/Conv2DConv2D;:uN%�?!�%�����?"^
4gradient_tape/model_4/conv2d_36/Conv2DBackpropFilterConv2DBackpropFilterD�܁�?!��~�p%�?"\
3gradient_tape/model_4/conv2d_36/Conv2DBackpropInputConv2DBackpropInput2���p�?!0�T���?"b
8gradient_tape/model_4/batch_norm_25/FusedBatchNormGradV3FusedBatchNormGradV3, ���e�?!6��
>��?"^
4gradient_tape/model_4/conv2d_34/Conv2DBackpropFilterConv2DBackpropFilter3SSbc�?!i�/t��?"^
4gradient_tape/model_4/conv2d_35/Conv2DBackpropFilterConv2DBackpropFilter��\��b�?!��{n�L�?"4
model_4/conv2d_34/Conv2DConv2D<8!�ܗ?!g���h��?2blackQ      Y@"�
device�Your program is NOT input-bound because only 2.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nomoderate"A6.7 % of the total step time sampled is spent on All Others time.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 