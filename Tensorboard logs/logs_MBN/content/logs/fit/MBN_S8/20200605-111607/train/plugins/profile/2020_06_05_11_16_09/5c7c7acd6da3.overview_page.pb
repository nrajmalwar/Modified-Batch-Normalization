�	(5
 Y@(5
 Y@!(5
 Y@	4!n��J�?4!n��J�?!4!n��J�?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-(5
 Y@@�j��?1�Q�d$W@I�o'�@Yj���J�?*	    B��@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator����L]Y@!9�C���X@)����L]Y@19�C���X@:Preprocessing2F
Iterator::Model�J��?!*��ç?)�L��~ޤ?1��#���?:Preprocessing2P
Iterator::Model::Prefetch�
F%uz?!����y?)�
F%uz?1����y?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap��E�]Y@!;/��X@)f���-=j?1�� 'a�i?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?5.0 % of the total step time sampled is spent on Kernel Launch.*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	@�j��?@�j��?!@�j��?      ��!       "	�Q�d$W@�Q�d$W@!�Q�d$W@*      ��!       2      ��!       :	�o'�@�o'�@!�o'�@B      ��!       J	j���J�?j���J�?!j���J�?R      ��!       Z	j���J�?j���J�?!j���J�?JGPU�"^
4gradient_tape/model_3/conv2d_25/Conv2DBackpropFilterConv2DBackpropFilter}8�#Ⱥ?!}8�#Ⱥ?"4
model_3/conv2d_25/Conv2DConv2D�.t�ڭ?!������?"\
3gradient_tape/model_3/conv2d_25/Conv2DBackpropInputConv2DBackpropInputJ��UH �?!�� ����?"4
model_3/conv2d_28/Conv2DConv2DI���1J�?!�6����?"^
4gradient_tape/model_3/conv2d_28/Conv2DBackpropFilterConv2DBackpropFilter]Q�8�?!�����=�?"\
3gradient_tape/model_3/conv2d_28/Conv2DBackpropInputConv2DBackpropInputa���e�?!u�P���?"b
8gradient_tape/model_3/batch_norm_25/FusedBatchNormGradV3FusedBatchNormGradV3<mS��G�?!a�F���?"^
4gradient_tape/model_3/conv2d_26/Conv2DBackpropFilterConv2DBackpropFilter|�����?!�)��%��?"^
4gradient_tape/model_3/conv2d_27/Conv2DBackpropFilterConv2DBackpropFilter�;g�J��?!A��Q
i�?"4
model_3/conv2d_27/Conv2DConv2D<� s��?!U��8U��?2blackQ      Y@"�
device�Your program is NOT input-bound because only 1.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate?5.0 % of the total step time sampled is spent on Kernel Launch.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 