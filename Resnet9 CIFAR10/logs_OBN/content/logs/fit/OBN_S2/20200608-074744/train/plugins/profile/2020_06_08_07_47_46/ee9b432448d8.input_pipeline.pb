$	t��*�Y@��C�N�@�O=ҼX@!��j,�[@$	�A'���?O��6{��?5��;\ϥ?!�m%.=@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��j,�[@+hZbe4-@1��IW@AB��=Њ�?I!"5�bz@Y�/EH��?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8Y���RyZ@�-v��@1O���*7W@A�r۾G��?I%���?Y��Dh�@"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8s�9>ZY@�,σ��@1d> ЙW@Au�yƾd�?I���@Yi���?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails86�U��"Z@4I,)w!@1�P��W@A�\n0�a�?I�y�Cn�@YJ/�ͦ?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8 ���Y@��iO�9@1��y�4W@A�9z��@I� �bG��?Y���;3�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8�º��Y@\���@1h�K6DW@A��P����?I<���.@Y���GS�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8���p]Y@m����@1"��3�*W@Ap\�M��?IN�q�*	@Y|�(B�v�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8cE�apY@ `��51@1�,��&W@A��5[yI�?I�A]�P��?Y���I��?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8#M�<HY@�ѓ2)@1�H�}'W@A�y���?I��e@Y�7j��{�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8	Ad�&�wY@4�s��u@1��$W@AA�>�D�?Iq:	@Y2U0*��?"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/
�O=ҼX@�w�~�@1�
�7W@I���X�?Y�ȓ�k��?*	��(\�-A2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator
<�8bM3�@!��j==�X@)<�8bM3�@1��j==�X@:Preprocessing2P
Iterator::Model::Prefetch*��Dh@!V��sH��?)*��Dh@1V��sH��?:Preprocessing2F
Iterator::Model�R�h�@!3�s[�C�?)幾	�?15�=��߇?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap
�5x_�3�@!�Iox�X@)����O��?1��~�m?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*moderate2A5.9 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	�&��[V@wN�'|	@�,σ��@!+hZbe4-@	!       "$	��n��(W@R�
)�D�?d> ЙW@!��IW@*	!       2	0e+[b��?�[T��?!�9z��@:$	�3XP@�E�Et�?%���?!N�q�*	@B	!       J$	&8�H�?�g���?|�(B�v�?!��Dh�@R	!       Z$	&8�H�?�g���?|�(B�v�?!��Dh�@JGPU