	P��5sY@P��5sY@!P��5sY@	/�{��?/�{��?!/�{��?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-P��5sY@0fKVE��?1�fהX@IC�Գ ��?Y9�]��?*	m�����F@2F
Iterator::Model�a�o�?!      Y@)���I'�?1��|�iU@:Preprocessing2P
Iterator::Model::Prefetch���=�z?!5h�2�,@)���=�z?15h�2�,@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	0fKVE��?0fKVE��?!0fKVE��?      ��!       "	�fהX@�fהX@!�fהX@*      ��!       2      ��!       :	C�Գ ��?C�Գ ��?!C�Գ ��?B      ��!       J	9�]��?9�]��?!9�]��?R      ��!       Z	9�]��?9�]��?!9�]��?JGPU