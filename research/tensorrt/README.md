# Running the TensorFlow Official ResNet with TensorRT

[TensorRT](https://developer.nvidia.com/tensorrt) is NVIDIA's inference
optimizer for deep learning. Briefly, TensorRT rewrites parts of the
execution graph to allow for faster prediction times.

Here we provide a sample script that can:

1. Convert a TensorFlow SavedModel to a Frozen Graph.
2. Load a Frozen Graph for inference.
3. Time inference loops using the native TensorFlow graph.
4. Time inference loops using FP32, FP16, or INT8 precision modes from TensorRT.

We provide some results below, as well as instructions for running this script.

## Results

TODO(@tfboyd)

## How to Run This Script

### Step 1: Install Prerequisites

[Install TensorFlow.](https://www.tensorflow.org/install/)

Install TensorRT: TODO(@tfboyd)

We use the [Official version of ResNet image preprocessing](/official/resnet/imagenet_preprocessing.py). Please checkout the Models repository if you haven't
already, and add the Official Models to your Python path:

```
cd /path/to
git clone https://github.com/tensorflow/models.git
export PYTHONPATH="$PYTHONPATH:/path/to/models"
```

### Step 2: Get a model to test

The provided script runs with the [Official version of ResNet trained with
ImageNet data](/official/resnet), but can be used for other models as well,
as long as you have a SavedModel or a Frozen Graph.

You can download the ResNet-ImageNet [SavedModel](TODO(karmel)) or
[Frozen Graph](TODO(karmel)), or, if you want to train the model yourself,
pass `--export_dir` to the Official ResNet [imagenet_main.py](/official/resnet/imagenet_main.py).

When running this script, you can pass in a SavedModel directory containing the
Protobuf MetaGraphDef and variables directory to `savedmodel_dir`, or pass in
a Protobuf frozen graph file directly to `frozen_graph`.

### Step 3: Get an image to test

The script can accept a JPEG image file to use for predictions. If none is
provided, random data will be generated. We provide a sample `dog.jpg` here
which can be passed in with the `--image_file` flag.

### Step 4: Run the model

You have TensorFlow, TensorRT, a graph def, and a picture.
Now it's time to time.

For the full set of possible parameters, you can run
`python tensorrt.py --help`. Assuming you used the files provided above,
you would run:

```
python tensorrt.py --frozen_graph=resnetv2_imagenet_frozen_graph.pb \
  --image_file=dog.jpg --native --fp32 --fp16 --int8 --output_dir=/my/output
```

This will print the predictions for each of the precision modes that were run
(native, which is the native precision of the model passed in, as well
as the TensorRT version of the graph at precisions of fp32, fp16, and int8):

```
INFO:tensorflow:Starting timing.
INFO:tensorflow:Timing loop done!
Predictions:
Precision:  native [u'Doberman, Doberman pinscher', u'briard', u'otterhound, otter hound', u'komondor', u'malamute, malemute, Alaskan malamute']
Precision:  FP32 [u'Doberman, Doberman pinscher', u'briard', u'otterhound, otter hound', u'komondor', u'malamute, malemute, Alaskan malamute']
Precision:  FP16 [u'Doberman, Doberman pinscher', u'briard', u'otterhound, otter hound', u'komondor', u'malamute, malemute, Alaskan malamute']
```

The script will generate or append to a file in the output_dir, `log.txt`,
which includes the timing information for each of the models:

```
==========================
network: native_imagenet_frozen_graph.pb,  batchsize 128, steps 100
  fps   median: 670.6,  mean: 664.1,  uncertainty: 6.5,   jitter: 0.8
  latency   median: 0.19089,  mean: 0.25054,  99th_p: 0.25119,  99th_uncertainty: 3.04235

==========================
network: tftrt_fp32_imagenet_frozen_graph.pb,  batchsize 128, steps 100
  fps   median: 822.5,  mean: 814.2,  uncertainty: 8.1,   jitter: 0.9
  latency   median: 0.15563,  mean: 0.24805,  99th_p: 0.24920,  99th_uncertainty: 4.71088

==========================
network: tftrt_fp16_imagenet_frozen_graph.pb,  batchsize 128, steps 100
  fps   median: 1279.1,   mean: 1265.6,   uncertainty: 12.6,  jitter: 2.4
  latency   median: 0.10007,  mean: 0.16619,  99th_p: 0.17133,  99th_uncertainty: 3.36558
```

The script will also output the GraphDefs used for each of the modes run,
for future use and inspection:

```
ls /my/output
log.txt
tftrt_fp16_imagenet_frozen_graph.pb
tftrt_fp32_imagenet_frozen_graph.pb
```

TODO(tfboyd): Numbers/testing for int8?

## Troubleshooting and Notes

### GPU/Precision Compatibility

Not all GPUs support the ops required for all precisions. For example, running
int8 precision on a NVIDIA P100 will result in a segfault:

```
E tensorflow/contrib/tensorrt/log/trt_logger.cc:38] DefaultLogger Parameter check failed at: Network.cpp::addScale::118, condition: shift.count == 0 || shift.count == weightCount
Segmentation fault (core dumped)
```
