# USAGE
# python detect_video.py --model mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite --labels mobilenet_ssd_v2/coco_labels.txt

# import the necessary packages
from imutils.video import FPS
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import platform
import subprocess
from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.utils import dataset_utils, image_processing
from PIL import Image
from PIL import ImageDraw
import numpy as np

def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  indices = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((indices >> channel) & 1) << shift
    indices >>= 3

  return colormap

def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
	# help="path to TensorFlow Lite object detection model")
# ap.add_argument("-l", "--labels", required=True,
	# help="path to labels file")
# ap.add_argument("-c", "--confidence", type=float, default=0.3,
	# help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())

parser = argparse.ArgumentParser()
parser.add_argument('--model',help='Path of the segmentation model.',required=True)
parser.add_argument('--input', help='File path of the input image.', required=False)
parser.add_argument('--output', help='File path of the output image.')
parser.add_argument('--keep_aspect_ratio',dest='keep_aspect_ratio',action='store_true',help=(
	'keep the image aspect ratio when down-sampling the image by adding '
    'black pixel padding (zeros) on bottom or right. '
    'By default the image is resized and reshaped without cropping. This '
    'option should be the same as what is applied on input images during '
    'model training. Otherwise the accuracy may be affected and the '
    'bounding box of detection result may be stretched.'))
    
parser.set_defaults(keep_aspect_ratio=False)
args = parser.parse_args()

# load the Google Coral object detection model
print("[INFO] loading Coral model...")
engine = BasicEngine(args.model)
_, height, width, _ = engine.get_input_tensor_shape()

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=False).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 500 pixels
	img = vs.read()
	img = imutils.resize(img, width=500)
	orig = img.copy()

	# prepare the frame for object detection by converting (1) it
	# from BGR to RGB channel ordering and then (2) from a NumPy
	# array to PIL image format
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = Image.fromarray(img)
	
	if args.keep_aspect_ratio:
		resized_img, ratio = image_processing.resampling_with_original_ratio(
			img, (width, height), Image.NEAREST)
	else:
		resized_img = img.resize((width, height))
		ratio = (1., 1.)
		
	input_tensor = np.asarray(resized_img).flatten()
	_, raw_result = engine.run_inference(input_tensor)
	result = np.reshape(raw_result, (height, width))
	new_width, new_height = int(width * ratio[0]), int(height * ratio[1])

	# If keep_aspect_ratio, we need to remove the padding area.
	result = result[:new_height, :new_width]
	vis_result = label_to_color_image(result.astype(int)).astype(np.uint8)
	vis_result = Image.fromarray(vis_result)

	vis_img = resized_img.crop((0, 0, new_width, new_height))

	# loop over the results
	# Concat resized input image and processed segmentation results.
	concated_image = Image.new('RGB', (new_width*2, new_height))
	concated_image.paste(vis_img, (0, 0))
	concated_image.paste(vis_result, (width, 0))
	concated_image.save('yeet.jpg')
			
	# # Display result.
	# if platform.machine() == 'x86_64':
		# # For gLinux, simply show the image.
		# concated_image.show()
	# elif platform.machine() == 'armv7l':
		# # For Raspberry Pi, you need to install 'feh' to display image.
			# subprocess.Popen(['feh', 'yeet.jpg'])
	# else:
		# print('Please check ', 'yeet.jpg')
	
	image = cv2.imread('yeet.jpg')
	#time.sleep(5)

	# # show the output frame and wait for a key press
	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
		
	fps.update()

fps.stop()
print("[INFO] elapse time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
