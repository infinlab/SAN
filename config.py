BATCH_SIZE = 8

IMAGE_SHAPE = 256

STEPS_PER_EPOCH = 300
VALIDATION_STEPS = 100

LRARNING_RATE = 0.001
MOMENTUM = 0.9

# loss_seg - alpha*loss_
ALPHA = 0.1

MODEL_DIR = '/home/henry/ai/tfProject/SAN/output/log'
SEGMENT_MASK = '/home/henry/ai/tfProject/SAN/output/result'
METRIC = '/home/henry/ai/tfProject/SAN/output/metric'
NAME = 'SAN'