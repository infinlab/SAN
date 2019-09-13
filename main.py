from model import SAN, train, segment, find_last, load_weights, set_log_dir, predict_on_batch
from util import data_prepare
import config
import os
import scipy.misc
import numpy as np
# 使用第一张与第三张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

"""
数据集结构
-Dataset
    -train
        -real
            -image
            -mask
        -fake
    -val
    -test
"""

if __name__ == '__main__':
    mode = 'trainS'  # trainS,trainD, trainSS, detect
    epoch, log_dir, checkpoint_path = set_log_dir()
    San_model = SAN(mode=mode, config=config).model
    # TODO: 加载预训练模型
    weights_path = ''
    if weights_path:
        if weights_path == 'last':
            weights_path = find_last()
        load_weights(weights_path, San_model, by_name=True)
        # Update the log directory
        epoch, log_dir, checkpoint_path = set_log_dir(weights_path)

    dataset_dir = '/home/henry/ai/dlData/BraTS/forSAN'
    if mode[:5] == 'train':
        # 加载训练、评估数据集
        folder = 'real'
        train_dataset = data_prepare(dataset_dir, 'train', folder)
        val_dataset = data_prepare(dataset_dir, 'val', folder)
        train(San_model, train_dataset, val_dataset, mode, log_dir, checkpoint_path, epoch, epochs=30)

    elif mode[:6] == 'detect':
        segmenter = False
        if segmenter:
            # 分割
            image_path = '/home/henry/ai/dlData/san_test/train/fake/image/G01_01113.jpg'
            pre_mask = segment(San_model, image_path)
            output_path = os.path.join(config.SEGMENT_MASK, os.path.split(image_path)[1])
            scipy.misc.imsave(output_path, np.squeeze(pre_mask))
        else:
            folder = 'test'
            test_dataset = data_prepare(dataset_dir, 'test', folder)
            Dices = predict_on_batch(test_dataset, San_model)
            print("Mean Dice overa {} images: {:.4f}".format(len(Dices), np.mean(Dices)))
    else:
        raise SystemExit('Please re-enter mode!')
