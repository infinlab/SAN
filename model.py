from keras.models import *
from keras.layers import *
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from util import *
import config

########################################################################
# MODEL
########################################################################


def discriminator(fm):
    f0 = UpSampling2D(size=(2, 2), name='dis_up1')(fm[0])
    m1 = concatenate([f0, fm[1]], axis=3)
    f1 = UpSampling2D(size=(2, 2), name='dis_up1')(m1)
    m2 = concatenate([f1, fm[2], fm[3]], axis=3)
    x = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dis_conv1')(m2)
    x = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dis_conv2')(x)
    x = Conv2D(64, 1, activation='relu', padding='same', kernel_initializer='he_normal', name='dis_conv3')(x)
    x = Conv2D(1, 1, activation='relu', padding='same', kernel_initializer='he_normal', name='dis_conv4')(x)
    x = Flatten()(x)
    x = Dense(1, activation='relu', name='dis_dense')(x)
    return x


def res_block(x, nb_filters, strides, key):
    res_path = BatchNormalization(name=key+"_bn1")(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0], name=key+"_conv1")(res_path)
    res_path = BatchNormalization(name=key+"_bn2")(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1], name=key+"_conv2")(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0], name=key+"_conv3")(x)
    shortcut = BatchNormalization(name=key+"_bn3")(shortcut)

    res_path = add([shortcut, res_path])
    return res_path


def encoder(x):
    to_decoder = []

    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1), name='seg_encoder_conv1')(x)
    main_path = BatchNormalization(name='seg_encoder_bn1')(main_path)
    main_path = Activation(activation='relu')(main_path)

    main_path = Conv2D(filters=64, kernel_size=(3, 3),
                       padding='same', strides=(1, 1), name='seg_encoder_conv2')(main_path)

    shortcut = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), name='seg_encoder_conv3')(x)
    shortcut = BatchNormalization(name='seg_encoder_bn2')(shortcut)

    main_path = add([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)], 'seg_encoder1')
    to_decoder.append(main_path)

    main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)], 'seg_encoder2')
    to_decoder.append(main_path)

    return to_decoder


def decoder(x, from_encoder):
    fm = []
    main_path = UpSampling2D(size=(2, 2), name='seg_up1')(x)
    main_path = concatenate([main_path, from_encoder[2]], axis=3)
    fm.append(main_path)
    main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)], 'seg_decoder1')

    main_path = UpSampling2D(size=(2, 2), name='seg_up2')(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=3)
    fm.append(main_path)
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)], 'seg_decoder2')

    main_path = UpSampling2D(size=(2, 2), name='seg_up3')(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=3)
    fm.append(main_path)
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)], 'seg_decoder3')
    fm.append(main_path)

    return main_path, fm


def res_unet(image):
    to_decoder = encoder(image)
    path = res_block(to_decoder[2], [512, 512], [(2, 2), (1, 1)], 'seg_path')
    path, fm = decoder(path, from_encoder=to_decoder)
    mask = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', name='seg_conv')(path)
    return mask, fm


class SAN():
    def __init__(self, mode, config):
        self.mode = mode
        self.config = config
        self.model = self.build()

    def build(self):
        input_image = Input(shape=(config.IMAGE_SHAPE, config.IMAGE_SHAPE, 1), name='input_image')
        segmentation, fm = res_unet(input_image)
        validity = discriminator(fm)
        if self.mode == 'trainS' or self.mode == 'detect':
            model = Model(input_image, segmentation)
        elif self.mode == 'trainD':
            model = Model(input_image, validity)
        elif self.mode == 'trainSS':
            model = Model(input_image, [segmentation, validity])
        else:
            raise SystemExit('Please re-enter mode')

        # TODO:添加 multi-GPU

        return model

########################################################################
# TRAIN and TEST
########################################################################


def train(model, train_dataset, val_dataset, mode,
          log_dir, checkpoint_path, epoch, epochs, learning_rate=config.LRARNING_RATE, momentum=config.MOMENTUM,
          augmentation=None):
    """
    # 先训练n次segmenter Loss_dice     数据集为source FCD
    # 在训练m次discriminator Loss_cls  训练D使其可以准确判别 source
    # 交替训练segmenter Loss_dice - alpha * Loss_cls 与 discriminator
    """
    """
    # layer select
    """

    # Data generator
    # 根据不同mode 生成不同的generator
    train_generator = data_generator(train_dataset, mode, batch_size=config.BATCH_SIZE)
    val_generator = data_generator(val_dataset, mode, batch_size=config.BATCH_SIZE)

    """
    # Callbacks
    """
    # 自适用学习率
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    # 保存权重
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 保存训练记录
    _, full_name = os.path.split(log_dir)
    csv_name, _ = os.path.splitext(full_name)
    csv_log = os.path.join(config.METRIC, csv_name)
    csv_logger = CSVLogger(csv_log)
    callbacks = [TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False),
                 ModelCheckpoint(checkpoint_path, verbose=0, save_weights_only=True),
                 reduce_lr,
                 csv_logger]

    # TODO:不同的mode
    # S_optimizer:adam, D_optimizer:SGD
    if mode == 'trainS':
        layer_regex = r"(seg\_.*)"
        optimizer = SGD(lr=learning_rate, momentum=momentum)
        model.compile(optimizer, loss=dice_coef_loss, metrics=[dice_coef])
        # model.compile(optimizer, loss='binary_crossentropy')
    elif mode == 'trainD':
        layer_regex = r"(dis\_.*)"
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=10e-8)
        model.compile(optimizer, loss='binary_crossentropy', metrics=['acc'])
    else:
        layer_regex = r"(seg\_.*)"
        optimizer = SGD(lr=learning_rate, momentum=momentum)
        model.compile(optimizer, loss=[dice_coef_loss, 'binary_crossentropy'], loss_weights=[1, config.ALPHA],
                      metrics=[dice_coef, 'acc'])

    for layer in model.layers:
        trainable = bool(re.fullmatch(layer_regex, layer.name))
        layer.trainable = trainable

    if epoch >= epochs:
        raise Exception("epoch must smaller than epochs!")

    model.fit_generator(
        train_generator,
        initial_epoch=epoch,
        epochs=epochs,
        steps_per_epoch=config.STEPS_PER_EPOCH,
        callbacks=callbacks,
        validation_data=val_generator,
        validation_steps=config.VALIDATION_STEPS,
    )


def segment(model, image_path):
    print("Running on {}".format(image_path))
    img = skimage.io.imread(image_path)
    img = reshape(img)
    img = np.expand_dims(img, axis=0)
    mask = model.predict(img, verbose=0)
    return mask


def predict_on_batch(data_meta, model):
    Dices = []

    return Dices
