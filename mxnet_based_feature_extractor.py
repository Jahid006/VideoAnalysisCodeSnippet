import os
import time
import gc
import tqdm

import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv.model_zoo import get_model
from gluoncv.data import VideoClsCustom
from gluoncv.utils.filesystem import try_import_decord


def read_video(opt, video_name, transform, video_utils):

    decord = try_import_decord()
    decord_vr = decord.VideoReader(video_name, width=opt.new_width, height=opt.new_height)
    duration = len(decord_vr)

    opt.skip_length = opt.new_length * opt.new_step
    segment_indices, skip_offsets = video_utils._sample_test_indices(duration)

    if opt.video_loader:
        if opt.slowfast:
            clip_input = video_utils._video_TSN_decord_slowfast_loader(
                video_name, decord_vr, duration, segment_indices, skip_offsets
            )
        else:
            clip_input = video_utils._video_TSN_decord_batch_loader(
                video_name, decord_vr, duration, segment_indices, skip_offsets
            )
    else:
        raise RuntimeError('We only support video-based inference.')

    clip_input = transform(clip_input)

    if opt.slowfast:
        sparse_sampels = len(clip_input) // (opt.num_segments * opt.num_crop)
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (sparse_sampels, 3, opt.input_size, opt.input_size))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
    else:
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (opt.new_length, 3, opt.input_size, opt.input_size))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))

    if opt.new_length == 1:
        clip_input = np.squeeze(clip_input, axis=2)

    return nd.array(clip_input)


def feature_extractor(opt):
    logger = opt.logger
    gc.set_threshold(100, 5, 5)
    if opt.gpu_id == -1:
        context = mx.cpu()
    else:
        gpu_id = opt.gpu_id
        context = mx.gpu(gpu_id)

    image_norm_mean = [0.485, 0.456, 0.406]
    image_norm_std = [0.229, 0.224, 0.225]
    if opt.ten_crop:
        transform_test = transforms.Compose([
            video.VideoTenCrop(opt.input_size),
            video.VideoToTensor(),
            video.VideoNormalize(image_norm_mean, image_norm_std)
        ])
        opt.num_crop = 10
    elif opt.three_crop:
        transform_test = transforms.Compose([
            video.VideoThreeCrop(opt.input_size),
            video.VideoToTensor(),
            video.VideoNormalize(image_norm_mean, image_norm_std)
        ])
        opt.num_crop = 3
    else:
        transform_test = video.VideoGroupValTransform(
            size=opt.input_size, mean=image_norm_mean, std=image_norm_std
        )
        opt.num_crop = 1

    # get model
    if opt.use_pretrained and len(opt.hashtag) > 0:
        opt.use_pretrained = opt.hashtag
    classes = opt.num_classes
    model_name = opt.model
    net = get_model(
        name=model_name,
        nclass=classes,
        pretrained=opt.use_pretrained,
        feat_ext=True,
        num_segments=opt.num_segments,
        num_crop=opt.num_crop
    )

    net.cast(opt.dtype)
    net.collect_params().reset_ctx(context)

    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)

    if opt.resume_params != '' and not opt.use_pretrained:
        net.load_parameters(opt.resume_params, ctx=context)
        logger.info('Pre-trained model %s is successfully loaded.' % (opt.resume_params))

    else:
        logger.info('Pre-trained model is successfully loaded from the model zoo.')

    logger.info("Successfully built model {}".format(model_name))

    anno_file = opt.data_list
    f = open(anno_file, 'r')
    data_list = f.readlines()
    logger.info('Load %d video samples.' % len(data_list))

    video_utils = VideoClsCustom(
        root=opt.data_dir,
        setting=opt.data_list,
        num_segments=opt.num_segments,
        num_crop=opt.num_crop,
        new_length=opt.new_length,
        new_step=opt.new_step,
        new_width=opt.new_width,
        new_height=opt.new_height,
        video_loader=opt.video_loader,
        use_decord=opt.use_decord,
        slowfast=opt.slowfast,
        slow_temporal_stride=opt.slow_temporal_stride,
        fast_temporal_stride=opt.fast_temporal_stride,
        data_aug=opt.data_aug,
        lazy_init=True
    )

    start_time = time.time()
    for vid, vline in tqdm.tqdm(enumerate(data_list)):
        video_path = vline.split()[0]
        z = vline.split()[0]
        if opt.need_root:
            video_path = os.path.join(opt.data_dir, video_path)
        video_data = read_video(opt, video_path, transform_test, video_utils)
        video_input = video_data.as_in_context(context)

        video_feat = net(video_input.astype(opt.dtype, copy=False))
        feat_file = opt.get_save_dir(z, model_name, opt.num_segments, opt.version, to_replace='video')

        os.makedirs(os.path.dirname(feat_file), exist_ok=True)
        np.save(feat_file, video_feat.asnumpy())

    end_time = time.time()
    logger.info('Total feature extraction time is %4.2f minutes' % ((end_time - start_time) / 60))


if __name__ == "__main__":
    import config as cfg

    extractor_config = cfg.CONFIG_MXNET
    feature_extractor(extractor_config)
