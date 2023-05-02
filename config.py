import glob, logging


def set_logger():
    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    return logger


def get_files(
        video_files_paths,
        extensions=['.mp4', '.avi', '.mov', '.mpg', 'mkv']
    ):

    files = []
    for video_files_path in video_files_paths:
        for extension in extensions:
            files.extend(glob.glob(video_files_path+'/**/*'+extension, recursive=True))

    print(f"Total {len(files)} files found")
    return files

    
def make_text_file(video_files_path, text_file_path='video.txt'):
    files = get_files(video_files_path)
    with open(text_file_path, 'w') as f:
        for file in files:
            f.write(file + '  1  1 '+'\n')

    return text_file_path


def get_save_dir(z, model_name, num_segment, version, to_replace='video'):
    tag = '@'+model_name+'.'+str(num_segment)+'.'+str(version)+'.'+to_replace+'.npy'
    file_name = z.replace(z[-4:], tag.lower())

    return file_name


class CONFIG_MXNET:
    __version__ = '0.0.1'

    data_list = make_text_file(
        video_files_path="path/to/video/files",
        text_file_path='videolist.txt'
    )
    save_dir = 'video/features'

    dtype = 'float32'
    model = 'i3d_resnet50_v1_kinetics400'
    input_size = 224
    num_segments = 10
    new_height = 256
    new_length = 32
    new_step = 1
    new_width = 340
    num_classes = 400
    fast_temporal_stride = 2
    gpu_id = 0
    hashtag = ''
    log_interval = 10
    mode = None
    need_root = False
    num_crop = 1
    resume_params = ''
    slow_temporal_stride = 16
    slowfast = False
    ten_crop = False
    three_crop = False
    use_decord = True
    use_pretrained = True
    video_loader = True
    logger = set_logger()
    get_save_dir = get_save_dir
