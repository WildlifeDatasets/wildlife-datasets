import os
import argparse
if __name__ == '__main__':
    import utils
else:
    from . import utils

def get_data(root):
    with utils.data_directory(root):
        import warnings        
        warnings.warn("You are trying to download a segmented dataset %s. \
                      It is already included in its non-segmented version download. \
                      Skipping." % os.path.split(root)[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))