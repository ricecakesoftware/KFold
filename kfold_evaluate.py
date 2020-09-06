import argparse
import datetime

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

def main(args):
    color_mode = 'grayscale' if args.image_channels == 1 else 'rgb'
    evaluate_generator = ImageDataGenerator(rescale=1.0/255.0).flow_from_directory(args.image_dir, target_size=(args.image_width, args.image_height), color_mode=color_mode, batch_size=args.batch_size)
    model = load_model(args.hdf5_path)
    result = model.evaluate_generator(evaluate_generator)
    print('{}: loss={:.5f}, acc={:.5f}'.format(datetime.datetime.now(), result[0], result[1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_width', type=int, default=32)
    parser.add_argument('--image_height', type=int, default=32)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--image_dir', type=str, default='cifar-10\\test')
    parser.add_argument('--hdf5_path', type=str, default='cifar-10.hdf5')
    parser.add_argument('--batch_size', type=int, default=64)
    args, _ = parser.parse_known_args()
    main(args)
