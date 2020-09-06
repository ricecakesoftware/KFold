import argparse
import datetime
import numpy as np
import os
import random
import statistics

from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPool2D
from keras.models import Model
from PIL import Image

class KFold:
    _classes = []
    _n_classes = 0
    _n_splits = 0
    _image_files = []

    def load(self, image_dir, n_splits):
        # ディレクトリーから分類リストを作成
        files = os.listdir(image_dir)
        self._classes = [f for f in files if os.path.isdir(os.path.join(image_dir, f))]
        self._n_classes = len(self._classes)
        # 画像ファイル一覧を作成(N分割)
        self._n_splits = n_splits
        for c in range(self._n_classes):
            class_dir = os.path.join(image_dir, self._classes[c])
            files = os.listdir(class_dir)
            image_files = []
            for f in files:
                image_files.append(os.path.join(class_dir, f))
            random.shuffle(image_files)
            self._image_files.append(self._chunks(image_files, self._n_splits))
            num = 0
            for i in range(self._n_splits):
                num += len(self._image_files[c][i])
            print('{}: {} has {} images.'.format(datetime.datetime.now(), self._classes[c], num))
        return self._n_classes

    def train(self, model, width, height, channels, n_epochs, batch_size, hdf5_path, patience):
        val_loss_min = 9.99999 # 最適値は不明
        val_loss_over_count = 0
        continuable = True
        for _ in range(n_epochs):
            for i in range(self._n_splits):
                train, val = self._divide(i)
                n_batches = len(train) // batch_size
                train_batches = self._chunks(train, n_batches)
                val_batches = self._chunks(val, n_batches)
                # 訓練
                losses = []
                accs = []
                for b in range(n_batches):
                    x_train, y_train = self._convert(train_batches[b], width, height, channels)
                    result = model.train_on_batch(x_train, y_train)
                    losses.append(result[0])
                    accs.append(result[1])
                    loss = statistics.mean(losses)
                    acc = statistics.mean(accs)
                    print('\r' + '{}: [{:5}/{:5}]({:6.2f}%) loss={:.5f}, acc={:.5f}'.format(datetime.datetime.now(), b + 1, n_batches, (b + 1) / n_batches * 100.0, loss, acc), end='')
                val_losses = []
                val_accs = []
                # 検証
                for b in range(n_batches):
                    x_val, y_val = self._convert(val_batches[b], width, height, channels)
                    result = model.test_on_batch(x_val, y_val)
                    val_losses.append(result[0])
                    val_accs.append(result[1])
                val_loss = statistics.mean(val_losses)
                val_acc = statistics.mean(val_accs)
                print(' => val_loss={:.5f}, val_acc={:.5f}'.format(val_loss, val_acc))
                # 後処理
                if (val_loss < val_loss_min):
                    model.save(hdf5_path)
                    print('{}: model saved. {:.5f}=>{:.5f}'.format(datetime.datetime.now(), val_loss_min, val_loss))
                    val_loss_min = val_loss
                    val_loss_over_count = 0
                else:
                    val_loss_over_count += 1
                    continuable = False if val_loss_over_count >= patience else True
                if not continuable:
                    print('training finished.')
                    break
            if not continuable:
                break

    def _chunks(self, list_, n_splits):
        # N分割(割り切れない場合は切り捨て)
        n = len(list_) // n_splits
        return [list_[i: i + n] for i in range(0, len(list_) - len(list_) % n_splits, n)]

    def _divide(self, val_index):
        # N分割された画像ファイルを訓練用、学習用に分割
        x_train = []
        y_train = []
        x_val = []
        y_val = []
        for c in range(self._n_classes):
            for s in range(self._n_splits):
                image_files = self._image_files[c][s]
                if s == val_index:
                    x_val.extend(image_files)
                    y_val.extend([c] * len(image_files))
                else:
                    x_train.extend(image_files)
                    y_train.extend([c] * len(image_files))
        # データ・ラベルをタプル化
        train = list(zip(x_train, y_train))
        val = list(zip(x_val, y_val))
        random.shuffle(train)
        random.shuffle(val)

        return (train, val)

    def _convert(self, chunk, width, height, channels):
        x = []
        y = []
        for (f, c) in chunk:
            image = Image.open(f).resize((width, height))
            if channels == 1:
                image = image.convert('L')
            # 画像水増し処理が必要
            x.append(np.array(image) / 255.0)
            y.append(np.identity(self._n_classes)[c])
        x = np.array(x).reshape((len(x), height, width, channels))
        y = np.array(y).reshape((len(y), self._n_classes))
        return (x, y)

def main(args):
    kfold = KFold()
    n_classes = kfold.load(args.image_dir, args.n_splits)
    model = create_model(args.image_width, args.image_height, args.image_channels, n_classes)
    kfold.train(model, args.image_width, args.image_height, args.image_channels, args.n_epochs, args.batch_size, args.hdf5_path, args.patience)

def create_model(width, height, channels, n_classes):
    inputs = Input(shape=(height, width, channels))
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(n_classes)(x)
    outputs = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='cifar-10\\train')
    parser.add_argument('--n_splits', type=int, default=10)
    parser.add_argument('--n_epochs', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hdf5_path', type=str, default='cifar-10.hdf5')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--image_width', type=int, default=32)
    parser.add_argument('--image_height', type=int, default=32)
    parser.add_argument('--image_channels', type=int, default=3)
    args, _ = parser.parse_known_args()
    main(args)
