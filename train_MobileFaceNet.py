import os
import sys
import time
import argparse
import numpy as np  

from sklearn import metrics
from scipy import interpolate
from scipy.optimize import brentq
from itertools import combinations
from sklearn.model_selection import KFold

from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from src.get_distance import GetDistanceTensorflow
from src.generate_pairs import generate_pairs_with_balance, generate_all_pairs
from src.feature_extract_model_evaluate import img_pipeline, read_paris_file, get_pairs, evaluate
from src.data_generators import FaceDataGenerator
from src.build_model import build_model, dummy_loss, lr_scheduler, ArcFaceLossLayer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def _main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    global gdt
    gdt = GetDistanceTensorflow(metric_type='cos', embedding_dim=args.embedding_dim, epslon=1e-8)
    
    #======================#
    # initialize generator #
    #======================#
    print('** Initializing data generator.')
    train_generator = FaceDataGenerator(args.train_directory, args.batch_size, args.aug_freq, args.image_size, args.shuffle)
    num_classes = train_generator.num_classes
    if args.valid_directory:
        valid_generator = FaceDataGenerator(args.valid_directory, args.batch_size, args.aug_freq, args.image_size, args.shuffle)
    
    #======================#
    # *** build model ***  #
    #======================#

    # ** define custom object
    custom_loss = ArcFaceLossLayer(s=args.loss_scale, m=args.loss_margin, num_classes=num_classes, is_use_bais=False)

    if args.pretrained_model:
        try:
            model = load_model(args.pretrained_model, custom_objects={'ArcFaceLossLayer':custom_loss, 'dummy_loss':dummy_loss})
            print('Susessed to load model from {}'.format(args.pretrained_model))
        except Exception as e:
            print('Failed to laod model, will train from scratch.')
            model = build_model((args.image_size, args.image_size, 3), num_classes, args.expansion_ratio, args.embedding_dim, args.loss_scale, args.loss_margin)
    else:
        print('Not use pretrained model, will train from scratch.')
        model = build_model((args.image_size, args.image_size, 3), num_classes, args.expansion_ratio, args.embedding_dim, args.loss_scale, args.loss_margin)

    model.summary()
    model.compile(optimizer='adam', loss=dummy_loss)

    #======================#
    # setup keras callback #
    #======================#
    filename = 'ep{epoch:03d}-loss{loss:.3f}.h5'
    save_path = os.path.join(args.save_model_directory, filename)
    checkpoint = ModelCheckpoint(save_path, monitor='loss', save_best_only=True, period=args.checkpoint_epochs)
    scheduler = LearningRateScheduler(lr_scheduler)

    #======================#
    #   training phase     #
    #======================#
    assert args.epochs % args.evaluate_epochs == 0
    print('** Start training.')
    for i in range(int(args.epochs / args.evaluate_epochs)):
        #=============#
        # ** train ** #
        #=============#
        print('** [Epoch] : {}'.format(i * args.evaluate_epochs))
        num_epochs = args.evaluate_epochs * (i + 1)
        ini_epochs = args.evaluate_epochs * (i + 0)
        model.fit_generator(generator=train_generator, epochs=num_epochs, callbacks=[checkpoint, scheduler], verbose=1, initial_epoch=ini_epochs)
        
        #======================#
        # ** generate pairs ** #
        #======================#
        print('\n\n')
        print('** Evaluate on {}'.format(args.data_directory))
        if not os.path.exists(args.pairs_filename):
            print('** start generate pairs.')
            print('** use dataset : {}'.format(args.data_directory))
            print('** sample type : {}'.format(args.sample_type))
            if args.sample_type == 0:
                try:
                    run_time, num_pairs = generate_pairs_with_balance(args.data_directory, args.pairs_filename, args.repeat_times)
                    print('** repeat times : {}'.format(args.repeat_times))
                    print('** thre are totally {} pairs.'.format(num_pairs))
                    print('** save file to {}'.format(args.pairs_filename))
                    print('** use time : {}s'.format(run_time))
                except Exception as e:
                    raise e
                    print('*** Failed to generate ***')
                    print(e)
            elif args.sample_type == 1:
                try:
                    run_time, num_combinations = generate_all_pairs(args.data_directory, args.pairs_filename, args.num_person, args.num_sample)
                    print('** num_person : {}'.format(args.num_person))
                    print('** num_sample : {}'.format(args.num_sample))
                    print('** thre are totally {} combinations.'.format(num_combinations))
                    print('** save file to {}'.format(args.pairs_filename))
                    print('** use time : {}s'.format(run_time))
                except Exception as e:
                    print('*** Failed to generate ***')
                    print(e)

        #================#
        # ** evaluate ** #
        #================#
        thresholds = np.arange(0, 2, 0.001)
        
        print('** start evaluation procedure.')
        print('(1) load model.\n')
        fe = FeatureExtractor(model, num_classes, args.loss_scale, args.loss_margin)

        print('\n(2) get pairs info.\n')
        pairs = read_paris_file(pairs_filename=args.pairs_filename)
        path_list, is_same_list = get_pairs(pairs)

        print('    * number of pairs : {}'.format(len(is_same_list)))

        print('\n(3) feature extraction.\n')
        num_embeddings = len(is_same_list) * 2 # num_pairs * 2 (2 images per pairs)
        num_batchs = num_embeddings // args.batch_size

        print('    * start feature extraction.')
        start_time = time.time()
        embeddings_array = np.zeros((num_embeddings, args.embedding_dim))
        for batch_index in range(num_batchs):
            start_index = int(batch_index * args.batch_size)
            end_index = min((batch_index + 1) * args.batch_size, num_embeddings)
            batch_img_list = path_list[start_index:end_index]
            embeddings_array[start_index:end_index, :] = fe.infer(img_pipeline(batch_img_list, args.image_size))
        use_time = time.time() - start_time
        print('    * use time : {}s for {} images, {}s per 1 image'.format(round(use_time, 4), num_embeddings, round(use_time / num_embeddings, 3)))

        print('\n(4) evaluate\n')
        tpr, fpr, accuracy, val, val_std, far, thresh_mean = evaluate(embeddings_array, is_same_list, thresholds, args.far_target, 100, gdt, 10)

        print('\n*** result ***\n')
        print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
        print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
        print('Threshold=%2.5f @ FAR=%2.5f' % (thresh_mean, far))
        auc = metrics.auc(fpr, tpr)
        print('Area Under Curve (AUC): %1.3f' % auc)
        eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
        print('Equal Error Rate (EER): %1.3f' % eer)
        print('\n\n')

class FeatureExtractor():
    def __init__(self, model, num_classes, s, m):
        self.model = model
        self.num_classes = num_classes
        self.__get_infer_model()
    
    def __get_infer_model(self):
        self.infer_model = Model(inputs=self.model.input, outputs=self.model.get_layer('embeddings').output)

    def __normalize(self, img):
        return np.multiply(np.subtract(img, 127.5), 1 / 128)

    def infer(self, faces):
        normalized = [self.__normalize(f) for f in faces]
        normalized = np.array(normalized)
        if len(normalized.shape) < 4:
            normalized = np.expand_dims(normalized, axis=0) 
        dummy_y = np.zeros((faces.shape[0], self.num_classes))
        return self.infer_model.predict([normalized, dummy_y])

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # ** for data generator
    parser.add_argument('train_directory', type=str, help='Training dataset directory')
    parser.add_argument('--valid_directory', type=str, help='Validation dataset directory', default='')
    parser.add_argument('--batch_size', type=int, help='Batch size of generator', default=200)
    parser.add_argument('--aug_freq', type=float, help='Data augmentation frequency', default=0.5)
    parser.add_argument('--image_size', type=int, help='Model input size', default=112)
    parser.add_argument('--shuffle', type=bool, help='Shuffle on end of epoch', default=True)

    # ** for model
    parser.add_argument('--expansion_ratio', type=int, help='Expansion ratio of res_block', default=6)
    parser.add_argument('--embedding_dim', type=int, help='Embedding Dimension', default=256)
    parser.add_argument('--loss_scale', type=int, help='Scale for arc-face loss', default=64)
    parser.add_argument('--loss_margin', type=float, help='Angular margin for arc-face loss', default=0.5)

    # ** for training
    parser.add_argument('--pretrained_model', type=str, help='Pre-trained model filename', default='')
    parser.add_argument('--save_model_directory', type=str, help='Directory to save model.', default='weights/')
    parser.add_argument('--checkpoint_epochs', type=int, help='Save checkpoint every n epochs.', default=5)
    parser.add_argument('--epochs', type=int, help='Number of training epochs', default=300)
    parser.add_argument('--valid_split_ratio', type=float, help='Split ratio of validation set', default=0.1)
    parser.add_argument('--evaluate_epochs', type=int, help='Evaluate model every n epochs', default=5)

    # ** for eveluattion pairs generation
    parser.add_argument('data_directory', type=str, help='Evaluation dataset directory.')
    parser.add_argument('pairs_filename', type=str, help='Pairs file to evaluate.')
    parser.add_argument('--sample_type', type=int, help='Sample type of the task. 0:balance pos/neg, 1:sample by person and img per person', default=0)
    parser.add_argument('--repeat_times', type=int, help='Repeat times of generation, this argument only be used when --sample type is 0.', default=10)
    parser.add_argument('--num_person',type=int, help='Number of person to sample, this argument only be used when --sample type is 1.', default=10)
    parser.add_argument('--num_sample', type=int, help='Number of sample per person, this argument only be used when --sample type is 1.', default=20)
    parser.add_argument('--far_target', type=float, help='target FAR(False Accept Rate)', default=1e-2)
    
    # ** for GPU setup
    parser.add_argument('--gpu', type=str, help='Specify a GPU.', default='1')
    
    return parser.parse_args(argv)


if __name__ == '__main__':
    _main(parse_arguments(sys.argv[1:]))


