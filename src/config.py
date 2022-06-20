
backbone = 'vgg19bn'   # Use xception or resnet as feature extractor,
clip = 1
workers=2
epochs=7
optim='Adam'
start_epoch=7
batch_size=1
lr=1e-5
momentum=0.9
weight_decay=5e-4
print_freq=100
print_model='course'
resume=True
evaluate=0
year=2017
eval_n_epoch=2    #eval skip n e7poch
#eval_index=['iou','pr','mae','map']
#printed attrs
train_index={'mae':0, 'pr':0, 'iou':1, 'map':0}
eval_index={'mae':0, 'pr':0, 'iou':1, 'map':0}
width,height = 400,800
pretrained='store_true'
half=None
save_dir='save_dir1'
nAveGra6=1
gpu_id = "1"
augmentoptions=[]
obj_id = "1"
sequence = "bike-packing"

youtube_datadir = "/disk1/gzzz/YouTubeVOS_2018/"
coco_datadir = "/home/ljj/data/COCO/"
davis_datadir = "/disk1/ztq/DAVIS-2017/"

res_savedir = "result0"

print('Using GPU: {} '.format(gpu_id))
print('batch_size:',batch_size)
class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return '/home/gzzz/jupyter/data/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif database == 'sbd':
            return '/path/to/Segmentation/benchmark_RELEASE/' # folder that contains dataset/.
        elif database == 'cityscapes':
            return '/path/to/Segmentation/cityscapes/'         # foler that contains leftImg8bit/
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

