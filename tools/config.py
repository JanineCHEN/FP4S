import argparse


def get_config():
    parser = argparse.ArgumentParser(description='FP4S segmentation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--backbone', type=str, 
                        default='resnet18', 
                        help="Choose from 'resnet18','deepbase_resnet18','resnet34','deepbase_resnet34','resnet50','deepbase_resnet50','resnet101','deepbase_resnet101','deepbase_resnet101','resnet152','deepbase_resnet152'")
    parser.add_argument('--root_dir', type=str, default='logs/')
    parser.add_argument('--log_name', type=str, default='/')
    parser.add_argument('--base_lr', type=float, default=5e-7)
    parser.add_argument('--max_lr', type=float, default=5e-4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--trainsize', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--step_size_up', type=int, default=150)
    parser.add_argument('--resume_ep', type=int, default=0)
    parser.add_argument('--w_sm', type=float, default=1.0)
    parser.add_argument('--w_edge', type=float, default=1.0)
    parser.add_argument('--lr_schedule', type=bool, default=False)
    parser.add_argument('--ifNorm', type=bool, default=False)
    parser.add_argument('--ifpretrain', type=bool, default=False)
    parser.add_argument('--useFocalLoss', type=bool, default=False)
    parser.add_argument('--useabCELoss', type=bool, default=False)
    parser.add_argument('--cutmix', type=bool, default=False)
    parser.add_argument('--download_fps', type=bool, default=False)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--reduction', type=str, default='mean', help="Choose from 'none'; 'mean';'sum'") 
    parser.add_argument('--num_groups', type=int, default=4)
    parser.add_argument('--num_unsupervised_imgs', type=int, default=1000, help="Choose from 1000,2000,5000,10000,20000")
    cfg = parser.parse_args()

    cfg._data = 'FP_train'
    cfg.image_root = f'./dataset/{cfg._data}/img/'
    cfg.gt_root = f'./dataset/{cfg._data}/gt/'
    cfg.mask_root = f'./dataset/{cfg._data}/mask/'
    cfg.gray_root = f'./dataset/{cfg._data}/gray/'
    cfg.edge_root = f'./dataset/{cfg._data}/edge/'
    cfg._data_val = 'FP_val'
    cfg.image_val_root = f'./dataset/{cfg._data_val}/img/'
    cfg.gt_val_root = f'./dataset/{cfg._data_val}/gt/'
    cfg.image_unsupervised_root = f'./dataset/FP_unsupervised/'
    cfg.image_unsupervised_list_path = f'./dataset/unAnnotated_FPs_{cfg.num_unsupervised_imgs}.txt'
    cfg.num_classes = 25
    cfg.channel = 8 * 4
    cfg.edge_loss_weight = cfg.w_edge
    cfg.sm_loss_weight = cfg.w_sm

    cfg.log_dir = cfg.root_dir + f'/{cfg.log_name}/CCT{cfg.num_unsupervised_imgs}_seed{cfg.seed}_{cfg.backbone}_img{cfg.trainsize}_n{str(cfg.ifNorm)}_abl{str(cfg.useabCELoss)}_fl{str(cfg.useFocalLoss)}_cutmix{str(cfg.cutmix)}_wsm{str(cfg.w_sm)}_we{str(cfg.w_edge)}/'
    cfg.model_path = cfg.log_dir + '/models/'
    cfg.txt_path = cfg.log_dir + '/txt/'

    print(cfg)
    return cfg


if __name__ == '__main__':
    print(get_config())
