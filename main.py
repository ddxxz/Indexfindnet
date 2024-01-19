import pandas as pd
import torch
from tqdm import tqdm
import os
import hydra
import logging
import random
from torch import optim
#from models.Model_1dref_attention import *
from models.Model import *
from models.Model_indicecnn import *
#from models.Model_3dimage_pca import *
from models.Model_indexfind import *

from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error

logger = logging.getLogger(__name__)


def seed_torch(seed,faster=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    if not faster:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
seed_torch(seed=1000,faster=True)

def loss_compres(output, target):
    return torch.sum(torch.pow(torch.abs(output - target), 4)) / output.size(0) + torch.sum(
        torch.pow(torch.abs(output - target), 2)) / output.size(0)


def loss_mse(output, target):
    return torch.sum(torch.square(output - target)) / output.size(0)


def loss_mae(output, target):
    return torch.sum(torch.abs(output - target)) / output.size(0)


# def loss_r2(output, target):
#     return -torch.log(1- loss_mse(output,target)/ torch.var(target))

def loss_r2(output, target):
    SStot = torch.sum(torch.pow(target, 2), -1)  # -torch.mean(target)
    SSres = torch.sum(torch.pow(target - output, 2), dim=-1)
    r2 = SStot / (SSres + 1e-8)  # 1-
    loss = -10 * torch.log10(r2)  #
    loss = torch.mean(loss)
    return loss


def evalution(args, model, loader):
    device = model.device
    model.eval()
    outputs = []
    labels = []
    diffs = []

    for data in loader:
        with torch.no_grad():
            # print(data)
            input = data['Spectral'].to(device=device, dtype=torch.float)
            #indice = data['Indice'].to(device=device, dtype=torch.float)
            if args.embedding:
                embedding = data['Embedding'].to(device=device, dtype=torch.float)
            #img = data['himg'].to(device=device, dtype=torch.float).cuda()
            # output = model.forward(input,indice)#, ,imgimg
            if args.task == 'hyper':
                output = model(input)  # ,input,indice
            else:
                output = model(img)
            outputs.append(output.cpu().numpy())
            label = data['Label'].to(device=device, dtype=torch.float)
            labels.append(label.cpu().numpy())
            # loss = loss_mse(output, label)
            diff = torch.abs(output - label)
            diffs.append(diff.cpu().numpy())
    outputs = np.concatenate(outputs)
    labels = np.concatenate(labels)
    diffs = np.concatenate(diffs)

    avg_r2 = r2_score(labels, outputs)
    return avg_r2, outputs, labels


#
def _main(args,_run=None):
    global __file__
    # Updating paths in config
    for key, value in args.dset.items():
        if isinstance(value, str) and key not in ["matching"]:
            args.dset[key] = hydra.utils.to_absolute_path(value)
    __file__ = hydra.utils.to_absolute_path(__file__)
    tb = SummaryWriter("tensorboard")

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    use_cuda = torch.cuda.is_available()
    device = torch.device(args.gpu_device if use_cuda else "cpu")
    print(device)

    if args.task == 'hyperimg':
        from data_proceed.hyperimg_data import BasicDataset
    else:
        from data_proceed.data import BasicDataset

    # print('embedding',args.embedding)
    train_data = BasicDataset(args,args.dset.tr_csv,train_flag=True)
    if args.task == 'hyperimg':
        train_data.allhyperimg = (train_data.allhyperimg - train_data.allhyperimg_mean) / train_data.allhyperimg_std
    train_data.input_x_norm = (train_data.input_x_norm - train_data.input_x_mean) / train_data.input_x_std
    train_data.label_norm = (train_data.label_norm - train_data.label_mean) / train_data.label_std
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False)

    val_data = BasicDataset(args,args.dset.cv_csv,train_flag=False)
    if args.task == 'hyperimg':
        val_data.allhyperimg = (val_data.allhyperimg - val_data.allhyperimg_mean) / val_data.allhyperimg_std
    val_data.input_x_norm = (val_data.input_x_norm - train_data.input_x_mean) / train_data.input_x_std
    val_data.label_norm = (val_data.label_norm - train_data.label_mean) / train_data.label_std
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False)

    print(len(train_loader))
    test_data = BasicDataset(args,args.dset.tt_csv,train_flag=False)
    print('test_data',len(test_data))
    if args.task == 'hyperimg':
        test_data.allhyperimg = (test_data.allhyperimg - test_data.allhyperimg_mean) / test_data.allhyperimg_std
    test_data.input_x_norm = (test_data.input_x_norm - train_data.input_x_mean) / train_data.input_x_std
    test_data.label_norm = (test_data.label_norm - train_data.label_mean) / train_data.label_std
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)

    MODELS = {
        'OnedCNN': OnedCNN,
        'IndiceCNN': IndiceCNN,
        'IndexfindNet': IndexfindNet
              }
    Vcmax_max = train_data.Vcmax_max
    Vcmax_min = train_data.Vcmax_min
    if args.task == 'hyper':
        model = MODELS[args.model](device, args.resample)
    else:
        model = MODELS[args.model](device, args.channel_num)
    #model = OnedCNNBERT.from_pretrained("bert-base-uncased")

    criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    #criterion = nn.SmoothL1Loss()
    model.to(device=device)
    logger.info(model)
    # if args.embedding:
    #     model.summary([2, 204],[2,1,256])
    # else:
    #     #model.summary([2, 21], [2, 3, 64, 64])
    #     model.summary([2, 204],[2,3,219,165])

    # tb.add_graph(model.cpu(), torch.zeros([1, 204]))
    #  if args.optim == 'Adam':
    #      optimizer = optim.Adam(model.parameters())
    #  elif args.optim == 'Adadelta':
    #      optimizer = optim.Adadelta(model.parameters())
    #  elif args.optim == 'Adamax':
    #      optimizer = optim.Adamax(model.parameters())
    #  elif args.optim == 'AdamW':
    #      optimizer = optim.AdamW(model.parameters())
    #  elif args.optim == 'ASGD':
    #      optimizer = optim.ASGD(model.parameters())
    #  elif args.optim == 'LBFGS':
    #      optimizer = optim.LBFGS(model.parameters())
    #  elif args.optim == 'NAdam':
    #      optimizer = optim.NAdam(model.parameters())
    #  elif args.optim == 'RAdam':
    #      optimizer = optim.RAdam(model.parameters())
    #  elif args.optim == 'RMSprop':
    #      optimizer = optim.RMSprop(model.parameters())
    #  elif args.optim == 'Rprop':
    #      optimizer = optim.Rprop(model.parameters())
    #  elif args.optim == 'SGD':
    #      optimizer = optim.SGD(model.parameters(),lr=0.05)
    #  elif args.optim == 'SparseAdam':
    #      optimizer = optim.SparseAdam(model.parameters())

    # lr = np.random.uniform(1e-3, 1e-4)
    # weight_decay = np.random.uniform(1e-2, 1e-6)
    # lr = 0
    # weight_decay = 0
    # optimizer = optim.AdamW(model.parameters())

    # if args.lr_scheduler == 'StepLR':
    #     scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
    # elif args.lr_scheduler == 'MultiStepLR':
    #     scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[3,7],gamma=0.1)
    # elif args.lr_scheduler == 'ExponentialLR':
    #     scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.1)
    # elif args.lr_scheduler == 'CosineAnnealingLR':
    #     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20)
    # elif args.lr_scheduler == 'CyclicLR':
    #     scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-7, max_lr=0.001,
    #                                         step_size_up=2000, step_size_down=2000, mode='triangular',
    #                                         gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False,
    #                                         base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
    # elif args.lr_scheduler == 'OneCycleLR':
    #     scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.9, total_steps=100, verbose=True)
    # elif args.lr_scheduler == 'CosineAnnealingWarmRestarts':
    #     scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    # elif args.lr_scheduler == 'ReduceLROnPlateau':
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.1,patience=2)
    epoch = 0
    if args.fine_tune:
        logger.info('-----------------------continue train------------------------')
        # torch.save(
        #     {"state": model.state_dict(),
        #      "epoch": epoch
        #      }
        #     , 'best_model.pt')
        checkpoint = torch.load(
            f'/mnt/e/deep_learning/outputs/outputs20230516/single_epoch2000weight0/contrast_Vcmax_A/{args.pretrain_name}/best_model_both.pt',
            map_location=device)
        model.load_state_dict(checkpoint['state'])
        # for p in model.parameters():#模型参数不更新
        #     p.requires_grad = False
        # for m in model.modules():#选定的bn模块参数更新,不固定
        #     if isinstance(m, nn.BatchNorm1d):
        #         m.required_grad = True

        # model.convblock1 = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=50, kernel_size=5, dilation=1),
        #     nn.BatchNorm1d(num_features=50),
        #     nn.ReLU()
        # )
        # model.convblock2 = nn.Sequential(
        #     nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5, dilation=2),
        #     nn.BatchNorm1d(num_features=50),
        #     nn.ReLU())
        # model.attention = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=50, nhead=5, dim_feedforward=50 * 4, batch_first=True), num_layers=2)
        # model.fc1 = nn.Linear(in_features=400, out_features=1000)
        # # model.relu2 = nn.ReLU()
        # # model.dropout = nn.Dropout(0.2)
        # model.fc2 = nn.Linear(in_features=1000, out_features=2)
        model.to(device)

        train_r2,  train_outputs, train_labels = evalution(args, model, train_loader)
        np.savez('train_out.npz',
                 labels=(train_labels * train_data.label_std + train_data.label_mean),  #
                 preds=(train_outputs * train_data.label_std + train_data.label_mean),  #

                 )
        logger.info(f"train_direct_transfer_r2: {train_r2} ")

        val_r2, val_outputs, val_labels = evalution(args, model, val_loader)
        np.savez('val_direct_transfer_out.npz',
                 labels=(val_labels * train_data.label_std + train_data.label_mean),  #
                 preds=(val_outputs * train_data.label_std + train_data.label_mean),  #
                 )
        logger.info(f"val_direct_transfer_r2:{val_r2}")  #
        test_r2, test_outputs, test_labels = evalution(args, model, test_loader)
        np.savez('test_direct_transfer_out.npz',
                 labels=(test_labels * train_data.label_std + train_data.label_mean),  #
                 preds=(test_outputs * train_data.label_std + train_data.label_mean),  #
                 )

        logger.info(f" test_direct_transfer_r2:{test_r2}")

    elif os.path.exists('best_model_both.pt'):
        logger.info('-----------------------continue train------------------------')
        # torch.save(
        #     {"state": model.state_dict(),
        #      "epoch": epoch
        #      }
        #     , 'best_model.pt')
        checkpoint = torch.load('best_model_both.pt', map_location=device)
        model.load_state_dict(checkpoint['state'])
        epoch = checkpoint['epoch']

    def get_params(model,weight_decay = 0.):
        decay_weights, not_decay_weights = [],[]
        for name,param in model.named_parameters():
            if 'att' in name or 'ref' in name:
                not_decay_weights += [param]
            else:
                decay_weights += [param]
        params = [{"params":decay_weights,"weight_decay":weight_decay},
                  {"params": not_decay_weights, "weight_decay": 0},
                  ]
        return params
    
    # optimizer = optim.Adam(get_params(model,weight_decay=args.weight_decay), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    optimizer = optim.RAdam(model.parameters(),weight_decay=args.weight_decay, 
                             lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-7, max_lr=0.001,
                                            step_size_up=2000, step_size_down=2000, mode='triangular',
                                            gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False,
                                            base_momentum=0.8, max_momentum=0.9, last_epoch=-1)

    #torch.set_float32_matmul_precision('high')

    # # API NOT FINAL
    # # default: optimizes for large models, low compile-time
    # #          and no extra memory usage
    #compiled_model = torch.compile(model)
    #
    # reduce-overhead: optimizes to reduce the framework overhead
    #                and uses some extra memory. Helps speed up small models
    # compiled_model = torch.compile(model, mode="reduce-overhead")

    # # max-autotune: optimizes to produce the fastest model,
    # #               but takes a very long time to compile
    #compiled_model = torch.compile(model, mode="max-autotune")#加速


    
    compiled_model = model#model.half().float()

    best_model_loss = 50000
    best_model_both_r2 = 0
    best_model_Vcmax_r2 = 0
    best_model_Jmax_r2 = 0
    train_logprog = tqdm(range(epoch, args.epochs), dynamic_ncols=True)
    # torch.save(model.state_dict(), '/mnt/e/deep_learning/hyperspec_rgb_photo/ckpt/model_0.pt')
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp == 1)
    #scaler = torch.cuda.amp.GradScaler()

    def mixup_data(x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        p = 0.25
        rand = np.random.uniform(0, 1)
        if rand>p:
            lam = 1
        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    epoch_stop=0
    evaluation=[]
    for epoch in train_logprog:
        model.train()
        outputs = []
        labels = []
        train_loss = []
        loss_attes = []
        sdr_losses = []
        for data in train_loader:
            # print(data)
            input = data['Spectral'].to(device=device, dtype=torch.float32)
            #indice = data['Indice'].to(device=device, dtype=torch.float32)
            if args.embedding:
                embedding = data['Embedding'].to(device=device, dtype=torch.float32)
            #img = data['himg'].to(device=device, dtype=torch.float32).cuda()
            # print(output)
            label = data['Label'].to(device=device, dtype=torch.float32)

            #input , label = mix_up(input,label)
            with torch.cuda.amp.autocast(dtype=torch.float16,enabled=args.amp==1):#混合精度
                if args.task == 'hyper':
                    output ,sdr_loss = compiled_model(input)  # ,input,indice
                    #loss_att = 0
                else:
                    output = compiled_model(img)
                # --------------------------------------------------------------------------
                loss = criterion(output, label)  + sdr_loss * 0.05

                # if args.task != 'hyper':
                #     input = img
                # input, targets_a, targets_b, lam = mixup_data(input, label,
                #                                        0.2, use_cuda)#0.2
                # input, targets_a, targets_b = map(Variable, (input,
                #                                             targets_a, targets_b))
                # output = compiled_model(input)
                # loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)

            # loss = loss_compres(output, label)
            # loss = loss_mse(output,label)+loss_mae(output, label)

            loss.backward()
            optimizer.step()

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            optimizer.zero_grad(set_to_none=True)#set to none faster

            scheduler.step()

            # if args.lr_scheduler in ['CyclicLR','OneCycleLR','CosineAnnealingWarmRestarts']:
            #     scheduler.step()

            outputs.append(np.array(output.detach().cpu().float().numpy()))
            labels.append(np.array(label.cpu().float()))
            train_loss.append(np.array(loss.detach().cpu().float().numpy()))
            #loss_attes.append(loss_att.detach().cpu().float().numpy())
            sdr_losses.append(sdr_loss.detach().cpu().float().numpy())

        # if args.lr_scheduler not in ['CyclicLR', 'OneCycleLR', 'CosineAnnealingWarmRestarts']:
        #     if args.lr_scheduler in ['ReduceLROnPlateau']:
        #         scheduler.step(metrics=np.mean(train_loss))
        #     else:
        #         scheduler.step()
        outputs = np.concatenate(outputs)  # n,3
        labels = np.concatenate(labels)  # n,3
        # print(outputs)
        # print(outputs.shape)
        train_avg_r2 = r2_score(labels, outputs)
        train_avg_loss = np.mean(train_loss)
        sdr_losses = np.mean(sdr_losses)
        #loss_attes = np.mean(loss_attes)
        # print(loss_attes)
        # print(train_avg_loss,np.mean(loss_attes))#max = -1
        # logger.info(
        #     f"train epoch:{epoch},avg_r2,{train_avg_r2}")
        train_logprog.set_postfix(epoch=epoch, avg_r2=train_avg_r2,all_loss=train_avg_loss,
                                  #att_loss=loss_attes,
                                  sdr_loss=sdr_losses)
        # print(outputs.shape,labels.shape)
        if (epoch + 1) % 50 == 0:
            avg_r2 = r2_score(labels, outputs)
            logger.info(
                f"train epoch:{epoch},avg_r2,{avg_r2}")

        val_r2,  val_outputs, val_labels = evalution(args, model, val_loader)
        val_loss = mean_squared_error(val_outputs, val_labels)
        val_loss_mean = np.mean(val_loss)
        tb.add_scalar("val_loss", val_loss_mean, epoch)

        test_r2,test_outputs, test_labels = evalution(args, model, test_loader)
        test_loss = mean_squared_error(test_outputs, test_labels) #测试集损失随epoch变化
        test_loss_mean = np.mean(test_loss) #测试集损失随epoch变化
        evaluation.append([train_avg_r2,train_avg_loss,val_r2,val_loss_mean,test_r2,test_loss_mean])

        avg_both_r2 = val_r2 /2  # val_r2

        if avg_both_r2 > best_model_both_r2:
            # print(outputs[:,0]-labels[:,0])
            best_model_both_r2 = avg_both_r2
            torch.save(
                {"state": model.state_dict(),  # 模型参数
                 "epoch": epoch
                 }
                , 'best_model_both.pt')
            # logger.info(f"val epoch:{epoch},best_model_r2,{best_model_r2}")
            logger.info(f"val epoch:{epoch},best_model_r2,{val_r2}")  # ,
            logger.info(f"test_avgr2:{test_r2}")
            epoch_stop=0
        
        if avg_both_r2 < best_model_both_r2:
            epoch_stop=epoch_stop+1
            if epoch_stop>=500:
                break
    evaluation = pd.DataFrame(evaluation)
    evaluation.to_csv(f'{args.model}_{args.label_name}_evaluation.csv',index=False)
        # if avg_Vcmax_r2 > best_model_Vcmax_r2:
        #     # print(outputs[:,0]-labels[:,0])
        #     best_model_Vcmax_r2 = avg_Vcmax_r2
        #     torch.save(
        #         {"state": model.state_dict(),
        #          "epoch": epoch
        #          }
        #         , 'best_model_Vcmax.pt')
        #     # logger.info(f"val epoch:{epoch},best_model_r2,{best_model_r2}")
        #     logger.info(
        #         f"val Vcmax epoch:{epoch},best_model_r2,{val_r2},Vcmax_r2:{val_Vcmax_r2},Jmax_r2:{val_Jmax_r2}")  # ,
        #     logger.info(f"test_Vcmax_avg_r2:{test_r2},test_Vcmax_r2:{test_Vcmax_r2},test_Jmax_r2:{test_Jmax_r2}")
        # if avg_Jmax_r2 > best_model_Jmax_r2:
        #     # print(outputs[:,0]-labels[:,0])
        #     best_model_Jmax_r2 = avg_Jmax_r2
        #     torch.save(
        #         {"state": model.state_dict(),
        #          "epoch": epoch
        #          }
        #         , 'best_model_Jmax.pt')
        #     # logger.info(f"val epoch:{epoch},best_model_r2,{best_model_r2}")
        #     logger.info(
        #         f"val Jmax epoch:{epoch},best_model_r2,{val_r2},Vcmax_r2:{val_Vcmax_r2},Jmax_r2:{val_Jmax_r2}")  # ,
        #     logger.info(f"test_Jmax_avg_r2:{test_r2},test_Vcmax_r2:{test_Vcmax_r2},test_Jmax_r2:{test_Jmax_r2}")

        # if (epoch+1)%50 == 0:
        #     torch.save(model.state_dict(), f'/mnt/e/deep_learning/hyperspec_rgb_photo/ckpt/model_{(epoch + 1)//50}.pt')
        #     np.savez('val_Chl_out.npz',
        #              #Vcmax_labels=(val_labels * train_data.label_std + train_data.label_mean),
        #              #Jmax_labels=(labels*train_data.label_std+train_data.label_mean),
        #              Chl_labels=(val_labels*train_data.label_std+train_data.label_mean),
        #              #Vcmax_preds=(val_outputs * train_data.label_std + train_data.label_mean),
        #              #Jmax_preds=(outputs*train_data.label_std+train_data.label_mean),
        #              Chl_preds=(val_outputs*train_data.label_std+train_data.label_mean)
        #              )
    if not args.data_clean:
        model.load_state_dict(torch.load('best_model_both.pt', map_location=device)['state'])
    if args.data_clean:
        test_loader = train_loader
    logger.info(f"---------------------------------best model results------------------------------------------")  #
    train_r2, train_outputs, train_labels = evalution(args, model,train_loader)

    np.savez('train_out.npz',
             labels=(train_labels * train_data.label_std + train_data.label_mean),  #

             preds=(train_outputs * train_data.label_std + train_data.label_mean),  #

             )
    logger.info(
        f"train_r2: {train_r2} ")

    val_r2,  val_outputs, val_labels = evalution(args, model, val_loader)
    np.savez('val_out.npz',
             labels=(val_labels * train_data.label_std + train_data.label_mean),  #

             preds=(val_outputs * train_data.label_std + train_data.label_mean),  #

             )
    logger.info(
        f"val_r2:{val_r2}")  #
    test_r2,  test_outputs, test_labels = evalution(args, model, test_loader)
    np.savez('test_out.npz',
             labels=(test_labels * train_data.label_std + train_data.label_mean),  #

             preds=(test_outputs * train_data.label_std + train_data.label_mean),  #

             )

    logger.info(
        f"test_r2:{test_r2}")


    with open('result.csv', mode='a') as f:
            f.write(
                f'{args.name} ,{train_r2},{val_r2},{test_r2} \n')

    # #
    # logger.info(f"lr:{lr},weight_decay:{weight_decay}")  #
    # history = {
    #     "lr": lr,
    #     "weight_decay": weight_decay,
    #     "train_Vcmax_r2": train_Vcmax_r2,
    #     "train_Jmax_r2": train_Jmax_r2,
    #     "val_Vcmax_r2": val_Vcmax_r2,
    #     "val_Jmax_r2": val_Jmax_r2,
    #     "test_Vcmax_r2": test_Vcmax_r2,
    #     "test_Jmax_r2": test_Jmax_r2,
    #     "optim": args.optim,
    #     "lr_scheduler": args.lr_scheduler,
    #     "batch_size": args.batch_size
    # }
    # json.dump(history, open(args.history_file, "a"), indent=2)


# @hydra.main(config_path="conf", config_name='config.yaml')
# def main(args):
#     try:
#         # print(args)
#         _main(args)

#     except Exception as e:
#         logger.exception(e)
#         # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
#         os._exit(1)


from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.stflow import LogFileWriter
import sacred

from omegaconf import DictConfig
@hydra.main(config_path="conf", config_name='config.yaml')
def main(cfg:DictConfig):
    try:
        if cfg.sacred:
            ex = Experiment(cfg.name)
            ex.observers.append(MongoObserver.create(url='localhost:27017',
                                                     db_name='sacred'))
            ex.add_config({'_args':cfg})
            @ex.main
            #@LogFileWriter(ex)
            def run(_args,_run):
                return _main(_args,_run)

            ex.run()
        else:
            _main(cfg)
    except Exception as e:
        #logger.exception("Some error happened")
        torch.cuda.empty_cache()
        logger.exception(e)
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)



if __name__ == "__main__":
    main()
