import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
from torch.autograd import Variable
from data import get_dataset  #@data.py
from preprocess import get_transform #@preprocess.py
from utils import * 
from datetime import datetime
from ast import literal_eval #用于字符串的类型转换
from torchvision.utils import save_image

#这一段不太懂，但应该是选models文件夹里的文件
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

#argparser，命令行传递参数
parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

#保存结果地址
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')                                          

#保存日志文件夹的名字（是上面result_dir的子目录
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')                                         

#数据名（好像文件夹也ok？）
parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name or folder')                             

#选模型' | '.join(model_names)说明help给出了可选
parser.add_argument('--model', '-a', metavar='MODEL', default='alexnet',
                    choices=model_names,                                         
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')

#输入图片的size
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')

#额外的模型设定，之后通过literal_eval转换
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')

#tensor的数据类型（float half 等等……）
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')

#一定要看看他们咋做多卡的
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')

#一定要看看他们咋多进程读数据的！！！！
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

#一共要跑多少个epoch呢～
parser.add_argument('--epochs', default=2500, type=int, metavar='N',
                    help='number of total epochs to run')

#起始epoch，因为学习率是随着epoch decay的，所以重新训练时最好规定从哪部分开始
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

#batch_size
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

#优化器选择，貌似是通过文件名传递的？？？？？？？？？？？
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')

#LR
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')

#momentum (if not SGD？？？？？？？？？？？？？？？
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

#weight(if not SGD????????????????????
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

#多少次更新打一次
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

#checkpoint的保存路径（check point是啥
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

#模型评价文件放哪
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')


def main():
    print('for this version, Datatype must be float tensor,modified would come as soon as possible') 
    global args, best_prec1 #声明全局变量
    best_prec1 = 0
    args = parser.parse_args() #全局变量初始化×2，args就是paser的结果
    
    #如果有模型评价就把最开始保存模型改了？？？
    if args.evaluate:
        args.results_dir = '/tmp'
    #如果保存那个每天就用年月日时分秒生成个文件名
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    #os.path三个最常见的用法都在这了
    save_path = os.path.join(args.results_dir, args.save) #拼接路径，中间自动+/
    if not os.path.exists(save_path):  #检测是否存在路径
        os.makedirs(save_path)         #mkdir

    #utils.py
    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')#.%s后面啥都不加出来的就只是.%s 是不是写错了？
    results = ResultsLog(results_file % 'csv', results_file % 'html')
    
    #python的logging标准库
    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    #寻找字符串子串666666
    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')] #教科书一样的拆字符串
        torch.cuda.set_device(args.gpus[0])                #应该是设为主控GPU
        cudnn.benchmark = True                             #经常反向优化的优化
    else:
        args.gpus = None

    # create model
    #看一下log.txt你就知道
    logging.info("creating model %s", args.model)
    #最最最最最上面我看不懂的那块xxxx
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset}

    #之前自定义的那个参数，literal_eval使用**前缀，参数会被认为是字典（一个*就是元组了）
    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    #使用**前缀，参数会被认为是字典（一个*就是元组了）
    model = model(**model_config)#此时，我们已经生成了模型！！！！！！
    logging.info("created model with configuration: %s", model_config)

    # optionally resume from a checkpoint
    #研究之前我得先知道checkpoint是干啥的orzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    #统计参数总量，肥肠有用！
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)#话说logging info除了打到日志里，还可以直接再shell里显示吗？

    # Data loading code
    #from preprocess.py import get_transform
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    
    #getattr() 函数用于返回一个对象属性值。
    '''
    getattr(a, 'bar2', 3)    # 属性 bar2 不存在，但设置了默认值3
    '''
    
    transform = getattr(model, 'input_transform', default_transform) 
    regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
                                           'lr': args.lr,
                                           'momentum': args.momentum,
                                           'weight_decay': args.weight_decay}})
    
    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    #tensor的数据类型
    criterion.type(args.type)
    model.type(args.type)

    #from data.py import get_dataset()
    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    train_data = get_dataset(args.dataset, 'train', transform['train'])
    
    #pin_memory=True属于锁页内存，转义到GPU的显存就会更快一些!!
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    logging.info('training regime: %s', regime)

    
    for epoch in range(args.start_epoch, args.epochs):
        #from utils.py import adjust_optimizer
        optimizer = adjust_optimizer(optimizer, epoch, regime)

        # train for one epoch,定义下面，但具体还要看model
        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, criterion, epoch, optimizer)

        # evaluate on validation set定义下面，但具体还要看model
        val_loss, val_prec1, val_prec5 = validate(
            val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1          #cool~学到了
        best_prec1 = max(val_prec1, best_prec1)   #还有这种操作
        # from utils.py import save_checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'config': args.model_config,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'regime': regime
        }, is_best, path=save_path)
        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))

        #results 是之前的 log
        results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                    train_error1=100 - train_prec1, val_error1=100 - val_prec1,
                    train_error5=100 - train_prec5, val_error5=100 - val_prec5)
        #results.plot(x='epoch', y=['train_loss', 'val_loss'],
        #             title='Loss', ylabel='loss')
        #results.plot(x='epoch', y=['train_error1', 'val_error1'],
        #             title='Error@1', ylabel='error %')
        #results.plot(x='epoch', y=['train_error5', 'val_error5'],
        #             title='Error@5', ylabel='error %')
        results.save()


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    #from utils import AverageMeter,但它有什么卯月呢？
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.cuda()
        #such two lines below can't work in pytroch1.0,so just瞎几把改一下试试
        #这段代码真滴是坑死我了orz,先这样吧orzzzzz其实还是要改的
#         input_var = Variable(inputs.type(args.type), volatile=not training)
#         target_var = Variable(target)

        
        # compute output
        output = model(inputs)
        loss = criterion(output, target)
        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, inputs.size(0)) #in pytorch1.0 data[0] could raise Index Error,same as prec1&5
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            for p in list(model.parameters()):
                #hasattr() 函数用于判断对象是否包含对应的属性。
                if hasattr(p,'org'):
                    p.data.copy_(p.org) #目测这一块是反向传播时区分量化后权重和全精度权重的梯度的
            optimizer.step()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp_(-1,1))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)


if __name__ == '__main__':
    main()
