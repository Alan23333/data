from torch.utils.data import DataLoader
from importlib import import_module


class Data:
    def __init__(self, args):
        ### import module
        # 导入数据
        # import_module(data.cave) 动态导入data.cave lower 大写转小写
        m = import_module('data.' + args.dataset.lower()) # 1

        if args.dataset == 'CAVE':
            trainset = getattr(m, 'TrainSet')(args) # getattr调用函数 args是参数好吧
            self.loader_train = DataLoader(
                trainset,
                batch_size=args.batch_size,
                num_workers=args.n_threads,
                shuffle=True,
                pin_memory=not args.cpu
            )
            # print("2222222222222222222\n")
            # print(type(trainset))  <class 'data.cave.TrainSet'>
            # print("2222222222222222222\n")
            testset = getattr(m, 'TestSet')(args)
            self.loader_test = DataLoader(
                testset,
                batch_size=1,
                num_workers=1,
                shuffle=False,
                pin_memory=not args.cpu
            )


        elif args.dataset in ['Harvard', 'PU']:
            testset = getattr(m, 'TestSet')(args)
            self.loader_test = DataLoader(
                testset,
                batch_size=1,
                num_workers=1,
                shuffle=False,
                pin_memory=not args.cpu
            )

        else:
            raise SystemExit('Error: no such type of dataset!')