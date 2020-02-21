import sys

from gomoku import train


def parse_args():
    args = {}
    for i, arg in enumerate(sys.argv):
        if i > 0 and arg.startswith('--'):
            k, v = tuple(arg[2:].split('='))
            args.update({k: eval(v)})
    return args


def main():
    args = parse_args()
    trainer = train.TrainPipeline(**args)
    trainer.run()


if __name__ == '__main__':
    main()
