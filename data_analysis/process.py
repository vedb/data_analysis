import multiprocessing as mp


class BaseProcess(mp.Process):
    """ A basic Python Process """

    def __init__(self, target, name, *args):

        super().__init__(target=target, name=name, args=args)
        print("Process {} for target {} initiated!".format(name, target))
        print("arg : {}".format(p for p in args))
