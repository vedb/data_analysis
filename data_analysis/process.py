import multiprocessing as mp
from multiprocessing import Pool
import pandas as pd
import numpy as np


class BaseProcess(mp.Process):
    """ A basic Python Process """

    def __init__(self, name, data_structure="pandas", data_size=100, *args, **kwargs):
        """ Constructor.

        Parameters
        ----------
        name: str
            The name of the process

        data_structure: str
            Data Structure format i.e. "pandas" or "array"

        data_size: int
            Number of rows of the output data structure
            i.e. pd.DataFrame()

        data_size: int
            Number of rows of the output data structure
            i.e. pd.DataFrame()

        args:
            Additional arguments

        kwargs:
            Additional keyword arguments
        """
        self.name = name
        self.data_structure = data_structure
        self.data_size = data_size
        self.output_data = self.create_output_data()
        self.args = args
        # Todo: Maybe pass this as a parameter?
        self.pool = Pool(processes=4)
        self.result = None
        super().__init__(target=self.target_method, name=name, *args, **kwargs)
        print("Parent Process {} for target {} initiated!".format(name, self.target_method))
        print("Number of args : {}".format(len(args)))
        print("kwargs : {}".format(kwargs.keys()))

    def target_method(self):
        def f(arguments=None):
            print("Parent Target Method is {}".format(arguments))
            return None
        print("Running Process {} on {} for {} frames".format(self.name, self.target_method, self.data_size))
        # self.result = self.pool.apply_async(self.target_method, args=self.args)
        self.result = self.pool.map(f, self.args)
        return self.result

    def create_output_data(self):
        df_keys = ["a", "b", "c"]
        df = pd.DataFrame(columns=df_keys, index=self.data_size, dtype=np.float64)
        return df

    def run(self):
        print("Running Process {} on {} for {} frames".format(self.name, self.target_method, self.data_size))
        self.result = self.pool.apply_async(self.target_method, args=self.args)
        return self.result

    def wait(self):
        if self.result is None:
            print("Process Result is None!!")
            return None
        else:
            print("Waiting on {} to finish!!".format(self.name))
            return self.result.wait()

    def close(self):
        self.pool.close()
