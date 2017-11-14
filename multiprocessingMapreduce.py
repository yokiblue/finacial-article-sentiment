
import collections
import itertools
import multiprocessing

class simpleMapReduce(object):
    
    def __init__(self, map_func, reduce_func, num_workers=None):

        self.map_func = map_func
        self.reduce_func = reduce_func
        self.pool = multiprocessing.Pool(num_workers)
        self.exit = multiprocessing.Event()
    
    def partition(self, mapped_values):

        partitioned_data = collections.defaultdict(list)
        for key, value in mapped_values:
            partitioned_data[key].append(value)
        return partitioned_data.items()
    
    def __call__(self, inputs, chunksize=1):
        
        print (inputs)
        map_responses = self.pool.map(self.map_func, inputs, chunksize=chunksize)
        print ("testMap")
        partitioned_data = self.partition(itertools.chain(*map_responses))
        reduced_values = self.pool.map(self.reduce_func, partitioned_data)
        print ("Terminate multiprocessing")
        self.exit.set()
        return reduced_values
