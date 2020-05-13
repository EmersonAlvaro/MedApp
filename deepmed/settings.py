from os.path import abspath, dirname, join
from pathlib import Path


class AbsolutePathBuilder:

    def __init__(self, dir_name, dir_parrent_name):
        self.dir_name = dir_name
        self.dir_parrent_name = dir_parrent_name

    def  get_absolute_path(self):
        path = ['/']
        for dir in str(dirname(abspath(__file__))).split('/'):
            path.append('/'+dir)
            if str(dir) == self.dir_parrent_name:
                path.append('/'+self.dir_name)
                break

        return Path(''.join(path))

apb = AbsolutePathBuilder(dir_name='data', dir_parrent_name='deepmed')
data_dir = apb.get_absolute_path()
