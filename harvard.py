import os
from data import common
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset


class TestSet(Dataset):
    def __init__(self, args):
        self.args = args
        data = sio.loadmat('data/Harvard/response coefficient')
        self.R = data['R']
        self.B = data['B']

        self.PrepareDataAndiniValue(self.R, self.B, self.args.scale, self.args.prepare)
        self._set_dataset_length()

    def __getitem__(self, idx):
        X, Y, Z, filename = self.all_test_data_in(idx)
        # D, A = common.Upsample(Y, Z, self.B, self.args.scale)
        Xes = common.Upsample2(Y, Z, self.B, self.args.scale)

        # A = common.np2Tensor(
        #     A, data_range=self.args.data_range
        # )
        Xes = common.np2Tensor(
            Xes, data_range=self.args.data_range
        )

        # return X, Y, Z, D, A, filename
        return X, Y, Z, Xes, filename

    def __len__(self):
        return self.dataset_length

    def _set_dataset_length(self):
        self.dataset_length = 10

    # Prepare dataset for testing
    def PrepareDataAndiniValue(self, R, C, sf, prepare='Yes', channel=29):
        DataRoad = 'data/Harvard/test/'
        if prepare != 'No':
            print('Generating the testing dataset in folder data/Harvard/test')

            common.mkdir(DataRoad + 'X/')
            common.mkdir(DataRoad + 'Y/')
            common.mkdir(DataRoad + 'Z/')

            for root, dirs, files in os.walk('data/Harvard/complete_ms_data/'):
                for i in range(10):
                    Z = np.zeros([1040 // sf, 1040 // sf, channel])
                    print('processing ' + str(i + 1))
                    data = sio.loadmat('data/Harvard/complete_ms_data/' + files[i])
                    X = data['ref']
                    X = X[:, 0:1040, 2:31] / np.max(X) * 255
                    Y = np.tensordot(X, R, (2, 0))
                    for j in range(channel):
                        subZ = np.real(np.fft.ifft2(np.fft.fft2(X[:, :, j]) * C))
                        subZ = subZ[0::sf, 0::sf]
                        Z[:, :, j] = subZ
                    sio.savemat(DataRoad + 'X/' + files[i], {'X': X})
                    sio.savemat(DataRoad + 'Y/' + files[i], {'Y': Y})
                    sio.savemat(DataRoad + 'Z/' + files[i], {'Z': Z})
                break

        else:
            print('Using the prepared testset and initial values in folder data/Harvard/test')

    def all_test_data_in(self, idx):
        for root, dirs, files in os.walk('data/Harvard/test/X/'):
            filename = files[idx]
            data = sio.loadmat('data/Harvard/test/X/' + filename)
            X = data['X']
            data = sio.loadmat('data/Harvard/test/Y/' + filename)
            Y = data['Y']
            data = sio.loadmat('data/Harvard/test/Z/' + filename)
            Z = data['Z']

        return X, Y, Z, filename