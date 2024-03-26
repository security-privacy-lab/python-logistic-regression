from data import x, y
import pandas as pd
import numpy as np
from Pyfhel import Pyfhel





def df_to_fhe_df(x, y):
    HE = Pyfhel()                               # Creating empty Pyfhel object
    ckks_params = {
        'scheme': 'CKKS',                       # can also be 'ckks'
        'n': 2 ** 14,                           # Polynomial modulus degree. For CKKS, n/2 values can be
                                                #  encoded in a single ciphertext.
                                                #  Typ. 2^D for D in [10, 15]
        'scale': 2 ** 30,                       # All the encodings will use it for float->fixed point
                                                #  conversion: x_fix = round(x_float * scale)
                                                #  You can use this as default scale or use a different
                                                #  scale on each operation (set in HE.encryptFrac)
        'qi_sizes': [60, 30, 30, 30, 60]        # Number of bits of each prime in the chain.
                                                # Intermediate values should be  close to log2(scale)
                                                # for each operation, to have small rounding errors.
    }
    HE.contextGen(**ckks_params)                # Generate context for ckks scheme
    HE.keyGen()                                 # Key Generation: generates a pair of public/secret keys
    HE.rotateKeyGen()



    c_x = pd.DataFrame().reindex_like(x)        # This copies the dimensions of the original DataFrame, preparing it
                                                #  for the encryption
    for i, row in x.iterrows():
        npRow = row.to_numpy(dtype=np.float64)

        for j, value in np.ndenumerate(npRow):


            pass

        ptxt_x = HE.encodeFrac(npRow)
        ctxt_x = HE.encryptPtxt(ptxt_x)




        # code here



        seriesRow = pd.Series(npRow)
        c_x.loc[i] = seriesRow.values


    return c_x

    print()

x_c = df_to_fhe_df(x, y)
x = 1
