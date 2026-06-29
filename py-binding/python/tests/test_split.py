import numpy as np
import alpha as al

def test_bw_split():
    price = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    dividend = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    transfer_shares = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right_shares = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    right_price = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    al.set_ctx(flags=0, groups=1)
    res = al.BW_SPLIT(price, dividend, transfer_shares, right_shares, right_price)
    
    assert np.allclose(res, np.array([4.0, 9.0, 10.0]))

def test_fw_split():
    price = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    dividend = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    transfer_shares = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right_shares = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    right_price = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    al.set_ctx(flags=0, groups=1)
    res = al.FW_SPLIT(price, dividend, transfer_shares, right_shares, right_price)
    
    assert np.allclose(res, np.array([10.0, 20.0, 22.0]))
