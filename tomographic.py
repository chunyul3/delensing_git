def W_tmg(z, suvy):
    if suvy == 'LSST':
        wg = np.zeros((10, len(z)))
        zi0 = np.array([0.4, 0.6, 0.9, 1.2, 1.5, 1.9, 2.2, 2.8, 3.4])
        zim = 0.82 * zi0 - 0.127
        zi = np.array([0] + list(zim) + [3.5])
        for i in range(10):            
            wg[i,:] = 1/2 * dn_dz(z, suvy) * ( special.erfc( (zi[i] - z) / np.sqrt(2) / 0.03 / (1 + z) ) - special.erfc( (zi[i + 1] - z) / np.sqrt(2) / 0.03 / (1 + z) ))
            sumwg = sum(wg[i,:])
            wg[i,:] = wg[i,:] * b(z, suvy)
            wg[i,:] /= sumwg
    elif suvy == 'DES':
        wg = np.zeros((4, len(z)))
        zi0 = np.array([0.5, 0.55, 0.6, 0.65, 0.7])
        zi = zi0
        for i in range(10):            
            wg[i,:] = 1/2 * dn_dz(z, suvy) * ( special.erfc( (zi[i] - z) / np.sqrt(2) / 0.03 / (1 + z) ) - special.erfc( (zi[i+1] - z) / np.sqrt(2) / 0.03 / (1 + z) ))
            wg[i,:] = wg[i,:] * b(z, suvy)
            sumwg = sum(wg[i,:])
            wg[i,:] = wg[i,:] * b(z, suvy)
            wg[i,:] /= sumwg
    return wg

