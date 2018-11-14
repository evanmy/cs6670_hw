def compute_mini_sift_desc(img, kp_locs, orientation_norm=False,
                           patch_size=32, num_spatial_bins=4, num_ori_bins=8):
    
    """ Compute the mini-SIFT descriptor described in the homework write-up
        NOTE : Orientation normalization is computed in image patch.
        HINT : `utils.crop_patch` and `utils.compute_histogram` will be useful.
    Args:
        [img]                   Shape:HxW   Input image (in grayscale).
        [kp_locs]               Shape:Nx2   Localtion of the keypoints: (row, col)
        [orientation_norm]      Boolean     Whether do orientation normalization.
        [patch_size]            Int         Size of the image patch.
        [num_spatial_bins]      Int         #spatial bins.
        [num_ori_bins]          Int         #bins for the orientation histogram.
    Rets:
        Shape Nxd where d = [num_spatial_bins]x[num_spatial_bins]x[num_ori_bins].
        The default settings hould produce Nx128.
    """

    ####### WAITING FOR ZHILU #######
    from skimage.filters import gaussian
    blurred_input = gaussian(img, sigma=3, output=None, mode='nearest')
    ######## Implement gaussian blur ########
    
    
    '''Orientation''' 
    padded_input = utils.pad_image(blurred_input, t=1, b=1, l=0, r=0, padding='replicate')
    row_grad = padded_input[0:-2, :] - padded_input[2:, :]

    padded_input = utils.pad_image(blurred_input, t=0, b=0, l=1, r=1, padding='replicate')
    col_grad = padded_input[:, 0:-2] - padded_input[:, 2:]

    grad = np.sqrt(row_grad**2+col_grad**2)
    theta = np.arctan2(col_grad, row_grad)
    theta = np.rad2deg(theta) # range is -pi to pi
    theta = (theta+360)%360   # range is 0 to 360

    '''Crop Patch'''
    descriptor = []
    for c in kp_locs:
        y, x = c

        if x+patch_size//2 > input.shape[1]:
            xmax = input.shape[1]
            xmin = input.shape[1]-patch_size
        else: 
            xmin = x-patch_size//2
            xmax = x+(patch_size-patch_size//2)    

        if y+patch_size//2 > input.shape[0]:
            ymax = input.shape[0]
            ymin = input.shape[0]-patch_size
        else: 
            ymin = y-patch_size//2
            ymax = y+(patch_size-patch_size//2)


        '''Patch Histogram'''        
        patch_angle = utils.crop_patch(theta, xmin, ymin, xmax, ymax)
        patch_grad = utils.crop_patch(grad, xmin, ymin, xmax, ymax)

        freq, edge = np.histogram(patch_angle.flatten(), 
                                  bins=num_ori_bins, 
                                  range=(0,360), 
                                  weights= patch_grad.flatten())
        '''Normalization'''
        if orientation_norm:
            max_idx = np.argmax(freq)
            patch_angle = patch_angle - edge[max_idx]
            patch_angle = (patch_angle+360)%360

        '''Features'''
        y_interval = patch_angle.shape[0]//num_spatial_bins
        x_interval = patch_angle.shape[1]//num_spatial_bins

        features = []
        for i in range(num_spatial_bins):
            for j in range(num_spatial_bins):
                pa = patch_angle[y_interval*i:y_interval*(i+1), 
                                 x_interval*j:x_interval*(j+1)]

                pg = patch_grad[y_interval*i:y_interval*(i+1), 
                                x_interval*j:x_interval*(j+1)]

                freq, edge = np.histogram(pa.flatten(), 
                                          bins=num_ori_bins, 
                                          range=(0,360), 
                                          weights= pg.flatten())     
                features += [freq]

        d = np.hstack(features)
        d = np.hstack(features)/np.linalg.norm(d, ord=2)
        descriptor += [d]
        
    return np.vstack(descriptor)