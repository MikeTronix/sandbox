# -*- coding: utf-8 -*-
"""
Created on Wed Jul 08 11:34:21 2015

@author: mike.wellfare

"""
import numpy as np

def scale(A, B, k):     # fill A with B scaled by k
    Y = A.shape[0]
    X = A.shape[1]
    for y in range(0, k):
        for x in range(0, k):
            A[y:Y:k, x:X:k] = B
            
def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).sum(-1).sum(1)

def scale_img_0to1_float(img):
    mx = img.max()
    mn = img.min()
    if mn >= mx:
        raise ValueError('Cannot find valid range of image values')
    tmp = (img.copy() - mn) / (mx - mn)
    return tmp

def convert_float_image_to_byte_full_range(img):
    immx = img.max()
    immn = img.min()
    if immx > immn:
        offs = immn
        scale = 255.0 / (immx - immn) 
        new_img = scale * (img.copy() - offs)
    else:
        scale = 1.0
        offs = 0.0
        new_img = img
    byte_img = np.uint8(new_img.copy())
    return byte_img, scale, offs
    
def show_float_image(label,img):
    import cv2
    tmp = convert_float_image_to_byte_full_range(img)[0]
    disp = np.dstack((tmp,tmp,tmp))
    cv2.imshow(label,disp)
    
def gauss_kern(sigma,hwsz,normalize=True):
    gconst = 0.5/(sigma*sigma)
    gpix = np.arange(-float(hwsz),float(hwsz+1),1.0)
    gval = np.exp(-gconst*gpix**2)
    kern = np.outer(gval,gval)
    if normalize:
        return kern / kern.sum()
    else:
        return kern

def gaussian_image(shape,meanXY,sigmaXY):
    '''return a floating point image of a gaussian profile
    given means and standard deviations in pixels
    '''
    gconstX = 0.5/(sigmaXY[0]*sigmaXY[0])
    gpixX = np.arange(-meanXY[0],float(shape[1]-meanXY[0]),1.0)
    gvalX = np.exp(-gconstX*gpixX**2)
    gconstY = 0.5/(sigmaXY[1]*sigmaXY[1])
    gpixY = np.arange(-meanXY[1],float(shape[0]-meanXY[1]),1.0)
    gvalY = np.exp(-gconstY*gpixY**2)
    img = np.outer(gvalY,gvalX)
    return img
    
def disk_kern(hwsz):
    width = 1 + (hwsz << 1)
    r2 = hwsz*hwsz
    kern = np.zeros((width,width),dtype=np.int32)
    for idx in range(width*width):
        tx = (idx % width) - hwsz
        ty = (idx // width) - hwsz
        if tx*tx + ty*ty <= r2:
            kern[hwsz+ty,hwsz+tx] = 1
    return kern
    
def whiten_image(img, whf_sz, sigma_tgt, sigma_bkg):
    import scipy.signal
    kern_b = gauss_kern(sigma_bkg,whf_sz)
    if sigma_tgt < 0.1:
        kern_t = np.zeros((2*whf_sz+1,2*whf_sz+1))
        kern_t[whf_sz,whf_sz] = 1.0
    else:
        kern_t = gauss_kern(sigma_tgt,whf_sz)
    kern_whf = kern_t - kern_b
    outimg = scipy.signal.correlate2d(np.float32(img), kern_whf,
                                      mode='same', boundary='symm')
    return outimg
        
def whiten_scaled(img, scl, whf_sz, sigma_tgt, sigma_bkg):
    import cv2
    if scl > 1:
        dsimg = downsample(img, scl)
    else:
        dsimg = img
    dsres = whiten_image(dsimg, whf_sz, sigma_tgt, sigma_bkg)
    return cv2.resize(dsres,(img.shape[1],img.shape[0]))
    
def tophat_image(img, thf_sz):
    import cv2
    outimg = cv2.morphologyEx(np.float32(img), 
                              cv2.MORPH_TOPHAT,
                              np.eye(1+2*int(thf_sz), dtype=np.uint8))
    return np.float32(outimg)
        
def top_bottom_hat_image(img, thf_sz):
    import cv2
    se = np.uint8(disk_kern(thf_sz))
    thimg = cv2.morphologyEx(np.float32(img), 
                              cv2.MORPH_TOPHAT,
                              se)
    bhimg = cv2.morphologyEx(np.float32(img), 
                              cv2.MORPH_BLACKHAT,
                              se)
    return np.float32(thimg - bhimg)
        
def tb_hat_multiscaled(img, scales, thf_sz):
    import cv2
    sf = 1
    sumi = np.zeros_like(img)
    for si in range(scales):
        if sf > 1:
            dsimg = downsample(img, sf)
        else:
            dsimg = img
        dsres = top_bottom_hat_image(dsimg, thf_sz)
        if sf > 1:
            res = cv2.resize(dsres,(img.shape[1],img.shape[0]))
        else:
            res = dsres
        sumi += res
        sf *= 2
    return sumi
    
def upsample(myarr, expand_by):
    """upsample an image, repeating each pixel value in a rectangular block within the result
    I have found that as_strided is much faster than a double repeat in many cases
    (for small arrays [<250x250] with only a doubling in each dimension, as_strided was slower).
    This works by using 0-length strides which causes numpy to read the same value multiple
    times (until it gets to the next dimension). The final reshape does copy the data, but only once
    unlike using a double repeat which will copy the data twice.
    author:  coderforlife on StackExchange
    """
    from numpy.lib.stride_tricks import as_strided
    # expand_by = number of times to replicate each point in each dimension
    if len(expand_by) == 0:
        N, M = expand_by, expand_by
    else:
        N, M = expand_by
    H, W = myarr.shape
    return as_strided(
            myarr, (H, N, W, M), 
            (myarr.strides[0], 0, myarr.strides[1], 0)).reshape((H*N, W*M))   
    
def downsample(myarr,factor,estimator=np.nanmean):
    """
    Downsample a 2D array by averaging over *factor* pixels in each axis.
    Crops upper edge if the shape is not a multiple of factor.
    This code is pure np and should be fast.
    keywords:
        estimator - default to mean.  You can downsample by summing or
            something else if you want a different estimator
            (e.g., downsampling error: you want to sum & divide by sqrt(n))
    """
    ys,xs = myarr.shape
    crarr = myarr[:ys-(ys % int(factor)),:xs-(xs % int(factor))]
    dsarr = estimator( np.concatenate([[crarr[i::factor,j::factor] 
        for i in range(factor)] 
        for j in range(factor)]), axis=0)
    return dsarr

def blank_borders(img, width, val=0):
    img[0:-width,0:width] = val
    img[0:width,width:] = val
    img[width:,-width:] = val
    img[-width:,0:-width] = val
    return img
        
def extend_line_to_borders(p1,p2,shape):
    nx1, ny1 = p1
    x1 = float(nx1)
    y1 = float(ny1)
    nx2, ny2 = p2
    x2 = float(nx2)
    y2 = float(ny2)
    pts = []
    if abs(x2-x1) > 1e-6:
        if abs(y2-y1) > 1e-6:
            m = (y2-y1) / (x2-x1)
            b = y2 - m * x2
            x_tp = int(-b/m)
            x_bt = int(-b/m + float(shape[0]-1)/m)
            y_lf = int(b)
            y_rt = int(b + float(shape[1]-1)*m)
            if (x_tp >= 0) and (x_tp < shape[1]):
                pts.append((x_tp,0))
            if (x_bt >= 0) and (x_bt < shape[1]):
                pts.append((x_bt,shape[0]-1))
            if (y_lf >= 0) and (y_lf < shape[0]):
                pts.append((0,y_lf))
            if (y_rt >= 0) and (y_rt < shape[0]):
                pts.append((shape[1]-1,y_rt))
        else:
            y1 = int(y1)
            if (y1 >= 0) and (y1 < shape[0]):
                pts.append((0,y1))
                pts.append((shape[1]-1,y1))
    else:
        x1 = int(x1)
        if (x1 >= 0) and (x1 < shape[1]):
            pts.append((x1,0))
            pts.append((x1,shape[0]-1))
    return pts

def gs_img_fig(title, img, vmin=None, vmax=None, pdf=False):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title(title)
    if vmin is not None and vmax is not None:
        ax.imshow(img,cmap='gray',vmin=vmin,vmax=vmax)
    else:
        ax.imshow(img,cmap='gray')        
    plt.show()
    if pdf:
        myPdf = PdfPages(filename='%s.pdf' % title, keep_empty=False)
        myPdf.savefigure(figure=fig, bbox_inches='tight')
        myPdf.close()
    return fig, ax
    
def gs_img_save(fname,img):
    fname += '.npy'
    fhandle = open(fname,'wb')
    np.save(fhandle,img)
    fhandle.close()

def gs_img_load(fname):
    fname += '.npy'
    fhandle = open(fname,'rb')
    out = np.load(fhandle)
    fhandle.close()
    return out
    
def clr_img_fig(title, img, pdf=False):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title(title)
    ax.imshow(img)
    fig.tight_layout()
    plt.show()
    if pdf:
        myPdf = PdfPages(filename='%s.pdf' % title, keep_empty=False)
        myPdf.savefigure(figure=fig, bbox_inches='tight')
        myPdf.close()
    return fig, ax

def clr_img_save(fname,img):
    fname += '.npy'
    fhandle = open(fname,'wb')
    np.save(fhandle,img)
    fhandle.close()

def clr_img_load(fname):
    fname += '.npy'
    fhandle = open(fname,'rb')
    out = np.load(fhandle)
    fhandle.close()
    return out   

def warpAffine(self, img, xform):
    import cv2
    myflags = cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS
    M = xform[0:2,0:3]
    wimg = cv2.warpAffine(img, M, img.shape[::-1], None, flags=myflags)
    return wimg
        

