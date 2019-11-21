# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 19:13:26 2017

@author: mike.wellfare
"""

import sys
import math
import random
import numpy as np
import time
import cv2
from copy import copy
from img_utils import gauss_kern
from copy_patch_centered import copy_patch_centered
from numpngw import write_png

from direct.showbase.ShowBase import ShowBase
from panda3d.core import FrameBufferProperties, WindowProperties
from panda3d.core import GraphicsPipe, GraphicsOutput
from panda3d.core import Material, TransparencyAttrib, AntialiasAttrib
from panda3d.core import Filename, PNMImage, ModelNode
from panda3d.core import loadPrcFileData
from panda3d.core import GeoMipTerrain
from panda3d.core import CardMaker
from panda3d.core import PointLight, AmbientLight
from panda3d.core import Texture, TextureStage
from panda3d.core import BitMask32
from panda3d.core import Point3, Vec3, VBase4

HORZ_RES_FACTOR = 1

loadPrcFileData('', 'show-frame-rate-meter true')
loadPrcFileData('', 'sync-video 0')
loadPrcFileData("", "LIDAR Simulator")
loadPrcFileData("", "fullscreen 0") # Set to 1 for fullscreen
loadPrcFileData("", "win-size %i %i" % (960,960))
loadPrcFileData("", "win-origin 20 20")
loadPrcFileData("", "framebuffer-multisample 1")
loadPrcFileData("", "multisamples 4")

def show_rgbd_image(image, depth_image, window_name='Image window', delay=1, depth_offset=0.0, depth_scale=1.0):
    if depth_image.dtype != np.uint8:
        if depth_scale is None:
            depth_scale = depth_image.max() - depth_image.min()
        if depth_offset is None:
            depth_offset = depth_image.min()
        depth_image = np.clip((depth_image - depth_offset) / depth_scale, 0.0, 1.0)
        depth_image = (255.0 * depth_image).astype(np.uint8)
    depth_image = np.tile(depth_image, (1, 1, 3))
    if image.shape[2] == 4:  # add alpha channel
        alpha = np.full(depth_image.shape[:2] + (1,), 255, dtype=np.uint8)
        depth_image = np.concatenate([depth_image, alpha], axis=-1)
    images = np.concatenate([image, depth_image], axis=1)
    # images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)  # not needed since image is already in BGR format
    cv2.imshow(window_name, images)
    key = cv2.waitKey(delay)
    key &= 255
    if key == 27 or key == ord('q'):
        print("Pressed ESC or q, exiting")
        exit_request = True
    else:
        exit_request = False
    return exit_request

def show_lidar_image(iimg, limg, window_name='LIDAR window', delay=1, range_offs=0.0, range_sc = 1.0):
    if limg.dtype != np.uint8:
        if range_sc is None:
            range_sc = limg.max() - limg.min()
        if range_offs is None:
            range_offs = limg.min()
        limg = np.clip((limg - range_offs) / range_sc, 0.0, 1.0)
        limg = (255.0 * limg).astype(np.uint8)
    limg = np.dstack((limg,limg,limg))
    if iimg.dtype != np.uint8:
        intens_sc = iimg.max() - iimg.min()
        intens_offs = iimg.min()
        iimg = np.clip((iimg - intens_offs) / intens_sc, 0.0, 1.0)
        iimg = (255.0 * iimg).astype(np.uint8)
    iimg = np.dstack((iimg,iimg,iimg))    
    print('limg:',limg.dtype,limg.shape,', iimg:',iimg.dtype,iimg.shape)
    dimg = np.vstack((limg,iimg))
    cv2.imshow(window_name, dimg)

class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Load the environment model.
        self.setup_environment()
        #self.scene = self.loader.loadModel("models/environment")
        # Reparent the model to render.
        #self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        #self.scene.setScale(0.25, 0.25, 0.25)
        #self.scene.setPos(-8, 42, 0)

        # Needed for camera image
        self.dr = self.camNode.getDisplayRegion(0)

        # Needed for camera depth image
        winprops = WindowProperties.size(self.win.getXSize(), self.win.getYSize())
        fbprops = FrameBufferProperties()
        fbprops.setDepthBits(1)
        self.depthBuffer = self.graphicsEngine.makeOutput(
            self.pipe, "depth buffer", -2,
            fbprops, winprops,
            GraphicsPipe.BFRefuseWindow,
            self.win.getGsg(), self.win)
        self.depthTex = Texture()
        self.depthTex.setFormat(Texture.FDepthComponent)
        self.depthBuffer.addRenderTexture(self.depthTex,
            GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth)
        lens = self.cam.node().getLens()
        lens.setFov(90.0,90.0)
        # the near and far clipping distances can be changed if desired
        # lens.setNear(5.0)
        # lens.setFar(500.0)
        self.depthCam = self.makeCamera(self.depthBuffer,
            lens=lens,
            scene=self.render)
        self.depthCam.reparentTo(self.cam)

        # TODO: Scene is rendered twice: once for rgb and once for depth image.
        # How can both images be obtained in one rendering pass?
        self.render.setAntialias(AntialiasAttrib.MAuto)
        
    def setup_environment(self):
        # encapsulate some stuff
        
        # set up ambient lighting
        self.alight = AmbientLight('alight')
        self.alight.setColor(VBase4(0.1,0.1,0.1,1))
        self.alnp = self.render.attachNewNode(self.alight)
        self.render.setLight(self.alnp)

        # set up a point light        
        self.plight = PointLight('plight')
        self.plight.setColor(VBase4(0.8,0.8,0.8,1))
        self.plnp = self.render.attachNewNode(self.plight)
        self.plnp.setPos(0,0,100)
        self.render.setLight(self.plnp)
        
        # set up terrain model
        self.terr_material = Material()
        self.terr_material.setShininess(1.0)
        self.terr_material.setAmbient(VBase4(0,0,0,0))
        self.terr_material.setDiffuse(VBase4(1,1,1,1))
        self.terr_material.setEmission(VBase4(0,0,0,0))
        self.terr_material.setSpecular(VBase4(0,0,0,0))

        # general scaling
        self.trrHorzSc = 4.0
        self.trrVertSc = 4.0 # was 4.0
        
        # Create sky
        #terrctr = self.trrHorzSc*65.0
        #self.setup_skybox(terrctr,800.0,2.0,0.3)
        
        self.skysphere = self.loader.loadModel("sky-forest/SkySphere.bam")
        self.skysphere.setBin('background', 1)
        self.skysphere.setDepthWrite(0) 
        self.skysphere.reparentTo(self.render)

        # Load some textures
        self.grsTxtSc = 5
        self.numTreeTexts = 7
    
        # ground texture
        self.txtGrass = self.loader.loadTexture('tex/ground005.png')
        self.txtGrass.setWrapU(Texture.WM_mirror)
        self.txtGrass.setWrapV(Texture.WM_mirror)
        self.txtGrass.setMagfilter(Texture.FTLinear)
        self.txtGrass.setMinfilter(Texture.FTLinearMipmapLinear)
        
        # set up terrain texture stages        
        self.TS1 = TextureStage('terrtext')
        self.TS1.setSort(0)
        self.TS1.setMode(TextureStage.MReplace)
    
        # Set up the GeoMipTerrain
        self.terrain = GeoMipTerrain("myDynamicTerrain")
        img = PNMImage(Filename('tex/bowl_height_map.png'))
        self.terrain.setHeightfield(img)
        self.terrain.setBruteforce(0)
        self.terrain.setAutoFlatten(GeoMipTerrain.AFMMedium)
        
        # Set terrain properties
        self.terrain.setBlockSize(32)
        self.terrain.setNear(50)
        self.terrain.setFar(500)
        self.terrain.setFocalPoint(self.camera)
        
        # Store the root NodePath for convenience
        self.root = self.terrain.getRoot()
        self.root.clearTexture()
        self.root.setTwoSided(0)
        self.root.setCollideMask(BitMask32.bit(0))
        self.root.setSz(self.trrVertSc)
        self.root.setSx(self.trrHorzSc)
        self.root.setSy(self.trrHorzSc)
        self.root.setMaterial(self.terr_material)
        self.root.setTexture(self.TS1, self.txtGrass)
        self.root.setTexScale(self.TS1, self.grsTxtSc, self.grsTxtSc)
        offset = 0.5 * img.getXSize() * self.trrHorzSc - 0.5
        self.root.setPos(-offset,-offset,0)       
     
        self.terrain.generate()
        self.root.reparentTo(self.render)

        # load tree billboards
        self.txtTreeBillBoards = []
        for a in range(self.numTreeTexts):
            fstr = 'trees/tree' + '%03d' % (a+991)
            self.txtTreeBillBoards.append( \
                self.loader.loadTexture(fstr + '-color.png', fstr + '-opacity.png'))
            self.txtTreeBillBoards[a].setMinfilter(Texture.FTLinearMipmapLinear)
            
        #self.placePlantOnTerrain('trees',300,0,20,20,self.trrHorzSc,self.trrVertSc, \
        #    self.numTreeTexts,self.txtTreeBillBoards,'scene-def/trees.txt')
        self.setup_house()
        self.setup_vehicle()

        self.taskMgr.add(self.skysphereTask, "SkySphere Task")
        
    def setup_house(self):
        # place farmhouse on terrain
        self.house = ModelNode('house1')
        self.loadModelOntoTerrain(self.render,
                                  self.terrain,
                                  self.house,
                                  43.0,0.275,0.0,0.0, 
                                  self.trrHorzSc,
                                  self.trrVertSc,
                                  'models/FarmHouse',
                                  Vec3(0,0,0), 
                                  Point3(-12.0567,-29.1724,0.0837742),
                                  Point3(12.2229,21.1915,21.3668))                         
    def setup_vehicle(self):
        # place HMMWV on terrain
        self.hmmwv = ModelNode('hmmwv1')
        self.loadModelOntoTerrain(self.render,
                                  self.terrain,
                                  self.hmmwv,
                                  33.0, 1.0, 20.0, 24.0, 
                                  self.trrHorzSc,
                                  self.trrVertSc,
                                  'models/hmmwv',
                                  Vec3(0,-90,0), 
                                  Point3(-1.21273,-2.49153,-1.10753),
                                  Point3(1.21273,2.49153,1.10753))                         

        
    def setup_skybox(self, terrctr=645.0, boxsz=1000.0, aspect=1.0, uplift=0.0):
        vsz = boxsz/aspect
        self.bckgtx = []
        self.bckgtx.append(self.loader.loadTexture('sky/Back2.png'))
        self.bckgtx.append(self.loader.loadTexture('sky/Right2.png'))
        self.bckgtx.append(self.loader.loadTexture('sky/Front2.png'))
        self.bckgtx.append(self.loader.loadTexture('sky/Left2.png'))
        self.bckgtx.append(self.loader.loadTexture('sky/Up.png'))
        for a in range(4):
            self.bckg = CardMaker('bkcard')
            lr = Point3(0.5*boxsz,0.5*boxsz,-0.5*vsz)
            ur = Point3(0.5*boxsz,0.5*boxsz,0.5*vsz)
            ul = Point3(-0.5*boxsz,0.5*boxsz,0.5*vsz)
            ll = Point3(-0.5*boxsz,0.5*boxsz,-0.5*vsz)
            self.bckg.setFrame(ll,lr,ur,ul)
            self.bckg.setHasNormals(0)
            self.bckg.setHasUvs(1)
            #self.bckg.setUvRange(self.bckgtx[a])
            bkcrd = self.render.attachNewNode(self.bckg.generate())
            bkcrd.setTexture(self.bckgtx[a])
            self.bckgtx[a].setWrapU(Texture.WMClamp)
            self.bckgtx[a].setWrapV(Texture.WMClamp)
            bkcrd.setLightOff()
            bkcrd.setFogOff() 
            bkcrd.setHpr(90.0*a,0,0)
            cz = 0.5*boxsz*uplift
            #print 'set card at:', terrctr,terrctr,cz, ' with points: ', lr,ur,ul,ll
            bkcrd.setPos(terrctr, terrctr, cz)
            self.top = CardMaker('bkcard')
            lr = Point3(0.5*boxsz,-0.5*boxsz,0)
            ur = Point3(0.5*boxsz,0.5*boxsz,0)
            ul = Point3(-0.5*boxsz,0.5*boxsz,0)
            ll = Point3(-0.5*boxsz,-0.5*boxsz,0)
            self.top.setFrame(ll,lr,ur,ul)
            self.top.setHasNormals(0)
            self.top.setHasUvs(1)
            #self.top.setUvRange(self.bckgtx[4])
            bkcrd = self.render.attachNewNode(self.bckg.generate())
            bkcrd.setTexture(self.bckgtx[4])
            self.bckgtx[4].setWrapU(Texture.WMClamp)
            self.bckgtx[4].setWrapV(Texture.WMClamp)
            bkcrd.setLightOff()
            bkcrd.setFogOff()
            bkcrd.setHpr(0,90,90)
            bkcrd.setPos(terrctr, terrctr, 0.5*vsz+0.5*boxsz*uplift)
    
    def placePlantOnTerrain(self,itemStr,itemCnt,Mode,typItemWidth,
                              typItemHeight,trrHorzSc,trrVertSc, 
                              numTxtTypes,txtList,planFileName):
        # Billboarding plants
        crd = CardMaker('mycard')
        crd.setColor(0.5,0.5,0.5,1)
        ll = Point3(-0.5*typItemWidth, 0, 0)
        lr = Point3( 0.5*typItemWidth, 0, 0)
        ur = Point3( 0.5*typItemWidth, 0, typItemHeight)
        ul = Point3(-0.5*typItemWidth, 0, typItemHeight)
        crd.setFrame(ll,lr,ur,ul)
        crd.setHasNormals(False)
        crd.setHasUvs(True)
        # generate/save/load locations
        try:
          plan_data_fp = open(planFileName,'r')
          item_list = []
          for line in plan_data_fp:
            toks = line.split(',')
            px = float(toks[0].strip(' '))
            py = float(toks[1].strip(' '))
            ang = float(toks[2].strip(' '))
            dht = float(toks[3].strip(' '))
            scl = float(toks[4].strip(' '))
            idx = int(toks[5].strip(' '))
            item_list.append((px,py,ang,dht,scl,idx))
          plan_data_fp.close()
          print 'loaded ', itemStr, ' data file of size:', len(item_list)
        except IOError:
            # generate list and try to save
            item_list = []
            for a in range(itemCnt):
              px = random.randrange(-self.trrHorzSc*64, self.trrHorzSc*64)
              py = random.randrange(-self.trrHorzSc*64, self.trrHorzSc*64)
              ang = 180*random.random()
              dht = 0.0
              scl = 0.75 + 0.25 * (random.random() + random.random())
              idx = random.randrange(0,numTxtTypes)
              item_list.append( [ px,py,ang,dht,scl,idx ] )
            try:
              plan_data_fp = open(planFileName,'w')
              for c in item_list:
                print >> plan_data_fp, c[0],',',c[1],',',c[2],',',c[3],',',c[4],',',c[5]
              plan_data_fp.close()
              print 'saved ', itemStr, ' data of size: ', len(item_list)
            except IOError:
              print 'unable to store ', itemStr, ' data of size: ', len(item_list)
        # define each plant
        for c in item_list:
          px = c[0]
          py = c[1]
          ang = c[2]
          dht = c[3]
          scl = c[4]
          idx = c[5]
          if idx >= numTxtTypes:
              idx = 0
          if Mode > 0:
              for b in range(Mode):
                  crdNP = self.render.attachNewNode(crd.generate())
                  crdNP.setTexture(txtList[idx])
                  crdNP.setScale(scl)
                  crdNP.setTwoSided(True)
                  ht = self.terrain.getElevation(px/trrHorzSc,py/trrHorzSc)
                  crdNP.setPos(px,py,ht*trrVertSc+dht)
                  crdNP.setHpr(ang+(180/Mode)*b,0,0)
                  crdNP.setTransparency(TransparencyAttrib.MAlpha)    
                  crdNP.setLightOff()
          else:
              # set up item as defined
              crd.setUvRange(txtList[idx])
              crdNP = self.render.attachNewNode(crd.generate())
              crdNP.setBillboardAxis()
              crdNP.setTexture(txtList[idx])
              crdNP.setScale(scl)
              ht = self.terrain.getElevation(px/trrHorzSc,py/trrHorzSc)
              crdNP.setPos(px,py,ht*trrVertSc)
              crdNP.setTransparency(TransparencyAttrib.MAlpha)    
              crdNP.setLightOff()
    
    def loadModelOntoTerrain(self,render_node,terr_obj,model_obj, 
                               hdg,scl,xctr,yctr,terr_horz_sc,terr_vert_sc, 
                               model_path,rotA,minP,maxP):
        # load model onto terrain
        hdg_rads = hdg*math.pi/180.0
        model_obj = self.loader.loadModel(model_path)
        rotAll = rotA
        rotAll.setX(rotAll.getX() + hdg)
        model_obj.setHpr(rotA)
        model_obj.setLightOff()
        # if model changes, these will have to be recomputed
        # minP = Point3(0,0,0)
        # maxP = Point3(0,0,0)
        # model_obj.calcTightBounds(minP,maxP)
        print minP
        print maxP
        htl = []
        maxzofs = -1000.0
        for xi in [ minP[0], maxP[0] ]:
            for yi in [ minP[1], maxP[1] ]:
                tx = xctr + scl*xi*math.cos(hdg_rads)
                ty = yctr + scl*yi*math.sin(hdg_rads)
                tht = self.terrain.getElevation(tx/terr_horz_sc, ty/terr_horz_sc)
                print 'tx=', tx, ', ty=', ty, ', tht=', tht 
                htl.append(tht*terr_vert_sc - minP.getZ())
        for hi in htl:
            if hi > maxzofs:
                maxzofs = hi
        print maxzofs
        model_obj.setPos(xctr,yctr,maxzofs)
        model_obj.setHpr(rotAll)
        model_obj.setScale(scl)
        model_obj.reparentTo(render_node)
        return maxzofs, minP, maxP
        
    def get_camera_image(self, requested_format=None):
        """
        Returns the camera's image, which is of type uint8 and has values
        between 0 and 255.
        The 'requested_format' argument should specify in which order the
        components of the image must be. For example, valid format strings are
        "RGBA" and "BGRA". By default, Panda's internal format "BGRA" is used,
        in which case no data is copied over.
        """
        tex = self.dr.getScreenshot()
        if requested_format is None:
            data = tex.getRamImage()
        else:
            data = tex.getRamImageAs(requested_format)
        image = np.frombuffer(data.get_data(), np.uint8)  # use data.get_data() instead of data in python 2
        image.shape = (tex.getYSize(), tex.getXSize(), tex.getNumComponents())
        image = np.flipud(image)
        return image

    def get_camera_depth_image(self):
        """
        Returns the camera's depth image, which is of type float32 and has
        values between 0.0 and 1.0.
        """
        data = self.depthTex.getRamImage()
        depth_image = np.frombuffer(data.get_data(), np.float32)
        depth_image.shape = (self.depthTex.getYSize(), self.depthTex.getXSize(), self.depthTex.getNumComponents())
        depth_image = np.flipud(depth_image)
        '''
        
        Surface position can be inferred by calculating backward from the
        depth buffer. Each pixel on the screen represents a ray from the
        camera into the scene, and the depth value in the pixel indicates a
        distance along the ray. Because of this, it is not actually necessary
        to store surface position explicitly - it is only necessary to store
        depth values. Of course, OpenGL does that for free.

        So the framebuffer now needs to store surface normal, diffuse color,
        and depth value (to infer surface position). In practice, most
        ordinary framebuffers can only store color and depth - they don't have
        any place to store a third value. So we need to use a special
        offscreen buffer with an "auxiliary" bitplane. The auxiliary bitplane
        stores the surface normal.

        So then, there's the final postprocessing pass. This involves
        combining the diffuse color texture, the surface normal texture, the
        depth texture, and the light parameters into a final rendered output.
        The light parameters are passed into the postprocessing shader as
        constants, not as textures.

        If there are a lot of lights, things get interesting. You use one
        postprocessing pass per light. Each pass only needs to scan those
        framebuffer pixels that are actually in range of the light in
        question. To traverse only the pixels that are affected by the light,
        just render the illuminated area's convex bounding volume.

        The shader to store the diffuse color and surface normal is trivial.
        But the final postprocessing shader is a little complicated. What
        makes it tricky is that it needs to regenerate the original surface
        position from the screen position and depth value. The math for that
        deserves some explanation.

        We need to take a clip-space coordinate and depth-buffer value
        (ClipX,ClipY,ClipZ,ClipW) and unproject it back to a view-space
        (ViewX,ViewY,ViewZ) coordinate. Lighting is then done in view-space.

        Okay, so here's the math. Panda uses the projection matrix to
        transform view-space into clip-space. But in practice, the projection
        matrix for a perspective camera always contains four nonzero
        constants, and they're always in the same place:
            
        -- here are the non-zero elements of the projection matrix --
        
        A	0	0	0
        0	0	B	1
        0	C	0	0
        0	0	D	0
        
        -- precompute these from above projection matrix --
        '''
        proj = self.cam.node().getLens().getProjectionMat()
        proj_x = 0.5 * proj.getCell(3, 2) / proj.getCell(0, 0)
        proj_y = 0.5 * proj.getCell(3, 2)
        proj_z = 0.5 * proj.getCell(3, 2) / proj.getCell(2, 1)
        proj_w = -0.5 - 0.5 * proj.getCell(1, 2)
        '''
        -- now for each pixel compute viewpoint coordinates --
        
        viewx = (screenx * projx) / (depth + projw)
        viewy = (1 * projy) / (depth + projw)
        viewz = (screeny * projz) / (depth + projw)
        '''
        grid = np.mgrid[0:depth_image.shape[0],0:depth_image.shape[1]]
        ygrid = np.float32(np.squeeze(grid[0,:,:])) / float(depth_image.shape[0] - 1)
        ygrid -= 0.5
        xgrid = np.float32(np.squeeze(grid[1,:,:])) / float(depth_image.shape[1] - 1)
        xgrid -= 0.5
        xview = 2.0 * xgrid * proj_x
        zview = 2.0 * ygrid * proj_z
        denom = np.squeeze(depth_image) + proj_w
        xview = xview / denom
        yview = proj_y / denom
        zview = zview / denom
        sqrng = xview**2 + yview**2 + zview**2
        range_image = np.sqrt(sqrng)
        range_image_1 = np.expand_dims(range_image,axis=2)
        
        return depth_image, range_image_1

    def compute_sample_pattern(self, limg_shape, res_factor):
        # assume velocity is XYZ and we are looking +X up and towards -Z
        pattern = []
        lens = self.cam.node().getLens()
        sx = self.win.getXSize()
        sy = self.win.getYSize()
        ifov_vert = 2.0*math.tan(0.5*math.radians(lens.getVfov()))/float(sy-1)
        ifov_horz = 2.0*math.tan(0.5*math.radians(lens.getHfov()))/float(sx-1)
        #ifov_vert = lens.getVfov() / float(sy-1)
        #ifov_horz = lens.getHfov() / float(sy-1)
        for ldr_row in range(limg_shape[0]):
            theta = -10.0 - 41.33 * (float(ldr_row) / float(limg_shape[0]-1) - 0.5)
            for ldr_col in range(limg_shape[1]):
                psi = 60.0 * (float(ldr_col) / float(limg_shape[1]-1) - 0.5)
                cpsi = math.cos(math.radians(psi))
                vert_ang = theta / cpsi 
                img_row_flt = (0.5*float(sy - 1) -
                                    (math.tan(math.radians(vert_ang)) / ifov_vert))
                #img_row_flt = 0.5*(sy-1) - (vert_ang / ifov_vert)
                if img_row_flt < 0:
                    print('img_row_flt=%f' % img_row_flt)
                    img_row_flt = 0.0
                if img_row_flt >= sy:
                    print('img_row_flt=%f' % img_row_flt)
                    img_row_flt = float(sy - 1)
                img_col_flt = (0.5*float(sx - 1) +
                                    (math.tan(math.radians(psi)) / ifov_horz))
                #img_col_flt = 0.5*(sx-1) + (psi / ifov_horz)
                if img_col_flt < 0:
                    print('img_col_flt=%f' % img_col_flt)
                    img_col_flt = 0.0
                if img_col_flt >= sx:
                    print('img_col_flt=%f' % img_col_flt)
                    img_col_flt = float(sx - 1)
                pattern.append((ldr_row,ldr_col,img_row_flt,img_col_flt))
        return pattern

    def find_sorted_ladar_returns(self, rangearr, intensarr, ks_m):
        my_range = rangearr.copy()
        my_inten = intensarr.copy()
        '''
        pixels data is organized by:
           [0] starting range of this return
           [1] ending range of this return
           [2] peak range of this return
           [3] total intensity of this return
        '''
        int_mult = len(my_inten)
        pixels = map(list, 
                     zip(my_range.tolist(),
                         my_range.tolist(),
                         my_range.tolist(),
                         my_inten.tolist()))
        spix = sorted(pixels, key=lambda x: x[0])
        done = False
        while not done:
            mxpi = len(spix)
            if mxpi > 2:
                mindel = 1e20
                mnidx = None
                for pidx in range(mxpi-1):
                    rdel = spix[pidx+1][0] - spix[pidx][1]
                    # must be within ks_m meters in range to merge
                    if (rdel < ks_m) and (rdel < mindel):
                        mindel = rdel
                        mnidx = pidx
                # merge best two returns
                if mnidx is not None:
                    # new range span for testing against neighbors
                    spix[mnidx][1] = spix[mnidx+1][1]
                    # new peak range is range of max contributor
                    if spix[mnidx+1][3] > spix[mnidx][3]:
                        spix[mnidx][2] = spix[mnidx+1][2]
                    # intensity of return is sum of contributors
                    spix[mnidx][3] += spix[mnidx+1][3]
                    # remove one of the two merged
                    del spix[mnidx+1]
                else:
                    done = True
            else:
                done = True
        # now eliminate all but max and last returns
        max_idx = None
        max_val = 0.0
        for ci, pix in enumerate(spix):
            if pix[3] > max_val:
                max_val = pix[3] / int_mult
                max_idx = ci
        # if they are the same, return only one
        if spix[-1][3] >= spix[max_idx][3]:
            return [ spix[-1] ]
        else:
            return [ spix[max_idx], spix[-1] ]

    def sample_range_image(self, rng_img, int_img, limg_shape, vel_cam, pps, ldr_err, pattern):
        # depth image is set up as 512 x 512 and is 62.5 degrees vertical FOV
        # the center row is vertical, but we want to sample from the
        # region corresponding to HDL-32 FOV: from +10 to -30 degrees
        detailed_sensor_model = False
        fwd_vel = vel_cam[1]
        beam_div = 0.002
        lens = self.cam.node().getLens()
        #sx = self.win.getXSize()
        sy = self.win.getYSize()
        ifov_vert = 2.0*math.tan(0.5*math.radians(lens.getVfov()))/float(sy-1)
        #ifov_horz = 2.0*math.tan(0.5*math.radians(lens.getHfov()))/float(sx-1)
        #ifov = math.radians(self.cam.node().getLens().getVfov() / self.win.getYSize())
        sigma = beam_div / ifov_vert
        hs = int(2.0*sigma+1.0)
        gprof = gauss_kern(sigma, hs, normalize=False)
        rimg = np.zeros(limg_shape, dtype=np.float32)       
        iimg = np.zeros(limg_shape, dtype=np.float32)
        margin = 10.0

        for pidx, relation in enumerate(pattern):
            
            # get the usual scan pattern sample
            ldr_row, ldr_col, img_row_flt, img_col_flt = relation
            
            if ((img_row_flt > -margin) and
                (img_col_flt > -margin) and
                (img_row_flt < rng_img.shape[0] + margin) and
                (img_col_flt < rng_img.shape[1] + margin)):
                
                # within reasonable distance from image limits
                img_row = int(round(img_row_flt))
                img_col = int(round(img_col_flt))
                
                # motion compensation
                trng = np.float32(rng_img[img_row,img_col])
                if trng > 0.0:
                    # TODO: change this back to False
                    done = True
                    ic = 0
                    while not done:
                        old_trng = trng
                        del_row = pidx * fwd_vel / (ifov_vert * trng * pps)
                        if (abs(del_row) > 1e-1) and (ic < 10):
                            img_row_f = img_row_flt + del_row
                            img_row = int(round(img_row_f))
                            trng = np.float32(rng_img[img_row,img_col])
                            ic += 1
                            if abs(trng - old_trng) < 0.5:
                                done = True
                        else:
                            done = True
                            
                    # simple sensor processing: just sample from large images
                    rimg[ldr_row,ldr_col] = np.float32(rng_img[img_row,img_col])
                    iimg[ldr_row,ldr_col] = np.float32(int_img[img_row,img_col])

                    if detailed_sensor_model:
                        # detailed model subsamples whole beam width
                        gpatch = copy_patch_centered((img_row,img_col),hs,int_img,0.0)
                        gpatch = np.float32(gpatch)
                        gpatch *= gprof
                        rpatch = copy_patch_centered((img_row,img_col),hs,rng_img,0.0)
                        rpatch = np.squeeze(rpatch)
                        valid = rpatch > 1e-3
                        if np.count_nonzero(valid) > 0:
                            rpatch_ts = rpatch[valid]
                            gpatch_ts = gpatch[valid]
                            returns = self.find_sorted_ladar_returns(rpatch_ts,gpatch_ts,2.5)
                            # for now we just take first return
                            rimg[ldr_row,ldr_col] = returns[0][2]
                            iimg[ldr_row,ldr_col] = returns[0][3]
                else:
                    rimg[ldr_row,ldr_col] = 0.0
                    iimg[ldr_row,ldr_col] = np.float32(int_img[img_row,img_col])

        rimg += ldr_err * np.random.standard_normal(rimg.shape)
        return rimg, iimg
    
    def skysphereTask(self, task):
       if self.base is not None:
           self.skysphere.setPos(self.base.camera, 0, 0, 0)
           self.terrain.generate()
       return task.cont        

def save_xyz_data(fname, rng16, int8, ss_pos):
    pos = None
    old_sf_idx = -1
    xyz_data = np.zeros((rng16.shape[0],rng16.shape[1],4), dtype=np.float32)
    intens = int8.astype(np.float32)
    rng = rng16.astype(np.float32)
    rng /= 64.0
    rng = np.where(rng > 1.0e3, 0.0, rng)
    mask = (rng > 1.0e-3)
    rix, cix = np.where(mask)           
    for a in range(len(rix)):
        ri, ci = rix[a], cix[a]
        subframe_idx = ci // 150
        if subframe_idx != old_sf_idx:
            if subframe_idx in ss_pos.keys():
                if pos is None:
                    print('ss_pos=',ss_pos[subframe_idx])
                pos = ss_pos[subframe_idx]
                # convert to Nav position
                npos = copy(pos)
                npos[0] = pos[1]
                npos[1] = pos[0]
                npos[2] = -pos[2]
            else:
                print('Error: no subframe position for pixel in subframe %i' % subframe_idx)
                npos = None
        old_sf_idx = subframe_idx
        coldel = (ci - 0.5 * (rng.shape[1] - 1))        
        theta = -10.0 - 41.33 * (float(ri) / float(rng.shape[0]-1) - 0.5)
        va = math.radians(theta)
        cva = math.cos(va)
        sva = math.sin(va)
        ha = math.radians(120 + 0.4 * coldel)
        cha = math.cos(ha)
        sha = math.sin(ha)                  
        vec = np.array( [ sva,  cha*cva,  sha*cva ] )
        vec *= rng[ri,ci]
        vec += npos
        xyz_data[ri,ci,0:3] = vec
        xyz_data[ri,ci,3] = intens[ri,ci]
    np.save(fname,xyz_data)
    
def main():
    app = MyApp()

    frames = 60
    start_time = time.time()
    ladar_pps = 4500.0
    ldr_num_pieces = 6
    ldr_piece_az = 360.0 / float(ldr_num_pieces)
    ldr_piece = (32,150)
    ldr_error = 0.0
    curr_pos = np.array( [ -5.0, -15.0, 50.0 ], dtype=np.float32 )
    curr_vel = np.array( [  5.0,  15.0,  0.0 ], dtype=np.float32 )
    pattern = app.compute_sample_pattern(ldr_piece,HORZ_RES_FACTOR)
    for fidx in range(frames):
        cvn = np.linalg.norm(curr_vel)
        curr_direction = curr_vel / cvn
        az = math.atan2(-curr_direction[0], curr_direction[1])
        
        ldr_rng_img = np.zeros((ldr_piece[0],
                                ldr_num_pieces*ldr_piece[1]), dtype=np.uint16)
        ldr_int_img = np.zeros((ldr_piece[0],
                                ldr_num_pieces*ldr_piece[1]), dtype=np.float32)
        
        subscan_pos = {}
        
        for k in [ 1, 2, 3 ]:
            app.cam.setPos(curr_pos[0], curr_pos[1], curr_pos[2])
            app.cam.setHpr(math.degrees(az),0.0,0.0)

            xform = app.cam.getTransform().getMat().getUpper3()
            print(xform)
            xform2 = copy(xform)
            xform2.transposeInPlace()
            img_pos = xform2.xformVec(Vec3(curr_pos[0],curr_pos[1],curr_pos[2]))
            img_vel = xform2.xformVec(Vec3(curr_vel[0],curr_vel[1],curr_vel[2]))
            print('img_pos=',img_pos,', img_vel=',img_vel)

            app.cam.setHpr(math.degrees(az),-90.0,0.0)
            app.cam.setHpr(app.cam, 120.0 - float(k) * ldr_piece_az, 0, 0) 

            subscan_pos[k] = img_pos
            app.graphicsEngine.renderFrame()
            image = app.get_camera_image()
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            depth_image, range_image = app.get_camera_depth_image()
            #write_png('depth_%i_%i.png' % (fidx,k), depth_image.astype(np.uint16))
            #write_png('range_%i_%i.png' % (fidx,k), range_image.astype(np.uint16))
            rng_img, int_img = app.sample_range_image(range_image,
                                                      gray_image,
                                                      ldr_piece,
                                                      img_vel,
                                                      ladar_pps,
                                                      ldr_error,
                                                      pattern)
            rng_vals = 64.0 * rng_img
            
            ldr_rng_img[0:ldr_piece[0],
                        ldr_piece[1]*k:ldr_piece[1]*(k+1)] = rng_vals.astype(np.uint16)
            ldr_int_img[0:ldr_piece[0],
                        ldr_piece[1]*k:ldr_piece[1]*(k+1)] = int_img
            #exit_request = show_rgbd_image(image, range_image, depth_scale=100.0)
            #show_lidar_image(image, ladar_image, range_sc=100.0)
            #if exit_request:
            #    cv2.destroyAllWindows()
            #    app.closeWindow(app.win)
            #    app.destroy()
            #    del app
            #    sys.exit()
            dt = float(ldr_piece[1]) / ladar_pps
            curr_pos += dt * curr_vel

        save_xyz_data('hdl32_xyz_%05i.npy' % fidx, ldr_rng_img, ldr_int_img, subscan_pos)
        lint_mx = ldr_int_img.max()
        lint_mn = ldr_int_img.min()
        lint_sc = 255.0 / (lint_mx-lint_mn)
        lint_img = lint_sc * (ldr_int_img - lint_mn)
        write_png('hdl32_rng_%05i.png' % fidx, ldr_rng_img)
        write_png('hdl32_int_%05i.png' % fidx, lint_img.astype(np.uint8))
        print('simulated frame %i completed' % fidx)

    end_time = time.time()
    print("average FPS: {}".format(frames / (end_time - start_time)))
    cv2.destroyAllWindows()
    app.closeWindow(app.win)
    app.destroy()
    del app
    sys.exit()

if __name__ == '__main__':
    main()
    
    
