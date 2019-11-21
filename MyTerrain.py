


import direct.directbase.DirectStart
from pandac.PandaModules import *
from direct.showbase.DirectObject import DirectObject
from direct.task import Task
import sys

class World(DirectObject):
  def __init__(self):   
    self.accept("escape", sys.exit)        # Escape quits
    base.disableMouse()
    base.camera.setPos(65,10,3)
    base.camera.setHpr(0,0,0)
    
    # Set up the GeoMipTerrain
    self.terrain = GeoMipTerrain("myDynamicTerrain")
    self.terrain.setHeightfield("new_height_map.png")
    self.terrain.setAutoFlatten(GeoMipTerrain.AFMMedium)
    
    # Load some textures
    self.txtGrass = loader.loadTexture("grass.png")
    self.txtbb = loader.loadTexture("tree-color.png", "tree-opacity.png")
    
    # Set terrain properties
    self.terrain.setBlockSize(32)
    self.terrain.setNear(10)
    self.terrain.setFar(100)
    self.terrain.setFocalPoint(base.camera)
    
    # Store the root NodePath for convenience
    self.root = self.terrain.getRoot()
    self.root.reparentTo(render)

    # Create card for billboarding
    crd = CardMaker('mycard')
    crd.setColor(0.5,0.5,0.5,1)
    ll = Point3(-10,0,-10)
    lr = Point3(10,0,-10)
    ur = Point3(10,0,10)
    ul = Point3(-10,0,10)
    crd.setFrame(ll,lr,ur,ul)
    crd.setHasNormals(False)
    crd.setHasUvs(True)
    crd.setUvRange(self.txtbb)
    crdNP = render.attachNewNode(crd.generate())
    crdNP.setBillboardAxis()
    crdNP.setTexture(self.txtbb)
    crdNP.setPos(65,65,2)
    crdNP.setScale(0.1)
    crdNP.setTransparency(TransparencyAttrib.MAlpha)    
    #crdNP.setHpr(0,0,0)
    
    # Set up materials
    self.tmatl = Material()
    self.tmatl.setShininess(5.0)
    self.tmatl.setAmbient(VBase4(0,0,1,1))
#    self.root.setMaterial(self.tmatl)
    self.root.setSz(100)
    self.root.setPos(-64,-64,0)
#    self.root.setTexture(self.txtGrass)
    
    # Set up lighting
    ambientLight = AmbientLight('ambientLight')
    ambientLight.setColor(Vec4(0.1, 0.1, 0.1, 1))
    ambientLightNP = render.attachNewNode(ambientLight)
    render.setLight(ambientLightNP)
    directionalLight = DirectionalLight('directionalLight')
    directionalLight.setColor(Vec4(0.8, 0.8, 0.7, 1))
    directionalLightNP = render.attachNewNode(directionalLight)
    directionalLightNP.setHpr(30,-30,0)
    render.setLight(directionalLightNP)
        
    # Generate it - and regenerate frequently.
    self.terrain.generate()
    taskMgr.add(self.updateTask, "update")
    
  # Add a task to keep updating the terrain
  def updateTask(self, task):
    self.terrain.update()
    return task.cont
      

# Finally, create an instance of our class and start 3d rendering
w = World()
run()
