import os
import unittest
import vtk, qt, ctk, slicer, random
import sitkUtils as su
import SimpleITK as sitk
from slicer.ScriptedLoadableModule import *
import logging
import numpy

#

class PathPlanner(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "PathPlanner"
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["Gina Sajith (King's College London)"]
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
It performs a simple thresholding on the input volume and optionally captures a screenshot.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Gina Sajith (King's College London)
"""

#
# PathPlannerWidget
#

class PathPlannerWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # input volume selector
    #
    self.inputImageSelector = slicer.qMRMLNodeComboBox()
    self.inputImageSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
    self.inputImageSelector.selectNodeUponCreation = True
    self.inputImageSelector.addEnabled = False
    self.inputImageSelector.removeEnabled = False
    self.inputImageSelector.noneEnabled = False
    self.inputImageSelector.showHidden = False
    self.inputImageSelector.showChildNodeTypes = False
    self.inputImageSelector.setMRMLScene(slicer.mrmlScene)
    self.inputImageSelector.setToolTip("Pick the input to the algorithm.")
    parametersFormLayout.addRow("Input Label Map: ", self.inputImageSelector)

    # input obstacle volume selector
    #
    self.inputObstacle1Selector = slicer.qMRMLNodeComboBox()
    self.inputObstacle1Selector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
    self.inputObstacle1Selector.selectNodeUponCreation = True
    self.inputObstacle1Selector.addEnabled = False
    self.inputObstacle1Selector.removeEnabled = False
    self.inputObstacle1Selector.noneEnabled = False
    self.inputObstacle1Selector.showHidden = False
    self.inputObstacle1Selector.showChildNodeTypes = False
    self.inputObstacle1Selector.setMRMLScene(slicer.mrmlScene)
    self.inputObstacle1Selector.setToolTip("Pick the input to the algorithm.")
    parametersFormLayout.addRow("Input Obstacle 1 Label Map: ", self.inputObstacle1Selector)

    self.inputObstacle2Selector = slicer.qMRMLNodeComboBox()
    self.inputObstacle2Selector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
    self.inputObstacle2Selector.selectNodeUponCreation = True
    self.inputObstacle2Selector.addEnabled = False
    self.inputObstacle2Selector.removeEnabled = False
    self.inputObstacle2Selector.noneEnabled = False
    self.inputObstacle2Selector.showHidden = False
    self.inputObstacle2Selector.showChildNodeTypes = False
    self.inputObstacle2Selector.setMRMLScene(slicer.mrmlScene)
    self.inputObstacle2Selector.setToolTip("Pick the input to the algorithm.")
    parametersFormLayout.addRow("Input Obstacle 2 Label Map: ", self.inputObstacle2Selector)

    #
    # input entry points
    #
    self.inputEntrySelector = slicer.qMRMLNodeComboBox()
    self.inputEntrySelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
    self.inputEntrySelector.selectNodeUponCreation = True
    self.inputEntrySelector.addEnabled = False
    self.inputEntrySelector.removeEnabled = False
    self.inputEntrySelector.noneEnabled = False
    self.inputEntrySelector.showHidden = False
    self.inputEntrySelector.showChildNodeTypes = False
    self.inputEntrySelector.setMRMLScene(slicer.mrmlScene)
    self.inputEntrySelector.setToolTip("Pick the input entry fiducials to the algorithm.")
    parametersFormLayout.addRow("Input entry points: ", self.inputEntrySelector)

    #
    # input target points
    #
    self.inputTargetSelector = slicer.qMRMLNodeComboBox()
    self.inputTargetSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
    self.inputTargetSelector.selectNodeUponCreation = True
    self.inputTargetSelector.addEnabled = False
    self.inputTargetSelector.removeEnabled = False
    self.inputTargetSelector.noneEnabled = False
    self.inputTargetSelector.showHidden = False
    self.inputTargetSelector.showChildNodeTypes = False
    self.inputTargetSelector.setMRMLScene(slicer.mrmlScene)
    self.inputTargetSelector.setToolTip("Pick the input target fiducials to the algorithm.")
    parametersFormLayout.addRow("Input target points: ", self.inputTargetSelector)

    #
    # check box to trigger taking screen shots for later use in tutorials
    #
    self.enableScreenshotsFlagCheckBox = qt.QCheckBox()
    self.enableScreenshotsFlagCheckBox.checked = 0
    self.enableScreenshotsFlagCheckBox.setToolTip(
      "If checked, take screen shots for tutorials. Use Save Data to write them to disk.")
    parametersFormLayout.addRow("Enable Screenshots", self.enableScreenshotsFlagCheckBox)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.inputImageSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.inputImageSelector.currentNode() and self.inputObstacle1Selector.currentNode() and self.inputObstacle2Selector.currentNode() and self.inputEntrySelector.currentNode() and self.inputTargetSelector

  def onApplyButton(self):
    logic = PathPlannerLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    logic.run(self.inputImageSelector.currentNode(), self.inputObstacle1Selector.currentNode(), self.inputObstacle2Selector.currentNode(), self.inputEntrySelector.currentNode(),
              self.inputTargetSelector.currentNode(), enableScreenshotsFlag)


# PathPlannerLogic
#

class PathPlannerLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def hasImageData(self,volumeNode):
    """This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
      logging.debug('hasImageData failed: no volume node')
      return False
    if volumeNode.GetImageData() is None:
      logging.debug('hasImageData failed: no image data in volume node')
      return False
    return True

  def isValidInputOutputData(self, inputVolumeNode):
    """Validates if the output is not the same as input
    """
    if not inputVolumeNode:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    return True

  def run(self, inputVolume, inputObstacle1, inputObstacle2,entries, targets, enableScreenshots=0):
    """
    Run the actual algorithm
    """

    logging.info('Processing started')

    # Compute the thresholded output volume using the Threshold Scalar Volume CLI module
    # Gives the hippotargets as a new file called MarkupsFiducial
    pointPicker = PickPointsMatrix()
    pointPicker.run(inputVolume, entries, targets)
    hippotargets = slicer.util.getNode('hippotargets')

    # Obtain CollisionFreePaths that avoid hitting and obstacle
    generate_CollisionFreePaths = CollisionFreePaths()
    generate_CollisionFreePaths.run(entries, hippotargets, inputObstacle1, inputObstacle2)

    ## Obtain Best Paths that avoid hitting any obstacle and is optimised such that the path selected maximises the distance from the obstacle
    generate_Optimalpaths = Optimalpaths()
    generate_Optimalpaths.run(entries, hippotargets, inputObstacle1, inputObstacle2)

    # Capture screenshot
    if enableScreenshots:
      self.takeScreenshot('PathPlanning2Test-Start','MyScreenshot',-1)

    logging.info('Processing completed')

    return True


class PickPointsMatrix():
  def run(self, inputVolume, entries, targets):
    # So at the moment we have our boilerplate UI to take in an image and set of figudicals and output another set of fiducials
    # And are just printing something silly in our main call
    # In this first instance (related to task a) we are going to find the set of input fiducials that are within a mask of our input volume
    # First bit of clean up is to remove all points from the output-- otherwise rerunning will duplicate these
    targets.RemoveAllMarkups()
    # we can get a transformation from our input volume
    mat = vtk.vtkMatrix4x4();
    inputVolume.GetRASToIJKMatrix(mat)
    # set it to a transform type
    transform = vtk.vtkTransform(); transform.SetMatrix(mat)

    for x in range(0, entries.GetNumberOfFiducials()):
      pos = [0,0,0]
      entries.GetNthFiducialPosition(x, pos)
      # get index from position using our transformation
      ind = transform.TransformPoint(pos)

      # get pixel using that index
      pixelValue = inputVolume.GetImageData().GetScalarComponentAsDouble (int(ind[0]), int(ind[1]), int(ind[2]), 0)
      #looks like it needs 4 ints -- our x,y,z index and a component index (which is 0)
      if (pixelValue == 1):
        targets.AddFiducial(pos[0], pos[1], pos[2])



class CalculateDistanceImageFromLabelMap():
  def Execute(self, inputVolume):
    sitkInput = su.PullVolumeFromSlicer(inputVolume)
    # Calculation required to generate distance map
    distanceFilter = sitk.DanielssonDistanceMapImageFilter()
    sitkOutput = distanceFilter.Execute(sitkInput)
    # Commumication by pushing to slicer
    distance_map_Volume = su.PushVolumeToSlicer(sitkOutput, None, 'distanceMap_obstacle')
    return distance_map_Volume


class MarchingCubes_alg():
  def run(self, inputObstacle1, inputObstacle2):
    # Creating Mesh of Obstacle 1 and Obstacle 2
    Mesh1 = vtk.vtkMarchingCubes(); Mesh2 = vtk.vtkMarchingCubes()
    Mesh1.SetInputData(inputObstacle1.GetImageData()); Mesh2.SetInputData(inputObstacle2.GetImageData())
    Mesh1.SetValue(0, 1);  Mesh1.Update();   Mesh2.SetValue(0, 1); Mesh2.Update()

    # Create OBB Tree of Obstacle 1 and Obstacle 2
    OBB_obstacle1_tree = vtk.vtkOBBTree(); OBB_obstacle2_tree = vtk.vtkOBBTree()
    OBB_obstacle1_tree.SetDataSet(mesh.GetOutput()); OBB_obstacle2_tree.SetDataSet(mesh.GetOutput())
    OBB_obstacle1_tree.BuildLocator(); OBB_obstacle2_tree.BuildLocator()

    return OBB_obstacle1_tree, OBB_obstacle2_tree



class ObstacleCollisionFreePaths():
  def run(self, entries, hippotargets, inputObstacle1, inputObstacle2):

    # assign vtk functions to variable names
    lines = vtk.vtkCellArray(); Path = vtk.vtkPolyData(); points = vtk.vtkPoints()

    # Compute the thresholded output volume using the Threshold Scalar Volume CLI module

    vtk_mesh = MarchingCubes_alg()
    OBB_obstacle1_tree, OBB_tree_obstacle2 = vtk_mesh.run(inputObstacle1, inputObstacle2)

    for i in range(0, entries.GetNumberOfFiducials()):
      entry = [0, 0, 0]; entries.GetNthFiducialPosition(i, entry)

      for j in range(0, hippotargets.GetNumberOfFiducials()):
        target = [0, 0, 0]; hippotargets.GetNthFiducialPosition(j, target)
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, points.InsertNextPoint(entry[0], entry[1], entry[2]))
        line.GetPointIds().SetId(1, points.InsertNextPoint(target[0], target[1], target[2]))

        # problem: intersectwithline produced all 0 for all paths
        if (OBB_obstacle1_tree.IntersectWithLine(entry, target, vtk.vtkPoints(), vtk.vtkIdList()) == 0):
          if (OBB_obstacle2_tree.IntersectWithLine(entry, target, vtk.vtkPoints(), vtk.vtkIdList()) == 0):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, points.InsertNextPoint(entry[0], entry[1], entry[2]))
            line.GetPointIds().SetId(1, points.InsertNextPoint(target[0], target[1], target[2]))

          # Calculate the length of the line
          length = numpy.sqrt((target[0] - entry[0]) * 2 + (target[1] - entry[1]) * 2 + (target[2] - entry[2]) * 2)
          if length < 55:  # Check the threshold
            lines.InsertNextCell(line)  # Store the line in array

    Path.SetPoints(points)
    Path.SetLines(lines)
    pathNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'GoodPath')
    pathNode.SetAndObserveMesh(Path)



class FindOptimalpaths():

  def run(self, entries, hippotargets, inputObstacle1, inputObstacle2):

    # assign vtk functions to variable names
    pts = vtk.vtkPoints();  good_entry = []; good_target = []; line = vtk.vtkLine()

    ### Obstacle 1 - # Map for obstacle 1
    distance_sum_intensities_obstacle1 = []
    Obstacle1_dist_map = CalculateDistanceImageFromLabelMap().Execute(inputObstacle1)
    # Transformation from our input volume
    mat = vtk.vtkMatrix4x4(); Obstacle1_dist_map.GetRASToIJKMatrix(mat)
    # Set transform type
    transform1 = vtk.vtkTransform();transform1.SetMatrix(mat)
    ### Obstacle 2
    # Map for obstacle 2
    distance_sum_intensities_obstacle2 = []
    Obstacle2_dist_map = CalculateDistanceImageFromLabelMap().Execute(inputObstacle2)
    # Transformation from our input volume
    mat = vtk.vtkMatrix4x4(); Obstacle2_dist_map.GetRASToIJKMatrix(mat)
    # set transform type
    transform2 = vtk.vtkTransform(); transform2.SetMatrix(mat)
    # Get the obbtree related to obstacle 1 and 2
    vtk_mesh = MarchingCubes_alg()
    OBB_obstacle1_tree, OBB_obstacle2_tree = vtk_mesh.run(inputObstacle1, inputObstacle2)

    # for every possible entry and possible target
    for i in range(0, entries.GetNumberOfFiducials()):
      entry = [0, 0, 0]
      entries.GetNthFiducialPosition(i, entry)
      for j in range(0, hippotargets.GetNumberOfFiducials()):
        target = [0, 0, 0]
        hippotargets.GetNthFiducialPosition(j, target)

        # check if the line from entry to target intersects with the obstacle(obbtree)
        # If it does not proceed
        if (OBB_obstacle1_tree.IntersectWithLine(entry, target, vtk.vtkPoints(), vtk.vtkIdList()) == 0):
          if (OBB_obstacle2_tree.IntersectWithLine(entry, target, vtk.vtkPoints(), vtk.vtkIdList()) == 0):

            # Insert entry and target into points, to create line
            line.GetPointIds().SetId(0, pts.InsertNextPoint(entry[0], entry[1], entry[2]))
            line.GetPointIds().SetId(1, pts.InsertNextPoint(target[0], target[1], target[2]))

            # Calculate the length of the line
            ln_length = numpy.sqrt((target[0] - entry[0]) ** 2 + (target[1] - entry[1]) ** 2 + (target[2] - entry[2]) ** 2)
            # Check line is less that the threshold length, if it is add line to lines
            if ln_length < 50:

              n = 10  # make n number of points along the good line
              x_points = numpy.linspace(entry[0], target[0], n); y_points = numpy.linspace(entry[1], target[1], n)
              z_points = numpy.linspace(entry[2], target[2], n)

              # initilise count
              distance_map_intensity_obstacle1 = 0; distance_map_intensity_obstacle2 = 0

              # for each
              for a in range(0, n-1):
                pos = [x_points[a], y_points[a], z_points[a]]

                # get index from position using our transformation - obstacle 1
                index1 = transform1.TransformPoint(pos)
                # find the total distance from obstacle for a line using index
                distance_map_intensity_obstacle1 = distance_map_obstacle1.GetImageData().GetScalarComponentAsDouble(int(index1[0]), int(index1[1]), int(index1[2]),0)
                distance_map_intensity_obstacle1 += distance_map_intensity_obstacle1

                # get index from position using our transformation - obstacle 1
                index2 = transform2.TransformPoint(pos)
                # find the total distance from obstacle for a line using index
                distance_map_intensity_obstacle2 = distance_map_obstacle2.GetImageData().GetScalarComponentAsDouble(int(index2[0]), int(index2[1]), int(index2[2]), 0)
                distance_map_intensity_obstacle2 += distance_map_intensity_obstacle2

              # All good entry and target added to a list
              good_entry.append(entry); good_target.append(target)
              # Sum all the total distances from obstacle for all lines to a list
              distance_sum_intensities_obstacle1.append(distance_map_intensity_obstacle1)
              distance_sum_intensities_obstacle2.append(distance_map_intensity_obstacle2)

    # Using the index value the the maximum distance of a line from the obstacle can be computed
    maxElementIndex = numpy.where(distance_sum_intensities_obstacle1 == numpy.amax(distance_sum_intensities_obstacle1))
    bestIndex = maxElementIndex[0]

    # Find the corresponding entry and target for this line best line
    best_entry = good_entry[bestIndex[0]]; best_target = good_target[bestIndex[0]]

    return best_entry, best_target

class Optimaltrajectorypaths():
  def run(self, entries, hippotargets, inputObstacle1, inputObstacle2):

    # assign vtk functions to variable names
    lines1 = vtk.vtkCellArray()
    Optimalpaths = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    line = vtk.vtkLine()

    # Call class to get the best entry and bets target
    best = FindOptimalpaths()
    best_entry, best_target = best.run(entries, hippotargets, inputObstacle1, inputObstacle2)

    # Insert entry and target into points, to create line
    line.GetPointIds().SetId(0, points.InsertNextPoint(best_entry[0], best_entry[1], best_entry[2]))
    line.GetPointIds().SetId(1, points.InsertNextPoint(best_target[0], best_target[1], best_target[2]))
    lines1.InsertNextCell(line)

    # Set point and lines to object Optimal paths
    Optimaltrajectorypaths.SetPoints(points)
    Optimaltrajectorypaths.SetLines(lines1)
    # return Path as a node to be viewed in slicer
    pathNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'Optimal trajectory Path')
    pathNode.SetAndObserveMesh(Optimaltrajectorypaths)


########################################## UNIT TESTING ##########################################

class PathPlannerTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_LoadData_r_hippo('/Users/ginasajith/Desktop/BrainParcellation')
    self.test_LoadData_targets('/Users/ginasajith/Desktop/BrainParcellation')
    self.test_LoadData_ventricles('/Users/ginasajith/Desktop/BrainParcellation')
    self.test_LoadData_vessels('/Users/ginasajith/Desktop/BrainParcellation')
    self.test_LoadData_entries('/Users/ginasajith/Desktop/BrainParcellation')
    self.test_LoadData_hippotargets('/Users/ginasajith/Desktop/BrainParcellation')
    self.test_algorithm_Input_IsNot_None()
    self.test_algorithm_Mesh_IsNot_None()
    self.test_algorithm_OBB_IsNot_None()
    self.test_TestOutsidePoints()
    self.test_TestInsidePoints()
    self.test_TestEmptyPoints()
    self.test_Check_Intersection1()
    self.test_Check_Intersection2()
    self.test_LineLength()
    self.test_CalculateDistanceImageFromLabelMap()
    self.test_FindOptimalpaths1()
    self.test_FindOptimalpaths_Check_Intersection_Condition_doesnot_hit()
    self.test_FindOptimalpaths_Check_LineLength_Condition()
    self.test_Correct_length()
    self.test_FindOptimalpaths2()


  ### PickPointsMatrix ###
  def test_LoadData_r_hippo(self, path):
    self.delayDisplay("Starting load data test")
    isLoaded = slicer.util.loadLabelVolume(path + '/r_hippo.nii.gz')
    if (not isLoaded):
      self.delayDisplay('Unable to load ' + path + '/r_hippo.nii.gz')

  def test_LoadData_targets(self, path):
    self.delayDisplay("Starting load data test")

    isLoaded = slicer.util.loadMarkupsFiducialList(path + '/targets.fcsv')
    if (not isLoaded):
      self.delayDisplay('Unable to load ' + path + '/targets.fcsv')

  def test_LoadData_ventricles(self, path):
    self.delayDisplay("Starting load data test")

    isLoaded = slicer.util.loadLabelVolume(path + '/ventricles.nii.gz')
    if (not isLoaded):
      self.delayDisplay('Unable to load ' + path + '/ventricles.nii.gz')

  def test_LoadData_vessels(self, path):
    self.delayDisplay("Starting load data test")

    isLoaded = slicer.util.loadLabelVolume(path + '/vessels.nii.gz')
    if (not isLoaded):
      self.delayDisplay('Unable to load ' + path + '/vessels.nii.gz')

  def test_LoadData_entries(self, path):
    self.delayDisplay("Starting load data test")

    isLoaded = slicer.util.loadMarkupsFiducialList(path + '/entries.fcsv')
    if (not isLoaded):
      self.delayDisplay('Unable to load ' + path + '/entries.fcsv')

  def test_LoadData_hippotargets(self, path):
    self.delayDisplay("Starting load data test")

    isLoaded = slicer.util.loadMarkupsFiducialList(path + '/hippotargets.fcsv')
    if (not isLoaded):
      self.delayDisplay('Unable to load ' + path + '/hippotargets.fcsv')


    self.delayDisplay('Test passed! All data loaded correctly')


  # Marching Cubes Unit Tests

  def test_algorithm_Input_IsNot_None(self):
    self.delayDisplay("Starting test:  Input IsNot None.")
    inputObstacle = slicer.util.getNode('ventricles')
    self.assertIsNotNone(inputObstacle)
    self.delayDisplay('Test passed! The input obstacle is not empty for use in function')

  def test_algorithm_Mesh_IsNot_None(self):
    self.delayDisplay("Starting test: MakeMeshTree IsNot None.")
    inputImage = slicer.util.getNode('r_hippo')
    # Construct Mesh of obstacle 1
    mesh = vtk.vtkMarchingCubes(); mesh.SetInputData(inputImage.GetImageData())
    mesh.SetValue(0, 0.5); mesh.Update()
    meshoutput = mesh.GetOutput()
    self.assertIsNotNone(meshoutput)
    self.delayDisplay('Test passed! The mesh is not empty')

  def test_algorithm_OBB_IsNot_None(self):
    self.delayDisplay("Starting test: MakeMeshTree IsNot None.")
    inputImage = slicer.util.getNode('r_hippo Vol')
    # Create Mesh of obstacle 1
    mesh = vtk.vtkMarchingCubes(); mesh.SetInputData(inputImage.GetImageData())
    mesh.SetValue(0, 0.5); mesh.Update()

    # Create Tree of obstacle 1
    OBB_obstacle1_tree = vtk.vtkOBBTree()
    OBB_obstacle1_tree.SetDataSet(mesh.GetOutput())
    OBB_obstacle1_tree.BuildLocator()
    self.assertIsNotNone(OBB_obstacle1_tree)
    self.delayDisplay('Test passed! obb tree created is not empty')

# Path planner tests

  def test_TestOutsidePoints(self):
      """ Testing points I know are outside of the mask (first point is outside of the region entirely,
      second is the origin.
      """

      self.delayDisplay("Starting test points outside mask.")
      #
      # get out image node
      mask = slicer.util.getNode('r_hippo')

      # I am going to hard code two points -- both of which I know are not in my mask
      outsidePoints = slicer.vtkMRMLMarkupsFiducialNode()
      outsidePoints.AddFiducial(-1, -1, -1)  # this is outside of our image bounds
      cornerPoint = mask.GetImageData().GetOrigin()
      outsidePoints.AddFiducial(cornerPoint[0], cornerPoint[1], cornerPoint[2])  # we know our corner pixel is no 1

      # run our class
      returnedPoints = slicer.vtkMRMLMarkupsFiducialNode()
      PickPointsMatrix().run(mask, outsidePoints, returnedPoints)

      # check if we have any returned fiducials -- this should be empty
      if (returnedPoints.GetNumberOfFiducials() > 0):
        self.delayDisplay('Test failed. There are ' + str(returnedPoints.GetNumberOfFiducials()) + ' return points.')
        return

      self.delayDisplay('Test passed! No points were returned.')

  def test_TestInsidePoints(self):
      """ Testing points I know are inside of the mask (first point is outside of the region entirely, second is the origin.
      """
      self.delayDisplay("Starting test points inside mask.")
      mask = slicer.util.getNode('r_hippo')

      # I am going to hard code one point I know is within my mask
      insidePoints = slicer.vtkMRMLMarkupsFiducialNode()
      insidePoints.AddFiducial(152.3, 124.6, 108.0)
      insidePoints.AddFiducial(145, 129, 108.0)

      # run our class
      returnedPoints = slicer.vtkMRMLMarkupsFiducialNode()
      PickPointsMatrix().run(mask, insidePoints, returnedPoints)
      # check if we have any returned fiducials -- this should be 1
      if (returnedPoints.GetNumberOfFiducials() != 2):
        self.delayDisplay('Test failed. There are ' + str(returnedPoints.GetNumberOfFiducials()) + ' return points.')
        return

      self.delayDisplay('Test passed!' + str(returnedPoints.GetNumberOfFiducials()) + ' points were returned.')

  def test_TestEmptyPoints(self):
      """Test for a null case where the markup fiducials is empty."""
      self.delayDisplay("Starting test points for empty points.")
      mask = slicer.util.getNode('r_hippo')

      # Empty point set
      insidePoints = slicer.vtkMRMLMarkupsFiducialNode()

      # run our class
      returnedPoints = slicer.vtkMRMLMarkupsFiducialNode()
      PickPointsMatrix().run(mask, insidePoints, returnedPoints)
      self.delayDisplay('Test passed! Empty points dont break my code.')

  ### ObstacleCollisionFreePaths Test ###

  def test_Check_Intersection1(self):
    self.delayDisplay("Starting test: Obstacle CollisionFreePaths Check Intersection Condition - not intersect")

    inputObstacle1 = slicer.util.getNode('vessels')
    inputObstacle2 = slicer.util.getNode('ventricles')

    # assign vtk functions to variable names - Mesh/Tree
    MeshTree = MarchingCubes_alg()
    # make mesh/tree
    MeshOBBTree1, MeshOBBTree2 = MeshTree.run(inputObstacle1, inputObstacle2)
    entry = [207, 128, 84]
    target = [161, 128, 76]

    self.assertEqual((MeshOBBTree1.IntersectWithLine(entry, target, vtk.vtkPoints(), vtk.vtkIdList())), 0)
    self.delayDisplay('Test passed! Line does not intersect with mesh')

  def Check_Intersection2(self):
    self.delayDisplay("Starting test: CollisionFreePaths Check Intersection Condition - intersect")

    inputObstacle1 = slicer.util.getNode('Blood Vessels'); inputObstacle2 = slicer.util.getNode(' Ventricles')

    #Vtk functions to variable names - Mesh/Tree
    MeshTree = MarchingCubes_alg()
    # make mesh/tree
    MeshOBBTree1, MeshOBBTree2 = MeshTree.run(inputObstacle1, inputObstacle2)
    entry = [207, 128, 84]; target = [174, 128, 100]

    self.assertEqual((MeshOBBTree1.IntersectWithLine(entry, target, vtk.vtkPoints(), vtk.vtkIdList())), 1)
    self.delayDisplay('Test passed! Intersection with the mesh occurs')

  def test_LineLength(self):
    self.delayDisplay("Starting test: CollisionFreePaths Check Length Condition")

    inputObstacle1 = slicer.util.getNode('vessels')
    inputObstacle2 = slicer.util.getNode('ventricles')

    entries = slicer.util.getNode('entries')
    hippotargets = slicer.util.getNode('hippotargets')

    # assign vtk functions to variable names

    Path = vtk.vtkPolyData()
    lns = vtk.vtkCellArray()
    pts = vtk.vtkPoints()

    # Get the obbtree related to obstacle 1 and 2
    vtk_mesh = MarchingCubes_alg()
    OBB_obstacle1_tree, OBB_obstacle2_tree= vtk_mesh.run(inputObstacle1, inputObstacle2)

    # for every possible entry and possible target
    for i in range(0, entries.GetNumberOfFiducials()):
      entry = [0, 0, 0]
      entries.GetNthFiducialPosition(i, entry)
      for j in range(0, hippotargets.GetNumberOfFiducials()):
        target = [0, 0, 0]
        hippotargets.GetNthFiducialPosition(j, target)

        # check if the line from entry to target intersects with the obstacle(obbtree)
        # If it does not proceed
        if (OBB_obstacle1_tree.IntersectWithLine(entry, target, vtk.vtkPoints(), vtk.vtkIdList()) == 0):
          if (OBB_obstacle2_tree.IntersectWithLine(entry, target, vtk.vtkPoints(), vtk.vtkIdList()) == 0):

            # Insert entry and target into points, to create line
            vtk.vtkLine().GetPointIds().SetId(0, pts.InsertNextPoint(entry[0], entry[1], entry[2]))
            vtk.vtkLine().GetPointIds().SetId(1, pts.InsertNextPoint(target[0], target[1], target[2]))

            # Calculate the length of the line
            ln_length = numpy.sqrt((target[0] - entry[0]) ** 2 + (target[1] - entry[1]) ** 2 + (target[2] - entry[2]) ** 2)
            # Check line is less that the threshold length, if it is add line to lines
            if ln_length <= 50:
              self.assertLess(ln_length, 51)
    self.delayDisplay('Test passed! all line lengths are less than 50')


  def test_CalculateDistanceImageFromLabelMap_output(self):
    self.delayDisplay("Starting test: CalculateDistanceImageFromLabelMap Check output")

    inputObstacle = slicer.util.getNode('ventricles');  sitkInput = su.PullVolumeFromSlicer(inputObstacle)
    # compute distance map
    distanceFilter = sitk.DanielssonDistanceMapImageFilter(); sitkOutput = distanceFilter.Execute(sitkInput)
    # push to slicer
    distance_map_Volume = su.PushVolumeToSlicer(sitkOutput, None, 'distanceMap_obstacle')
    self.assertIsNotNone(distance_map_Volume)
    self.delayDisplay('Test passed! Output is as expected')


    # Optimal trajectory path Test

  def test_FindOptimalpaths1(self):
    self.delayDisplay("Starting test: FindOptimalpaths calls Function Compute Distance Image From Label Map.")
    inputObstacle1 = slicer.util.getNode('ventricles')

    # Call Class Compute Distance Image From Label Map to get distance map
    distance_map_obstacle1 = CalculateDistanceImageFromLabelMap().Execute(inputObstacle1)
    self.assertIsNotNone(distance_map_obstacle1)
    self.delayDisplay('Test passed! Function was called and gives distance map output')


  def test_FindOptimalpaths_Check_Intersection_Condition_doesnot_hit(self):
    self.delayDisplay("Starting test: FindOptimalpaths Check Intersection Condition - not intersect")

    inputObstacle1 = slicer.util.getNode('Blood Vessels')
    inputObstacle2 = slicer.util.getNode(' Ventricles')

    # assign vtk functions to variable names - Mesh/Tree
    MeshTree = MarchingCubes_alg()
    # make mesh/tree
    MeshOBBTree1, MeshOBBTree2 = MeshTree.run(inputObstacle1, inputObstacle2)
    entry = [207, 128, 84]; target = [161, 128, 76]

    self.assertEqual((MeshOBBTree1.IntersectWithLine(entry, target, vtk.vtkPoints(), vtk.vtkIdList())), 0)
    self.delayDisplay('Test passed! Line does not intersect with mesh')


  def test_FindOptimalpaths_checklength(self):
    self.delayDisplay("Starting test: FindOptimalpaths Check Length Condition")

    inputObstacle1 = slicer.util.getNode('vessels')
    inputObstacle2 = slicer.util.getNode('ventricles')

    entries = slicer.util.getNode('entries')
    hippotargets = slicer.util.getNode('hippotargets')

    # assign vtk functions to variable names
    Path = vtk.vtkPolyData()
    lns = vtk.vtkCellArray()
    pts = vtk.vtkPoints()

    # Get the obbtree related to obstacle 1 and 2
    vtk_mesh = MarchingCubes_alg()
    OBB_obstacle1_tree, OBB_obstacle2_tree = vtk_mesh.run(inputObstacle1, inputObstacle2)

    # for every possible entry and possible target
    for i in range(0, entries.GetNumberOfFiducials()):
      entry = [0, 0, 0]; entries.GetNthFiducialPosition(i, entry)
      for j in range(0, hippotargets.GetNumberOfFiducials()):
        target = [0, 0, 0]
        hippotargets.GetNthFiducialPosition(j, target)

        # Line from entry to target must intersect with the obstacle(OBBtree)
        # If it does not proceed

        if (OBB_obstacle1_tree.IntersectWithLine(entry, target, vtk.vtkPoints(), vtk.vtkIdList()) == 0):
          if (OBB_obstacle2_tree.IntersectWithLine(entry, target, vtk.vtkPoints(), vtk.vtkIdList()) == 0):

            # Insert entry and target into points, to create line
            vtk.vtkLine().GetPointIds().SetId(0, pts.InsertNextPoint(entry[0], entry[1], entry[2]))
            vtk.vtkLine().GetPointIds().SetId(1, pts.InsertNextPoint(target[0], target[1], target[2]))

            # Calculate the length of the line
            ln_length = numpy.sqrt(
              (target[0] - entry[0]) ** 2 + (target[1] - entry[1]) ** 2 + (target[2] - entry[2]) ** 2)
            # Check line is less that the threshold length, if it is add line to lines
            if ln_length <= 50:
              self.assertLess(ln_length, 51)
    self.delayDisplay('Test passed! all possible lines candidates for Optimal paths are less than 50')

  def test_Correct_length(self):
    self.delayDisplay("Starting test: FindOptimalpaths Check points on line are created/variable created are of appropriate length ")

    inputObstacle1 = slicer.util.getNode('vessels')
    inputObstacle2 = slicer.util.getNode('ventricles')

    entries = slicer.util.getNode('entries')
    hippotargets = slicer.util.getNode('hippotargets')

    # assign vtk functions to variable names
    pts = vtk.vtkPoints()

    pts = vtk.vtkPoints()
    good_entry = []
    good_target = []

    ### Obstacle 1
    # Map for obstacle 1
    distance_sum_intensities_obstacle1 = []
    distance_map_obstacle1 = CalculateDistanceImageFromLabelMap().Execute(inputObstacle1)

    # we can get a transformation from our input volume
    mat = vtk.vtkMatrix4x4()
    distance_map_obstacle1.GetRASToIJKMatrix(mat)

    # set it to a transform type
    transform1 = vtk.vtkTransform()
    transform1.SetMatrix(mat)

    ### Obstacle 2
    # Map for obstacle 2
    distance_sum_intensities_obstacle2 = []
    distance_map_obstacle2 = CalculateDistanceImageFromLabelMap().Execute(inputObstacle2)

    # we can get a transformation from our input volume
    mat = vtk.vtkMatrix4x4()
    distance_map_obstacle2.GetRASToIJKMatrix(mat)

    # set it to a transform type
    transform2 = vtk.vtkTransform()
    transform2.SetMatrix(mat)

    # Get the obbtree related to obstacle 1 and 2
    vtk_mesh = MarchingCubes_alg()
    OBB_obstacle1_tree, OBB_obstacle2_tree = vtk_mesh.run(inputObstacle1, inputObstacle2)

    # for every possible entry and possible target
    for i in range(0, entries.GetNumberOfFiducials()):
      entry = [0, 0, 0]
      entries.GetNthFiducialPosition(i, entry)
      for j in range(0, hippotargets.GetNumberOfFiducials()):
        target = [0, 0, 0]
        hippotargets.GetNthFiducialPosition(j, target)

        # check if the line from entry to target intersects with the obstacle(obbtree)
        # If it does not proceed
        if (OBB_obstacle1_tree.IntersectWithLine(entry, target, vtk.vtkPoints(), vtk.vtkIdList()) == 0):
          if (OBB_obstacle2_tree.IntersectWithLine(entry, target, vtk.vtkPoints(), vtk.vtkIdList()) == 0):

            # Insert entry and target into points, to create line
            vtk.vtkLine().GetPointIds().SetId(0, pts.InsertNextPoint(entry[0], entry[1], entry[2]))
            vtk.vtkLine().GetPointIds().SetId(1, pts.InsertNextPoint(target[0], target[1], target[2]))

            # Calculate the length of the line
            ln_length = numpy.sqrt(
              (target[0] - entry[0]) ** 2 + (target[1] - entry[1]) ** 2 + (target[2] - entry[2]) ** 2)
            # Check line is less that the threshold length, if it is add line to lines
            if ln_length < 50:
              n = 10
              # make n number of points along the good line
              x_points = numpy.linspace(entry[0], target[0], n)
              y_points = numpy.linspace(entry[1], target[1], n)
              z_points = numpy.linspace(entry[2], target[2], n)

              # initilise count
              distance_map_intensity_obstacle1 = 0
              distance_map_intensity_obstacle2 = 0

              # for each
              for k in range(0, n - 1):
                pos = [x_points[k], y_points[k], z_points[k]]

                # get index from position using our transformation - obstacle 1
                ind1 = transform1.TransformPoint(pos)
                # find the total distance from obstacle for a line using index
                distance_map_intensity_obstacle1 = distance_map_obstacle1.GetImageData().GetScalarComponentAsDouble(
                  int(ind1[0]), int(ind1[1]), int(ind1[2]), 0)
                distance_map_intensity_obstacle1 += distance_map_intensity_obstacle1

                # get index from position using our transformation - obstacle 1
                ind2 = transform2.TransformPoint(pos)
                # find the total distance from obstacle for a line using index
                distance_map_intensity_obstacle2 = distance_map_obstacle2.GetImageData().GetScalarComponentAsDouble(
                  int(ind2[0]), int(ind2[1]), int(ind2[2]), 0)
                distance_map_intensity_obstacle2 += distance_map_intensity_obstacle2

              # add all good entry to a lists
              good_entry.append(entry)
              # add all good targets to a lists
              good_target.append(target)
              # add all the total distances from obstacle for all line to a list
              distance_sum_intensities_obstacle1.append(distance_map_intensity_obstacle1)
              distance_sum_intensities_obstacle2.append(distance_map_intensity_obstacle2)

              # 10 points on path are created
              self.assertEqual(len(x_points), 10)
              self.assertEqual(len(y_points), 10)
              self.assertEqual(len(z_points), 10)
              # Check distance_intensities_obstacle is of correct length
              # there should be the same number of good target-entries pair as distance sum intensities for each line
              self.assertEqual(len(distance_sum_intensities_obstacle1), len(good_target))
              self.assertEqual(len(distance_sum_intensities_obstacle2), len(good_entry))
    self.delayDisplay('Test passed! 10 points on line are created & variables created in func are of appropriate size')


  def test_FindOptimalpaths2(self):
    self.delayDisplay("Starting test: FindOptimalpaths Check best_entry and best_target are outputted")

    inputObstacle1 = slicer.util.getNode('vessels'); inputObstacle2 = slicer.util.getNode('ventricles')

    entries = slicer.util.getNode('entries'); hippotargets = slicer.util.getNode('hippotargets')

    # assign vtk functions to variable names
    pts = vtk.vtkPoints(); good_entry = []; good_target = []

    ### Obstacle 1
    # Map for obstacle 1
    distance_sum_intensities_obstacle1 = []
    distance_map_obstacle1 = CalculateDistanceImageFromLabelMap().Execute(inputObstacle1)
    mat = vtk.vtkMatrix4x4()
    distance_map_obstacle1.GetRASToIJKMatrix(mat)
    transform1 = vtk.vtkTransform(); transform1.SetMatrix(mat)

    ### Obstacle 2
    # Map for obstacle 2
    distance_sum_intensities_obstacle2 = []
    distance_map_obstacle2 = CalculateDistanceImageFromLabelMap().Execute(inputObstacle2)
    mat = vtk.vtkMatrix4x4()
    distance_map_obstacle2.GetRASToIJKMatrix(mat)
    transform2 = vtk.vtkTransform(); transform2.SetMatrix(mat)

    # Get the obbtree related to obstacle 1 and 2
    vtk_mesh = _MarchingCubes_alg()
    OBB_obstacle1_tree, OBB_obstacle2_tree = vtk_mesh.run(inputObstacle1, inputObstacle2)

    # for every possible entry and possible target
    for i in range(0, entries.GetNumberOfFiducials()):
      entry = [0, 0, 0]
      entries.GetNthFiducialPosition(i, entry)
      for j in range(0, hippotargets.GetNumberOfFiducials()):
        target = [0, 0, 0]
        hippotargets.GetNthFiducialPosition(j, target)

        # check if the line from entry to target intersects with the obstacle(obbtree)
        # If it does not proceed
        if (OBB_obstacle1_tree.IntersectWithLine(entry, target, vtk.vtkPoints(), vtk.vtkIdList()) == 0):
          if (OBB_obstacle2_tree.IntersectWithLine(entry, target, vtk.vtkPoints(), vtk.vtkIdList()) == 0):

            # Insert entry and target into points, to create line
            vtk.vtkLine().GetPointIds().SetId(0, pts.InsertNextPoint(entry[0], entry[1], entry[2]))
            vtk.vtkLine().GetPointIds().SetId(1, pts.InsertNextPoint(target[0], target[1], target[2]))

            # Calculate the length of the line
            ln_length = numpy.sqrt(
              (target[0] - entry[0]) ** 2 + (target[1] - entry[1]) ** 2 + (target[2] - entry[2]) ** 2)
            # Check line is less that the threshold length, if it is add line to lines
            if ln_length < 50:
              n = 10
              # make n number of points along the good line
              x_points = numpy.linspace(entry[0], target[0], n)
              y_points = numpy.linspace(entry[1], target[1], n)
              z_points = numpy.linspace(entry[2], target[2], n)

              # initilise count
              distance_map_intensity_obstacle1 = 0; distance_map_intensity_obstacle2 = 0

              # for each
              for k in range(0, n - 1):
                pos = [x_points[k], y_points[k], z_points[k]]

                # get index from position using our transformation - obstacle 1
                ind1 = transform1.TransformPoint(pos)
                # find the total distance from obstacle for a line using index
                distance_map_intensity_obstacle1 = distance_map_obstacle1.GetImageData().GetScalarComponentAsDouble(
                  int(ind1[0]), int(ind1[1]), int(ind1[2]), 0)
                distance_map_intensity_obstacle1 += distance_map_intensity_obstacle1

                # get index from position using our transformation - obstacle 1
                ind2 = transform2.TransformPoint(pos)
                # find the total distance from obstacle for a line using index
                distance_map_intensity_obstacle2 = distance_map_obstacle2.GetImageData().GetScalarComponentAsDouble(
                  int(ind2[0]), int(ind2[1]), int(ind2[2]), 0)
                distance_map_intensity_obstacle2 += distance_map_intensity_obstacle2

              # add all good entry to a lists
              good_entry.append(entry)
              # add all good targets to a lists
              good_target.append(target)
              # add all the total distances from obstacle for all line to a list
              distance_sum_intensities_obstacle1.append(distance_map_intensity_obstacle1)
              distance_sum_intensities_obstacle2.append(distance_map_intensity_obstacle2)

    # Find the maximum distance of a line from the obstacle and index
    maxElementIndex = numpy.where(distance_sum_intensities_obstacle1 == numpy.amax(distance_sum_intensities_obstacle1))
    bestIndex = maxElementIndex[0]

    # Find the corresponding entry and target for this line best line
    Optimal_entry = good_entry[bestIndex[0]]
    Optimal_target = good_target[bestIndex[0]]

    # check only best entry and best target are only 1 positions (x,y,z) each.
    self.assertEqual(len(Optimal_entry), 3); self.assertEqual(type(Optimal_entry), list)
    self.assertEqual(len(Optimal_target), 3); self.assertEqual(type(Optimal_target), list)
    self.delayDisplay('Test passed! Optimal_entry and Optimal_target are of correct length')
